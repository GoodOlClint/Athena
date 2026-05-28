# M49 — StructuredIndex cache (pre-prefill setup-gap fix)

Planning doc captured 2026-05-28 after the consuming application shipped a schema
change that exposed a ~60-second per-request CPU gap between request
entry and the first prefill chunk — the time outlines-core spends
compiling the JSON-schema DFA inside `StructuredIndex`.

## Status

| Slice | Tag | Date |
|---|---|---|
| M49.1 — StructuredIndex cache | v0.10.73 | 2026-05-28 |

- ✅ **M49.1** — v0.10.73 — `cachedStructuredIndex` field on
  `MLXLLMModule` keyed by schema-JSON, lookup+build hoisted outside
  the main `container.perform { ... }` block, captured as an
  immutable `let` into the worker closure. `StructuredIndex` marked
  `@unchecked Sendable` (DFA is read-only post-construction).
  Invalidated at all 5 sites that nil `cachedVocabTokens` (unload,
  allowlist add/drop, rebind, pre-pull). Cache hit logs `.debug`;
  miss logs `.notice`. Pure unit tests pin the shared-index +
  independent-walker contract:
  `testSharedIndexProducesIndependentWalkers`,
  `testSharedIndexSurvivesWalkerDeinit`,
  `testStructuredIndexIsSendableForCrossActorCapture`.
- ⏸ **M49.2** — deferred. Phase-aware heartbeat (`phase=setup|
  prefill|decode`). Would have made the setup-gap diagnosis
  self-evident, but with M49.1 the gap is gone; hold unless another
  setup-class surprise lands.

Diagnosed from a live `/usr/bin/log stream` capture of two
consecutive structured requests on v0.10.72:

```
elapsed=15s  tokens=0   (no prefill field — DFA still compiling)
elapsed=20s  tokens=0
... ~50 seconds of CPU-bound DFA compilation ...
elapsed=62s  tokens=0
elapsed=68s  prefill=4/22  tokens=0      ← prefill starts
elapsed=83s  prefill=22/22 tokens=109    ← decode starts at ~83s wall
elapsed=88s  prefill=22/22 tokens=241 tokens_per_sec=25.9
```

Operator confirmed the gaps did NOT exist before the consuming application's latest
schema change — likely the addition of a `maxItems` constraint on a
nested array, which forces outlines-core's DFA to compile a counting
structure that materially blows up the state space for an 11 KB schema
with `$ref` chains.

## Trigger

the consuming application's full-corpus eval works on v0.10.72, but every email
request now eats ~60s of pre-prefill setup. Compounded across a
9-file corpus, that's ~9 minutes of pure DFA-recompile waste before
any GPU activity. The schema is identical across all requests in
the consuming application's workload — same extraction schema, every email.

## Root cause

`MLXLLMModule.runSpeculative` calls `makeGuide()` inside
`container.perform { ... }` on every structured request. `makeGuide()`
constructs a fresh `StructuredIndex` via
`StructuredIndex(jsonSchema:vocabulary:)`, which calls outlines-core's
`oc_index_from_schema` Rust FFI to compile the schema into a DFA.

The compiled DFA is a function of `(schemaJSON, vocabulary)` only —
both are stable across requests when the operator-loaded model is
fixed and the consumer reuses the same schema. **The compile is
recomputed every request, throwing away an identical DFA each time.**

`vocabTokens` is already cached on `MLXLLMModule` for exactly this
reason (the ~150k `tokenizer.decode` calls). `StructuredIndex` was not
included in that cache when M3 shipped because the per-request cost
was originally small. the consuming application's `maxItems` addition pushed the
compile from ~milliseconds to ~60s; the per-request waste is now
material.

## Goal

Cache the compiled `StructuredIndex` by schema-JSON string. First
request after model load/rebind pays the full DFA compile; every
subsequent request with the same schema picks up the cached index in
<1 ms. The per-request `StructuredGuide` (stateful walker) is still
constructed fresh — it has rollback/advance state that MUST be
per-request — but it wraps the shared `StructuredIndex` instead of
recompiling.

## Implementation surface

### `MLXLLMModule.swift`

A single private field next to `cachedVocabTokens`:

```swift
/// M49 — cached compiled StructuredIndex keyed by schema-JSON.
/// outlines-core's DFA compile (`oc_index_from_schema`) is a pure
/// function of `(schemaJSON, vocabulary)`; both are stable across
/// requests for a fixed model + consumer schema. Compile time scales
/// with schema complexity — the consuming application's 11 KB extraction schema
/// with maxItems takes ~60 s to compile, paid once instead of every
/// request. Invalidated on rebind (alongside cachedVocabTokens —
/// rebind changes the vocabulary, so the cached DFA is no longer
/// valid).
private var cachedStructuredIndex:
    (schemaJSON: String, index: StructuredIndex)?
```

The cache lookup in `makeGuide()`:

```swift
func makeGuide() throws -> StructuredGuide? {
    guard let schemaJSON, let vt = vocabTokens else { return nil }
    let index: StructuredIndex
    if let hit = cachedStructuredIndex,
       hit.schemaJSON == schemaJSON {
        index = hit.index
        Self.log.debug("structured-index cache hit ...")
    } else {
        let vocab = StructuredVocabulary(
            tokens: vt.tokens, eosTokenId: vt.eos)
        index = try StructuredIndex(
            jsonSchema: schemaJSON, vocabulary: vocab)
        cachedStructuredIndex = (schemaJSON, index)
        Self.log.notice("structured-index cache miss ...")
    }
    let g = try StructuredGuide(index: index)
    g.openerAlias = vt.opener
    return g
}
```

`makeGuide()` runs inside `container.perform { ... }`, which is
actor-isolated, so mutating `cachedStructuredIndex` is safe without
extra synchronization.

### Cache invalidation sites

Every site that nils `cachedVocabTokens` also nils
`cachedStructuredIndex`. There are 6 such sites:

1. `unload()` (line ~232) — slot unloaded
2. `setAllowlist(...)` (line ~280) — allowlist changed
3. `removeFromAllowlist(...)` (line ~295) — model dropped
4. `rebind(_:)` (line ~329) — model swapped
5. `prePullModel(_:)` (line ~353) — new model prepped
6. (no 6th — re-check)

### Sendable / safety

`StructuredIndex` is a `final class` holding an `OpaquePointer` to
outlines-core's compiled DFA + a `StructuredVocabulary` reference.
After construction the DFA is immutable; `StructuredGuide` reads it
concurrently with no shared mutable state. Storing it on the actor's
isolated state is sound. No `Sendable` annotation needed (it stays
on the actor and is only read through `container.perform`).

## Tests

### Pure unit (CI-safe)

The cache logic itself is pure: same schemaJSON → reuse; different
schemaJSON → replace. Can test with a stub StructuredIndex factory
(no real outlines-core needed):

```swift
// Test the cache key match + replacement on schema change.
let cache = StructuredIndexCache()
let i1 = cache.getOrBuild(schemaJSON: "{\"x\":1}", build: { stub1 })
let i2 = cache.getOrBuild(schemaJSON: "{\"x\":1}", build: { fail() })
XCTAssertIdentical(i1, i2)  // cache hit
let i3 = cache.getOrBuild(schemaJSON: "{\"x\":2}", build: { stub2 })
XCTAssertNotIdentical(i1, i3)  // cache miss on different schema
```

If lifting a `StructuredIndexCache` type out is over-engineering for
a single use site, the test can sit on a small `private` cache helper
inside the module — still pure logic.

### Heavy (gated, manual host-bound tier)

The win is observable end-to-end via the M48.1 + M48.4 heartbeat:
first structured request to a fresh daemon takes ~83 s to first token;
second request with the same schema should take ~20 s to first token
(the saved ~60 s is the DFA recompile cost).

No automated assertion — operator confirms with the heartbeat output
the same way M48.3 was validated.

## Risks

| Risk | Mitigation |
|---|---|
| Stale cache after a Guide-internal change | The `StructuredIndex` is immutable post-construction; `StructuredGuide` has its own per-request state. Reuse is safe by construction. |
| Memory growth on schema-heavy workloads | Single-entry cache (last schema wins). Schema-switching consumers pay the recompile on every switch but never accumulate more than one cached DFA. Could grow to LRU(N) if a real workload demands it; out of scope for M49.1. |
| Schema-JSON byte-equality miss on whitespace differences | The cache key is exact string match. A consumer that serializes the same schema with different whitespace would miss. Acceptable — the consuming application's broker serializes consistently. If a real consumer hits this, canonicalize the key (sort keys, strip whitespace). Out of scope. |
| Cache outlives a model rebind | Explicit invalidation at every site that nils `cachedVocabTokens`. Mirror exactly. |
| Wrong vocabulary in cached index after rebind | Same invalidation sites handle it. Each cached entry is implicitly tied to the model resident at compile time; rebind nils both. |

## Sequencing within M49

1. **M49.1** — `StructuredIndex` cache + invalidation + cache hit/miss
   log + pure unit test. Single ship.
2. **M49.2** (optional, defer) — phase-aware heartbeat (`phase=setup`
   / `phase=prefill` / `phase=decode`). Observability-only. Would have
   made M49.1's diagnosis self-evident at the time, but with the cache
   in place the setup gap collapses to <1 s; the phase indicator is
   no longer load-bearing. Hold unless we hit another setup-class
   surprise.

## Definition of done for M49.1

- [ ] `cachedStructuredIndex` field on `MLXLLMModule`, mirroring
      `cachedVocabTokens` lifecycle exactly.
- [ ] `makeGuide()` reuses cached index when `schemaJSON` matches.
- [ ] `.notice`-level log on cache miss (one line per fresh schema),
      `.debug` on hit (quiet on the hot path).
- [ ] Cache nil'd at every site that nils `cachedVocabTokens`.
- [ ] Pure unit test asserts hit/replace semantics.
- [ ] Operator confirms the heartbeat shows the second structured
      request to a fresh daemon starts prefill within ~5 s of request
      entry instead of ~65 s.
- [ ] `MEMORY.md` updated with `project_m49-structured-index-cache.md`.

## Out of scope (deferred follow-ups)

- LRU cache (N>1). Single-entry suffices for the consuming application's workload.
- Schema-key canonicalization (whitespace-insensitive).
- M49.2 phase-aware heartbeat.
- Cross-process / persistent DFA cache (would require serializing
  the outlines-core DFA, which isn't a supported operation today).
