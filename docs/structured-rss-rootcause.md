# Structured-output RSS blowup — root cause (2026-05-29)

## TL;DR

The unbounded RSS growth is the **outlines-core DFA construction**
(`Index::new`, in the Rust shim) running out of bounded memory. The
driver is **a `maxItems`-bounded OUTER array whose item subschema is
large**: outlines-core unrolls the item subschema once per allowed
position. For the the consuming application schema that is `events: {maxItems: 30}`
over a fat `EventExtraction` object → the per-event DFA is duplicated 30×,
and `Index::new` allocates the cross-product against the full model vocab.

This is **not** a version regression (the structured-compile binary is
identical v0.10.74→0.80) and **not** related to the inner arrays the
operator tried to bound. It is a property of the schema shape + the
pinned outlines-core 0.2.14.

## Method

Pure-CPU harness `rust-shim/examples/schema_bench.rs` calls the exact
compile path Athena uses — `json_schema::regex_from_str` then
`Index::new(regex, vocab)` — against a synthetic Qwen-sized vocabulary.
No MLX, no model, no daemon. Run under `/usr/bin/time -l` for peak RSS,
`timeout` to bound runaway compiles. Synthetic vocab = 15 000 tokens
(1/10th of Qwen's ~151 936) for fast iteration; cost is ~linear in vocab
(see linearity check), so multiply RSS by ~10 to project to production.

`regex_from_str` is trivial for every variant (≈1 ms, ~19.7 KB regex) —
the `{0,N}` quantifiers stay compact in the regex *string*; they only
unroll into states inside `Index::new`. So regex length is NOT a useful
gate signal.

## Evidence — ablation table (synthetic vocab = 15 000)

| variant                       | events maxItems | inner maxItems | Index::new | peak RSS |
|-------------------------------|-----------------|----------------|-----------:|---------:|
| bounded (operator's repro)    | 30              | 10             |     48.4 s | 16.2 GB  |
| flatten-nullable              | 30              | 10             |     47.5 s | 16.2 GB  |
| inner2                        | 30              | 2              |     41.5 s | 15.3 GB  |
| unbounded-inner               | 30              | ∞ (none)       |     42.0 s | 15.3 GB  |
| **outer2**                    | **2**           | 10             |  **3.4 s** | **1.17 GB** |
| **fully-unbounded**           | **∞ (none)**    | ∞              |  **3.0 s** | **1.11 GB** |
| fatouter-thinevent            | 30 (event={title}) | —           |     5.7 s  | 1.98 GB  |
| outer10                       | 10              | 10             |     16.4 s | 5.40 GB  |
| outer20                       | 20              | 10             |     32.4 s | 10.7 GB  |

### What the table proves

1. **Inner arrays are irrelevant.** bounded(10) = 16.2 GB,
   unbounded-inner(∞) = 15.3 GB, inner2(2) = 15.3 GB. Bounding or
   unbounding the 5 inner arrays changes nothing. The operator's "fix"
   (adding `maxItems:10` to the inner arrays) addressed the wrong axis —
   which is exactly why it slipped the M49.5 gate AND didn't help.

2. **The OUTER `events` bound is the driver.** Drop it (fully-unbounded)
   or shrink it (outer2) → 1.1 GB / 3 s. Restore it to 30 → 16 GB / 48 s.

3. **It's outer-count × per-item SIZE, not outer-count alone.**
   events=30 with a thin `{title}` event = 1.98 GB; events=30 with the
   full fat event = 16.2 GB.

4. **Linear in outer count N** (full event): N=2→1.17, N=10→5.4,
   N=20→10.7, N=30→16.2 GB. Slope ≈ 0.53 GB per event-position @15k vocab.

5. **Linear in vocab:** fully-unbounded 1.11 GB @15k → 1.90 GB @30k.

### Production projection

events=30 full-event @ 15k vocab = 16.2 GB. Real Qwen vocab is ~151k
(≈10×) → **~150–160 GB**. This matches the operator's 200 GB peak and the
v0.10.82 snapshot (`rss - resident - mlx_cache = 58.7 GB` mid-build, before
completion). The "unaccounted" 58.7 GB **is** the outlines-core `Index`
under construction: Rust uses the system allocator, so it lands in
MALLOC/RSS, is invisible to the MLX per-module governor (which sums only
MLX arrays), and is not `mlx_cache`. Pool identified.

## Why there is no version regression

- `v0.10.74..v0.10.80` touches only: M50 `clearCache` calls in
  transcription / speaker-embedding / diarization / triattention /
  vectorstore (all audio/vector modules, none in a chat structured-output
  decode path — and TriAttention is MTP-inert), docs, tests, and the
  appVersion bump.
- outlines-core is pinned at 0.2.14; `rust-shim/Cargo.lock` and the
  committed `libathena_structured_shim.a` are unchanged since 2026-05-16;
  `Package.resolved` for the structured path is unchanged.
- Therefore the structured-compile behaviour is byte-identical 74→80. The
  variable that changed between the "90 s / 40 GB success" and the
  "unbounded blowup" runs is **the schema**, not Athena.
- 40 GB at full vocab ≈ events bound of ~8 by the linear model
  (≈5 GB/event-position × 8). The most likely explanation: the v0.10.74
  request used a smaller `events.maxItems` and/or a leaner `EventExtraction`
  object, and the schema has since grown. **Open question for the operator:
  do you still have the exact v0.10.74 request body?** If `events.maxItems`
  was smaller or the event object was leaner, the paradox is fully resolved
  with no Athena regression.

## Why M49.1 caching does not save us

The compiled `Index` is cached per schema-JSON — but the *first* compile
never completes (it OOMs or hits the M33.1 540 s deadline at ~150 GB), so
there is nothing to cache. The transient construction peak is the killer.

## Why the M49.5 gate is mis-modelled

`SchemaComplexity` counts "unbounded inner arrays nested inside a bounded
outer." The data shows inner-array boundedness has ~zero effect on the
cost. The gate happens to refuse the original schema (which had ≥3
unbounded inner arrays under the bounded outer), but for the wrong reason,
and its remedy #1 ("add `maxItems` to the inner arrays") is actively
wrong — it's exactly what the operator did, and it neither helped nor
tripped the gate again. Remedy #2 ("remove `maxItems` from the outer
array and validate post-decode") is the only correct advice in the list.

## Proposed directions (not yet implemented — awaiting decision)

1. **Re-key the gate on the real axis.** Estimate per-item subschema cost
   (fields × enum branches × nested arrays × string length classes) and
   gate on `outerMaxItems × perItemCost`. This refuses BOTH failing
   schemas and correctly passes the cheap shapes (unbounded outer, thin
   item, small N). Fix the remedy text: tell the caller to **remove
   `maxItems` from the outer array** (and post-validate the count), not to
   bound the inner arrays.

2. **Strip-and-post-validate (better UX).** Before compile, treat a large
   bounded outer array as unbounded for the guide (cheap self-loop),
   then enforce the count after decode (truncate to N or 422). Preserves
   the user's "≤30 events" intent without the DFA blowup. This turns a
   refused request into a working one.

3. Accept "outlines-core cannot compile this shape in bounded memory" and
   document the constraint; offer (1)/(2) as the supported path.

`rust-shim/examples/schema_bench.rs` was the reusable outlines harness used
to localise this (removed in M53 along with outlines-core).

## Resolution — M53: engine swapped to llguidance

Decision (operator): keep the hard guarantee, replace the engine. The
`outlines-core` full-DFA precompile was swapped for **llguidance**
(guidance-ai, Rust, C-ABI, MIT) — a lazy/incremental grammar engine that
parses bounded repetition with a counter instead of unrolled states, behind
the **same C ABI** so the Swift surface barely moved.

Validated locally (`/tmp/llg-spike`, llguidance 1.7.5 + toktrie 1.7.5)
against the exact failing schema at the full 151k vocab:

| metric | outlines-core | llguidance |
|---|---|---|
| construct (parser/index) | ~48 s, OOM-prone | **0.24 s** |
| peak RSS | ~150 GB (projected) | **79 MB** |
| 100k-token walk | — | 81 MB (flat) |
| real schema, valid doc | — | accepts, ends accepting ✓ |

Cost decomposition drove the cache redesign: the per-model `ParserFactory`
(vocab slicer, ~0.24 s) is the only non-trivial cost and is
schema-independent → cached once per model. The per-schema compile is now
~1 ms, so the M49.1 per-schema `StructuredIndex` cache **and** the M49.5
complexity gate (whose model was wrong, and which would otherwise still
refuse the now-working schema) were both removed.

Implementation: `rust-shim/` (Cargo + `src/lib.rs` + header), the
`AthenaStructured` wrappers, `MLXLLMModule` caching, and the gate's
config/CLI/error wiring. `cargo test` (3) + `swift build` (compile + link
against the new `.a`) green.
