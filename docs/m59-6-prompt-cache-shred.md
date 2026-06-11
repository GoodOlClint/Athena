# M59.6 — scoped prompt-cache invalidation ("shred", SPEC, proposed)

**Status:** spec for review. Not built. Milestone `M59.6` is a proposal —
versioning is the maintainer's call. Independently shippable now; the disk-tier
half (§7) rides along whenever the deferred M59.5 disk tier lands.

**One-line:** let a requestor explicitly drop their own document's cached prefix
KV the moment they're done with it — owner-scoped, refcount-safe,
audit-fingerprinted, and designed to be cryptographically tier-complete once the
prefix can spill to disk — instead of waiting for idle-TTL / LRU / pressure-shed.

---

## 1. Motivation

The prompt-prefix KV cache **is** the document's content (the model's internal
representation of the prompt). In the motivating legal-document workload that is
sensitive data with retention obligations. Today the only ways an entry leaves
the pool are passive — idle-TTL sweep, LRU eviction, or a global admin
flush-all-idle ([`handlePromptCacheFlush`](../Sources/athena/Server/AthenaServer.swift#L2255-L2275)).
There is no way for the **requestor** to say "this document's multi-pass
extraction is complete — erase its prefix now."

Two concrete wins:
- **Data-minimization / right-to-erasure.** A client contractually required to
  purge a document's derived artifacts can prove the cached representation was
  dropped on demand, not "eventually, when TTL expired."
- **Resource hygiene.** A completed document's prefix is dead weight occupying
  the bounded pool (and, with the disk tier, encrypted bytes on disk).
  Proactive shred frees a slot for the next document instead of forcing an LRU
  eviction of something still useful.

This is the active counterpart to the passive eviction the pool already does, and
the natural follow-on to the M59.4 management surface.

---

## 2. Non-goals

- **Not a guaranteed byte-scrub in RAM.** See §6 — RAM removal is immediate and
  the content is no longer reachable or served, but byte-level zeroization of the
  MLX buffer pool is best-effort. We will not market more than is true.
- **Not general cache-entry CRUD.** No "edit", no "list every entry's content".
  Shred is removal, plus the stats that already exist.
- **Not auto-shred-on-disconnect** or any implicit lifetime coupling. Shred is
  explicit (or, as deferred sugar, opt-in per submit — §4.5).
- **Not a new global-flush mechanism.** The admin global flush
  (`DELETE /api/cache/prompt`) is unchanged; this adds *scoped* removal.

---

## 3. Design overview

| Slice | What | Risk |
|---|---|---|
| **M59.6a** | Pool internals: tag entries with `(principal, cache_key)`, add tombstone, `shred(principal:cacheKey:)`, `acquire` skips tombstoned, `release` drops tombstoned | low (additive, behind no behavior change until a shred is issued) |
| **M59.6b** | HTTP routes + DTOs + owner-tier auth + audit fingerprint + `athena cache shred` CLI + OpenAPI + drift-guard + e2e | medium (new auth tier on the cache surface) |
| *(M59.5)* | Disk-tier propagation of `invalidate` = crypto-erase the spilled file | folded into the disk-tier milestone |

With M59.6 present but no shred issued, behavior is byte-identical to today, and
the M59.1 bit-identical warm-reuse gate is unaffected (tagging doesn't touch the
reuse path).

---

## 4. M59.6a — pool internals

### 4.1 Entry tagging
Today an `Entry` stores the *composite scope string* but not the originating
`prompt_cache_key` ([PrefixKVCache.swift:97-126](../Sources/AthenaLLM/PrefixKVCache.swift#L97-L126)).
Under the `principal` scope default the scope string is just the principal, so two
documents are indistinguishable in the pool. Add explicit management labels,
**decoupled from the reuse scope**:

```swift
private final class Entry {
    // ... existing: id, scope, tokens, attn, checkpoints, byteEstimate, lastUsed, refcount
    let principal: String?   // NEW — owning principal (for owner-scoped shred)
    let cacheKey: String?    // NEW — originating prompt_cache_key (the doc label)
    var tombstoned: Bool     // NEW — shred-on-release flag (default false)
}
```

`runSpeculative` already has both `principal` and `promptCacheKey` where it builds
the scope ([MLXLLMModule.swift ~685-696](../Sources/AthenaLLM/MLXLLMModule.swift#L685-L696));
thread them through `SpeculativeGeneration.generate` into `store(...)`:

```swift
public func store(
    scope: String, principal: String?, cacheKey: String?,   // NEW params
    promptTokens: [Int], backbone: [KVCache], recorder: Recorder)
```

This makes targeted shred work **regardless of `prompt_cache_scope`** — the tag,
not the scope key, is what shred matches on.

### 4.2 The shred method

```swift
/// Remove entries owned by `principal` (and, if non-nil, matching `cacheKey`).
/// refcount==0 entries are removed immediately; in-flight (refcount>0) entries
/// are tombstoned and removed on release. Returns (removed, tombstoned).
/// Idempotent: matching nothing returns (0, 0) — never an error.
public func shred(principal: String?, cacheKey: String?) -> (removed: Int, tombstoned: Int)
```

Matching:
- **Owner shred** (handler passes the bearer principal): `e.principal == principal
  && (cacheKey == nil || e.cacheKey == cacheKey)`.
- **Admin shred** (handler may pass an explicit target or nil): `(principal == nil
  || e.principal == principal) && (cacheKey == nil || e.cacheKey == cacheKey)`.

The handler — not the pool — decides which form by resolving the caller (§5). The
pool method is mechanism only.

### 4.3 Refcount-safe tombstoning
- An entry with `refcount == 0` is removed from `entries` immediately (and its
  arrays dropped; see §6).
- An entry with `refcount > 0` is **tombstoned**, not removed — a live decode is
  mid-flight against it. It is freed in `release(_:)` when its refcount reaches 0.
- `acquire(...)` skips tombstoned entries: the candidate scan at
  [PrefixKVCache.swift:214](../Sources/AthenaLLM/PrefixKVCache.swift#L214) gains
  `&& !e.tombstoned`. So the instant a shred lands, a tombstoned prefix is **never
  served to a new request**, even while an existing decode finishes on it. This
  closes the window where a "shredded" document could still warm-start a new pass.
- `release(_:)` ([PrefixKVCache.swift:254-260](../Sources/AthenaLLM/PrefixKVCache.swift#L254-L260))
  removes the entry when `--refcount == 0 && tombstoned`.

### 4.4 `invalidate` as the tier-complete seam
Expose the public entry point as `invalidate(principal:cacheKey:)` that today
calls `shred(...)` (RAM). When the disk tier (M59.5) exists, the same call also
unlinks / crypto-erases the matching on-disk file(s) under the same
`(model-fingerprint, principal, cacheKey)` key. Designing the API around
`invalidate` now means the HTTP/CLI contract doesn't change when disk lands.

### 4.5 Deferred sugar — auto-shred-on-completion
A submit flag (`shred_prompt_cache_after: true`) that invalidates the prefix when
*that* job completes. Elegant for a **serial tail**, but awkward under the M61
fanned-out topology (passes 2/3/4 run concurrently; "the last one" is whichever
finishes last — the client's reconcile cron knows that, a per-job flag can't). So
ship the explicit endpoint first; add the flag later only if a serial pattern
wants it. **Deferred.**

---

## 5. M59.6b — API, auth, audit

### 5.1 Routes (two tiers, two paths)
Path-based permission resolution ([Auth.swift:323-326](../Sources/athena/Server/Auth.swift#L323-L326))
is a string match on `path` and can't see query params, so the two tiers get two
paths:

| Verb · Path | Permission | Behavior |
|---|---|---|
| `DELETE /api/cache/prompt` *(existing)* | `daemon.admin` | unchanged: flush-all-idle. **Extended** with optional `?principal=&cache_key=` for operator-driven targeted erasure (e.g. a support-desk GDPR purge); no filter ⇒ today's flush-all-idle |
| `DELETE /api/cache/prompt/mine` *(new)* | `inference` | shred the **caller's own** entries; optional `?cache_key=<doc>` to target one document. Hard principal-match — the body/query can never name another principal |

Auth wiring: add `if path == "/api/cache/prompt/mine" { return .inference }` ahead
of the existing `/api/cache/prompt → .daemonAdmin` rule. The `/mine` handler
resolves the bearer principal and passes it as the `principal` filter — there is
no way to express "someone else's" through this path.

> Safety argument for the inference tier: the worst a caller can do via `/mine`
> is shred their *own* cache and slow their *own* future passes. They cannot
> touch another principal's entries (principal-match) and cannot flush the global
> pool (that stays `daemon.admin`). This is why dropping below `daemon.admin` is
> safe here — confirmed direction.

### 5.2 DTO

```swift
/// DELETE /api/cache/prompt/mine (and the targeted admin form).
struct PromptCacheShredResponse: Codable {
    let shredded: Int      // entries removed immediately
    let tombstoned: Int    // in-flight entries marked, freed on release
    let entries: Int       // pool occupancy after
    let bytes: Int
}
```

Sits beside the existing `PromptCacheStatsResponse` / `PromptCacheFlushResponse`
([NativeAPIDTO.swift:59-79](../Sources/athena/Server/NativeAPIDTO.swift#L59-L79)).

### 5.3 Audit (data-minimizing)
Audit-log every shred (M30), mirroring `prompt_cache.flush`:
- `action = "prompt_cache.shred"`
- `target =` a **fingerprint** of the cache_key (e.g. `sha256(cache_key)` first 12
  hex), **never the raw key** — proves *which* document's prefix was erased for
  compliance without writing a document id into the audit log (consistent with
  the content-opt-out posture).
- `detail = "shredded=N tombstoned=M scope=owner|admin"`
- `result = ok` even when N==M==0 (idempotent no-op is still an auditable
  "erasure requested").

### 5.4 CLI
Extend `CacheCmd` ([RemoteCache.swift:91-119](../clients/Sources/AthenaClient/RemoteCache.swift#L91-L119)):
- `athena cache shred <cache_key>` → `DELETE /api/cache/prompt/mine?cache_key=<…>`
- `athena cache shred --all-mine` → `DELETE /api/cache/prompt/mine` (no filter) —
  requires the explicit flag so a bare `shred` can't nuke all your prefixes by
  accident (§9 Q1).
- Admin targeted form via the existing `flush` command extended with
  `--principal`/`--cache-key`, or a dedicated admin verb — minor, settle in
  review.

### 5.5 OpenAPI + drift-guard
Add `/api/cache/prompt/mine` to the embedded spec
([OpenAPISpec.swift:501-522](../Sources/athena/Server/OpenAPISpec.swift#L501-L522))
and the bidirectional drift-guard in `deploy/integration/e2e-m59-cache-api.sh`
(every route documented, no stale specs). Document the extended query params on
the existing `DELETE /api/cache/prompt`.

---

## 6. Erasure semantics (what "shred" actually guarantees)

Stated plainly because the KV is content:

- **RAM (today):** removal is **immediate and complete at the reachability level**
  — the entry leaves the pool, is skipped by any in-flight `acquire`, and is never
  served again. Byte-level zeroization is **best-effort**: dropping the MLXArray
  references returns buffers to MLX's internal pool where the bytes persist until
  overwritten by reuse. Shred calls `MLX.Memory.clearCache()` (as the flush path
  already does) to trim that pool, but we do not promise a guaranteed scrub.
- **Disk tier (M59.5):** this is where shred has teeth. The spilled prefix is
  encrypted at rest (mandatory — content-at-rest), so `invalidate` deleting the
  file (and, if per-entry-keyed, dropping its key) is **cryptographic erasure** —
  unrecoverable regardless of SSD wear-leveling, which defeats naive overwrite.
  Crypto-erase is the strong guarantee; design the disk tier so shred can deliver
  it.

So the honest contract: **shred makes the content immediately unserveable and
unreachable; on disk it makes it cryptographically unrecoverable.** RAM
byte-residue is bounded by pool reuse, not zeroed on demand.

---

## 7. Disk-tier propagation (forward-compat note for M59.5)

When the disk tier lands, `invalidate(principal:cacheKey:)` must:
1. Remove/tombstone matching RAM entries (M59.6a, unchanged).
2. Resolve matching on-disk files by `(model-fingerprint, principal, cacheKey)`
   and crypto-erase them (delete file; drop per-entry key if keyed that way).
3. Honor content-opt-out: an opted-out tenant's prefix was never written to disk,
   so step 2 is a no-op for them.

No HTTP/CLI change at that point — the contract is already `invalidate`.

---

## 8. Edge cases & failure modes

| Case | Behavior |
|---|---|
| Shred after TTL/LRU already dropped it | `(0,0)`, `result=ok` — idempotent, not an error |
| Shred while a pass is mid-decode | entry tombstoned; blocked from new `acquire`; freed on `release` |
| Owner shred names another principal's key | impossible via `/mine` (principal-match); admin path required |
| `cache_key` omitted on `/mine` | shreds ALL caller's entries — gated behind `--all-mine` in CLI (§9 Q1) |
| Scope mode = `principal` (key not in scope key) | still works — shred matches the `cache_key` **tag**, not the scope string |
| Job rebound to a different model | tag carries `principal`+`cacheKey`; entries also carry `scope` (model-bound), so cross-model entries simply don't match the live model's reuse — shred still removes by tag |
| Cache disabled (`prefixCache == nil`) | `(0,0)`, audited `detail=disabled`, same as flush |
| Concurrent shred + acquire of same entry | lock-guarded; acquire either sees tombstoned (skips) or hasn't-yet (gets refcount, freed on release) — no use-after-free |

---

## 9. Test plan

Manual host-bound tier ([integration-test-topology]) + e2e gate (extend
`e2e-m59-cache-api.sh`; running total advances):

1. **Tag present** — after a cached store, the entry carries `principal` +
   `cacheKey` (unit).
2. **Targeted shred** — store two docs (keys A, B) for one principal; shred A;
   assert A gone, B intact, count `(1,0)`.
3. **Idempotent** — shred A again → `(0,0)`, `result=ok`.
4. **Cross-principal isolation** — principal P1 shred cannot remove P2's entry
   (via `/mine`); admin targeted form can.
5. **In-flight tombstone** — issue shred during a decode holding the entry;
   assert a *new* same-key request does NOT warm-start (cached_tokens 0), and the
   entry is freed once the in-flight decode completes.
6. **acquire-skips-tombstoned** — explicit unit: tombstoned entry never returned
   by `acquire`.
7. **Auth tiers** — an `inference` token shreds its own via `/mine`; the same
   token gets 403 on `DELETE /api/cache/prompt` (global flush stays admin); admin
   token can do both.
8. **Audit fingerprint** — `GET /api/audit?action=prompt_cache.shred` shows the
   row; `target` is a hash, raw `cache_key` appears nowhere in the audit row.
9. **Bit-identical gate** — re-run the M59.1 warm-vs-cold gate; tagging changes
   nothing.
10. **OpenAPI drift-guard** — `/api/cache/prompt/mine` documented; no undocumented
    route; no stale path.
11. **CLI** — `athena cache shred <key>` round-trips; `--all-mine` required for
    the keyless form.

---

## 10. Open questions

1. **Keyless `/mine` ergonomics** — require an explicit `--all-mine` (proposed) so
   a bare `shred` can't drop all of a principal's prefixes by accident? Leaning
   yes.
2. **Auto-shred-on-completion flag** (§4.5) — build the `shred_prompt_cache_after`
   submit flag now, or wait for a concrete serial-tail need? Leaning wait.
3. **Audit raw-key escape hatch** — fingerprint is the default; do operators ever
   need the raw key in audit (debug), behind a config toggle? Leaning no
   (minimization wins).
4. **Disk crypto-erase keying** (M59.5) — per-entry key (true crypto-erase on key
   drop) vs whole-store key (erase = file delete under SQLCipher-style at-rest)?
   Defer to the disk-tier spec, but M59.6's `invalidate` seam is agnostic to the
   choice.
```
