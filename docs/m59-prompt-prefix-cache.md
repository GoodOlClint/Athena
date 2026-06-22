# M59 — Cross-request prompt-prefix KV cache

Status: **planned, not started.** Milestone number provisional (latest shipped = M58).

## 1. Problem & motivation

Today every `/v1/chat/completions` request runs a full prefill from scratch.
There is **no** cross-request KV reuse: `SpeculativeGeneration.generate` allocates
a fresh `backbone = model.newCache(...)` per call and prefills the entire prompt
([SpeculativeGeneration.swift:71](../Sources/AthenaLLM/SpeculativeGeneration.swift#L71),
[:99-118](../Sources/AthenaLLM/SpeculativeGeneration.swift#L99-L118)). The existing
"prompt cache" (`preflightPromptCache` / `promptCacheCapBytes`) is an OOM **admission
guard**, not a reuse cache.

The driving consumer (the consuming application) wants to split one big extraction call per
document into 2–4 narrower passes. Each pass shares a bit-identical first ~20 KB
(static system prompt + verbatim document text); only the trailing instruction and
the `response_format` json_schema differ. On the production `Qwen3.5-27B-8bit-mtp`
a ~4 KB doc is ~190 s, **prefill-dominated**. Without prefix reuse, N passes = N×190 s
— a non-starter. With reuse, passes 2..N pay suffix-prefill + generation only.

**Goal:** transparently reuse a previously-computed KV prefix across separate chat
completion requests that share a leading token run, producing **bit-identical**
output to a cold prefill.

## 2. Surface decision — OpenAI-standard transparent caching

The OpenAI-compatible standard already specifies this pattern and it is **transparent**
(no enable flag, no stateful conversation handle in the data path):

- **Automatic prefix caching** — server reuses the longest previously-computed prefix.
- **`prompt_cache_key`** (optional request string) — a routing/scoping hint combined
  with the prefix hash to raise hit rates. This is the standard form of the consuming application's
  requested opt-in signal; we adopt it instead of a bespoke header.
- **`usage.prompt_tokens_details.cached_tokens`** — standard field reporting how many
  input tokens were served from cache.

Refs: <https://developers.openai.com/api/docs/guides/prompt-caching>,
<https://openai.com/index/api-prompt-caching/>.

**Decisions:**
- Caching is **transparent** on `/v1/chat/completions`, native `/api/chat`, and the
  queue `conversation` job — **zero client change** to get hits.
- Accept OpenAI `prompt_cache_key` as the scoping hint; default scope also includes
  the authenticated principal and the resident model id.
- Emit `usage.prompt_tokens_details.cached_tokens` on all three surfaces.
- A native `/api/cache/prompt` surface is added **only** for operator
  management/observability (stats + manual flush, `daemonAdmin`-gated) — **not** a
  data-path conversation context. the consuming application Q3 (server-side conversation handle)
  is intentionally **declined**: transparent caching + `prompt_cache_key` covers it
  with none of the lifecycle/ownership burden and keeps sync/queue symmetric.

## 3. Memory vs disk decision — in-memory for v1, disk deferred

Two code facts drive this:

1. **No ready-made serializer.** Substrate `KVCache` conformers (`KVCacheSimple`,
   `MambaCache`/GatedDelta) expose no public serialize/deserialize — state is opaque
   MLX-resident memory. The M20 "prompt-cache round-trip" is in-inference quantization
   validation, **not** a byte-serialization seam.
2. **Blobs are huge, access is hot.** At fp16 the 27B backbone KV is ~256 KB/token →
   a ~6 K-token shared prefix is ~1.5 GB (~400 MB under TurboQuant 4-bit). Too large
   for SQLite BLOBs to be sane. And the access pattern is back-to-back passes within
   seconds on a warm daemon, where an in-memory entry is still live — disk buys ~nothing.

**Decisions:**
- v1 = in-memory, governed, bounded LRU of live cache snapshots.
- Disk persistence = **M59.5, deferred** (parked, not built). If ever revisited, the
  home is **flat files in the data dir keyed by prefix hash**, NOT SQLite, due to size.

## 4. Central technical risk — heterogeneous MTP backbone

The production model is MTP, so this is worthless unless it works on the speculative
backbone, which mixes two cache kinds:

- **Attention** (`KVCacheSimple`): append-only, already `.trim()`-able
  ([SpeculativeGeneration.swift:81](../Sources/AthenaLLM/SpeculativeGeneration.swift#L81)).
  Trimming to an arbitrary common-prefix length `L` is fine.
- **Recurrent** (Mamba/GatedDelta): **fixed-size, non-positional** state — cannot be
  trimmed to an arbitrary earlier offset. But it **is** snapshottable: `GDNRollback`
  already snapshots `mc.state` as `[MLXArray]`
  ([AthenaQwen35.swift:215-216](../Sources/AthenaModels/AthenaQwen35.swift#L215-L216))
  and restores it on draft rejection.

**Design that reconciles them.** Prefill already runs in 512-token chunks
([SpeculativeGeneration.swift:99-118](../Sources/AthenaLLM/SpeculativeGeneration.swift#L99-L118)).
Snapshot the recurrent state (small, fixed-size, cheap) at each chunk boundary
alongside the trimmable attention KV. On a hit at common-prefix length `L`:

1. trim attention KV to `L`;
2. restore the recurrent snapshot at `floor(L/512)·512`;
3. replay the sub-chunk remainder (`L mod 512` tokens) to advance recurrent state to
   exactly `L`;
4. prefill only the divergent suffix.

The stored snapshot is **cloned** on use so in-flight decode never mutates the cached
entry. The **MTP draft cache** (`makeMTPCache()`,
[:72](../Sources/AthenaLLM/SpeculativeGeneration.swift#L72)) must be snapshotted/re-warmed
too, or drafts diverge — named risk for slice 1.

**Hard gate (matches M20/M21):** prefix-reuse + suffix-prefill must yield
**bit-identical greedy** output vs. a cold full prefill. This is the slice-1 acceptance
criterion, validated on the real 27B-mtp via the manual host-bound tier.

## 5. Slice plan

### M59.1 — In-memory prefix-cache core (MTP path) + bit-identical gate
- New `actor PrefixKVCache` (in `AthenaLLM`). Stores entries keyed by
  `(principal, modelId, prompt_cache_key?)` → list of `{tokens:[Int], attnKV, recurrentCheckpoints, mtpDraftState, byteEstimate, lastUsed}`.
- Longest-common-prefix match within a key scope; clone-on-hit; chunk-boundary
  recurrent snapshots + sub-chunk replay (§4).
- Wired into `runSpeculative` ([MLXLLMModule.swift:488](../Sources/AthenaLLM/MLXLLMModule.swift#L488))
  around `promptTokens` ([:569](../Sources/AthenaLLM/MLXLLMModule.swift#L569)) and into
  `SpeculativeGeneration.generate` at the cache-creation/prefill seam
  ([:71](../Sources/AthenaLLM/SpeculativeGeneration.swift#L71)).
- Behind config flag `[prompt_cache].enabled`, **default false**. Bounded by entry
  count only in this slice (governor accounting is M59.2).
- **Acceptance:** bit-identical-greedy e2e on real 27B-mtp (cold vs. warm produce
  identical token ids); manual wall-time showing pass-2 prefill collapses to
  suffix-only. Non-MTP substrate path untouched (out of scope, §7 risk 7).

### M59.2 — Governor integration
- Register pool bytes with `MemoryGovernor` (new reserved field alongside
  `residentBytes`, [MemoryGovernor.swift:98](../Sources/AthenaCore/MemoryGovernor.swift#L98);
  reserve/release at [:264](../Sources/AthenaCore/MemoryGovernor.swift#L264)/[:274](../Sources/AthenaCore/MemoryGovernor.swift#L274)).
- LRU eviction by **bytes + count + idle-TTL** (default 600 s, mirrors OpenAI's
  5–10 min inactivity eviction); evict under memory pressure before model-load refusal.
- Reconcile with `promptCacheCapBytes` so per-request admission and the persistent pool
  don't double-count or starve each other. Expose pool bytes in `GovernorSnapshot`
  ([:29](../Sources/AthenaCore/MemoryGovernor.swift#L29)).
- **Acceptance:** pool never drives the box to OOM under a sustained multi-pass loop;
  governor snapshot + `/healthz` report pool bytes/entries.

### M59.3 — OpenAI-surface observability + hint
- Parse `prompt_cache_key` from chat bodies (`OpenAIDTO`).
- Add `cachedTokens` to `TokenUsage` ([GenChunk.swift:8-16](../Sources/AthenaLLM/GenChunk.swift#L8-L16))
  and emit `usage.prompt_tokens_details.cached_tokens` in the `Usage` DTO
  ([OpenAIDTO.swift:256](../Sources/athena/Server/OpenAIDTO.swift#L256)) across **all**
  surfaces: `/v1/chat` (stream + non-stream), native `/api/chat`, queue `conversation`.
- Heartbeat gains `phase=prefix-replay` + hit/miss/replayed-token counters; add a
  `doctor` check for cache posture.
- **Acceptance:** the consuming application confirms hits via `cached_tokens` during a 2-pass ingest;
  e2e asserts `cached_tokens>0` on the second of two prefix-sharing requests.

### M59.4 — Operator management
- `GET /api/cache/prompt` (stats: entries, bytes, hit/miss) and
  `DELETE /api/cache/prompt` (flush), `daemonAdmin`-gated; audit the flush at the shared
  handle chokepoint (M30).
- `athena cache` CLI (local + remote via `DaemonOptions.isRemote`); `/ui` panel.
- OpenAPI spec entries + bidirectional drift-guard (M32).
- **Acceptance:** e2e gate; spec↔routes exact.

### M59.5 — Disk persistence (SHIPPED as ADR 027, v0.10.212–216)
- No longer deferred. The cross-restart reuse pattern emerged (the downstream client/`the downstream client`
  motivation), and the §3 blockers were overturned: the serializer now exists
  (`KVFrame`/`KVByteCodec` from ADR 024 T3), and resume-across-restart is exactly
  the access pattern disk pays off for. Built as **ADR 027** — encrypted flat files
  in the data dir keyed by prefix hash (not SQLite, as predicted), a disk L2 under
  this in-RAM L1, off by default. See `docs/decisions/027-disk-kv-snapshots.md` and
  `docs/kv-cache-disk-snapshots.md`. Proven bit-identical across a real restart.

## 6. Config (5-touchpoint, mirrors `kv_compression`)

New `[prompt_cache]` TOML block:

| key | default | meaning |
|---|---|---|
| `enabled` | `false` (flip to `true` once M59.1 gate passes) | master switch |
| `max_entries` | `4` | LRU entry cap |
| `max_bytes` | governor-derived | pool byte cap |
| `idle_ttl_secs` | `600` | evict entries idle longer than this |
| `scope` | `principal` | `principal` \| `cache_key` \| `both` |

Wire through the same 5 touchpoints as `kvCompression`:
[AthenaConfig.swift:38](../Sources/AthenaDeploy/AthenaConfig.swift#L38) (field),
[:121](../Sources/AthenaDeploy/AthenaConfig.swift#L121) (init param),
[:152](../Sources/AthenaDeploy/AthenaConfig.swift#L152) (assign),
[:295](../Sources/AthenaDeploy/AthenaConfig.swift#L295) (TOML scalar),
[DefaultConfig.swift:54](../Sources/AthenaDeploy/DefaultConfig.swift#L54) (default
comment). Per-principal keying mirrors the M29 rate-limit `[String: Bucket]` in
[RateLimit.swift:30](../Sources/athena/Server/RateLimit.swift#L30).

## 7. Named risks

1. **Recurrent arbitrary-trim limit** → chunk-boundary snapshot + sub-chunk replay (§4).
2. **Bit-identical contract** incl. the MTP draft cache → slice-1 gate is the proof.
3. **GB-scale entries** → small bound + governor eviction; this is *why* it's in-memory.
4. **Clone-on-hit immutability** → cached entry must not be mutated by in-flight decode;
   explicit MLXArray copy semantics + eviction-during-use guard (refcount the entry
   while a request holds it).
5. **TurboQuant interaction** → stored KV is already 4-bit; reuse must round-trip
   identically (leverage M20 validation).
6. **Structured-output independence** → confirmed: schema drives the request-local
   `GuidedDecoder`/`StructuredIndex`
   ([MLXLLMModule.swift:104-114](../Sources/AthenaLLM/MLXLLMModule.swift#L104-L114)),
   generation-side, never touches prefill. Different `response_format` across passes is
   a **guaranteed hit** (answers the consuming application Q1e). Verify no schema bytes leak into the
   prefix key.
7. **Substrate (non-MTP) path** → `container.generate()` doesn't expose its cache to
   Athena; prefix reuse there is **out of scope** (would need substrate support). The
   production path is MTP, so the consuming application is unaffected. Document the limitation.
8. **Concurrency** → `PrefixKVCache` is an `actor`; entries handed to a running
   generation must be cloned before mutation and protected from eviction mid-use.

## 8. Validation

Per the manual host-bound integration tier (Mac Studio real 27B-mtp daemon +
MacBook client; not CI): the e2e shell script sets up + checklists, a human runs the
multi-pass actions. Gates: (a) bit-identical greedy cold-vs-warm; (b) `cached_tokens>0`
on pass 2; (c) wall-time: pass-2 prefill ≈ suffix-only; (d) governor never OOMs under a
sustained loop; (e) OpenAPI spec↔routes exact.

## 9. Out of scope

- Disk persistence (M59.5, deferred).
- Non-MTP substrate-path prefix reuse.
- Prefix reuse for embeddings / transcription / other module classes (the win is the
  long LLM prefill).
- A stateful server-side conversation handle (declined; §2).
