# Collapse persistent data tenants — change plan (ADR 025)

**Status:** Proposed — awaiting operator approval before any implementation
(brownfield change gate). Realizes [ADR 025](decisions/025-collapse-persistent-data-tenants.md),
motivated by [ADR 024](decisions/024-in-memory-data-protection-coresident-threat.md).

**Goal:** minimize request-related data-at-rest. Delete the async queue + built-in
vector DB entirely, drop the `/v1/store/*` endpoints, and add a stateless loopback
mode where no `athena.sqlite` is created. Harden the (non-removable) temp-upload
path.

## Persistence inventory (what this plan acts on)

One SQLite file `<data-dir>/athena.sqlite` (+ `-wal`/`-shm`), `AthenaStore` actor.

| Table / artifact | Disposition under ADR 025 |
|---|---|
| `jobs` (queue: prompts + results) | **deleted** |
| `vectors` (embeddings + metadata) | **deleted** |
| `auth_users`/`auth_tokens`/`auth_user_roles` | kept (authed mode); absent in stateless mode |
| `audit_log` | opt-out / ephemeral; inert in stateless mode |
| `usage_counters` | opt-out / ephemeral; inert in stateless mode |
| `model_allowlist` | config-seeded; SQLite only when DB exists (narrows M42) |
| temp upload files (`NSTemporaryDirectory()/athena-*`) | **hardened** (S5), not removed |
| store exports, `athena.err.log`, config TOML, weights, Keychain | unchanged |

## Slices

Each slice is a test-pinned commit (house style: stacked, regression test per fix,
pre-commit pipeline Tests → Security → Quality → Refactor). `appVersion` bump in
the slice commit.

### S1 — Vector DB out — SHIPPED v0.10.201 (merged with S3)
- Delete `Sources/AthenaStore/VectorStore.swift`; `vectors` table + index +
  migrations in `AthenaStore.swift`; `allVectors`/`putVector`/`getVector`/
  `deleteVector`/`pruneVectors`.
- Remove `/v1/vectors`, `/v1/vectors/query`, `/v1/vectors/{id}`, `/v1/vectors/stats`
  from OpenAPISpec.swift + handlers in AthenaServer.swift.
- Remove vector DTOs (OpenAIDTO.swift) and `athena vectors` CLI
  (`clients/.../VectorsCmd.swift`, `Vectors.self` in `Athena.swift`).
- Remove `vector_cap_bytes` / `vector_ttl_secs` config (Load.swift).
- Remove `vectors.read`/`vectors.write` RBAC permissions.
- Delete `VectorStoreTests.swift`; trim `AthenaStoreTests.swift`; update
  `e2e-rbac.sh`; update docs (`at-rest.md`, CLAUDE.md M7 ref).

### S2 — Queue out — SHIPPED v0.10.203
- **Fork RESOLVED: (a) synchronous + SSE progress, no job ids/persistence.**
  `POST /api/models/{pull,convert,prune}` now run in-process and stream SSE
  (`data: {"event":"progress"|"done"|"error",…}` then `[DONE]`); `.modelWrite`
  gate + M30 audit kept. The WebUI console calls the same handlers synchronously
  (EventSource is GET-only). The CLI consumes the SSE stream; `--wait` removed,
  `--follow` = print progress.
- Deleted `RequestQueue.swift`, `QueueWorkerService.swift`, the `jobs` table +
  all helpers (`insertJob`/`updateJob`/`getJob`/`listJobs`/`nextPendingJob`/
  `cancelQueuedJob`/`deleteJob`/`clearJobRequest`/`pruneJobs`/`trimJobs`/
  `recentJobSummaries`/`queuedJobCount`/`jobCount`) + `JobRow`/`JobSummary`.
- Removed `/v1/queue*` routes + DTOs + handlers + the `queuedExecute`/
  `enqueueModelOp` dispatch; `athena queue` CLI; `queue_result_ttl_secs`/
  `queue_max_rows`/`drop_request_content` config (all 5 touchpoints); the
  `queue.submit` RBAC permission; the `/healthz` `queueDepth` field; the `/ui`
  recent-jobs dashboard panel + job poller; queue tests; queue docs
  (`queue-prompt-cache-contract.md`, `m61-prefix-affinity-queue.md`).

### S3 — Store endpoints out — SHIPPED v0.10.201 (merged with S1)
- Remove `/v1/store/export` + `/v1/store/stats` routes, DTOs, handlers,
  `athena store` CLI, `store.admin` RBAC. Keep `AthenaStore` (auth/audit/usage/
  allowlist). Path-confinement export code goes with it.

### S4 — Stateless loopback mode + audit/usage opt-out — SHIPPED v0.10.204
- Mode selection is the MLX-free, unit-pinned `AthenaCore/StoreMode.swift`
  (`resolve` + `isLoopback`, pinned in `StoreModeTests`): **ephemeral** (no
  `athena.sqlite`) iff a loopback bind with no bootstrap keys, no existing DB
  file, no `encrypt_store`, and no `persist_store` override — anything that
  needs durable state forces **persistent**. With S7 (allowlist retired) the
  only remaining DB tenants are auth/audit/usage, so a credential-less loopback
  daemon writes nothing.
- `AthenaStore(ephemeral:)` opens an in-memory SQLite store (`:memory:`
  sentinel, `isEphemeral`/`dbLocationLabel` helpers); the schema is factored
  into a shared `createSchema`. Audit/usage exist in RAM and vanish on exit —
  inert, never on disk.
- `persist_store` config switch (5 touchpoints + `--persist-store` flag) forces
  a persistent store even in the stateless case (the ADR-025 "config switch
  controls persistence independent of mode" lever).
- Doctor reports the mode (`store: STATELESS …` vs `store: persistent at …`)
  and no longer creates a phantom `athena.sqlite` just to count rows.
- Verified end-to-end: a fresh loopback run with no keys serves `/healthz` and
  creates no `athena.sqlite`; the same run with `ATHENA_ADMIN_KEYS` set creates
  the DB and enables auth.

### S5 — Eliminate upload temp files via in-memory decode (Option D) — SHIPPED v0.10.199

**Status: shipped.** `Sources/AthenaTranscription/InMemoryAsset.swift` backs an
`AVURLAsset` with an `AVAssetResourceLoaderDelegate` over the `athena-mem://`
scheme; `AudioDecode.pcm16kMono(from:Data,filename:)` and
`VideoAudioTrack.extractPCM(from:Data,filename:)` run one shared `AVAssetReader`
core (`readFirstAudioTrackPCM`). All six serve-path call sites (transcription,
diarization×2, speaker-embedding×2, video) pass `Data` and the temp-write/`defer`
blocks are gone; a boot sweep (`sweepLegacyUploadTempFiles`) clears any pre-S5
`athena-*` orphan. **Decode-parity finding:** a resource-loader-backed asset does
not byte-sniff (unlike the old `AVAudioFile(forReading:)`), so a magic-byte
sniffer (`sniffExtension`) recovers the container for unnamed uploads — preserving
the pre-S5 behavior where a filename-less upload still decoded. Unit-pinned in
`InMemoryAssetTests` (parity, nil-filename, too-short floor, garbage→400, sweep).

**Decision:** feed Apple's decoders from the in-memory upload `Data` instead of
staging a temp file. No new dependency, same Apple decoders (no format-coverage or
PCM-parity risk, hardware decode retained), and the on-disk staging — the
crash-residue surface ADR 025 flagged — is **removed**, not merely hardened.

**Mechanism.** Back an `AVURLAsset` with an `AVAssetResourceLoaderDelegate` on a
custom URL scheme (e.g. `athena-mem://<uuid>`) that serves byte ranges from the
in-memory `Data`. `AVAssetReader` then demuxes/decodes audio and video — and
future M78.2 video frames via `AVAssetImageGenerator` over the same in-memory
asset — with no file. Audio File Stream Services (`AudioFileStreamParseBytes` +
`AudioConverter`) is the audio-only fallback if the resource-loader route is
awkward for bare audio.

**De-risk spike first (~½ day):** confirm an `AVAssetResourceLoaderDelegate`-backed
asset cleanly feeds `AVAssetReader` across our real container/codec set
(mp3 / m4a-aac / wav / flac / ogg / mov+mp4 audio tracks). If a format regresses,
fall back to Audio File Stream Services for audio and keep the resource-loader
route for video.

**Work:**
- Add the in-memory asset/resource-loader helper (Swift + thin Core Audio); handle
  byte-range requests, content-type/length, threading, cancellation.
- Refactor the two chokepoints to accept `Data`: `AudioDecode.pcm16kMono(from:)`
  ([AudioDecode.swift:110](AudioDecode.swift)) and `VideoAudioTrack.extractPCM`
  ([AthenaServer.swift:2100](../Sources/athena/Server/AthenaServer.swift#L2100)).
- Update the ~7 call sites (transcription, diarization×2, speaker-embedding×2,
  video) to pass `Data`; **delete** the temp-write + `defer`-delete blocks.
- Preserve the MLX-free bounds logic unchanged: `sampleBoundError` (D4
  decompression-bomb ceiling + min-sample floor) and the `DecodeError`→`AthenaError`
  taxonomy (issue #6), still unit-pinned (ADR 008/009).
- Backstop: a startup sweep of any legacy `athena-*` upload files (migration
  insurance; should be a no-op once D lands).

**LOE:** ~1–1.5 weeks. **No new dependency.** Risk concentrated in the
resource-loader plumbing (settled by the spike), not in correctness — the decoder
is unchanged, so model output is unchanged.

**Honesty boundary:** the upload `Data` + decoded PCM still live in process RAM
during the request → the live co-resident read stays an ADR-024 Tier-1 concern.
D removes disk residue, not in-memory exposure.

**Rejected here:** a bespoke Rust/symphonia decoder (no cross-platform driver;
owned codec/parity/CVE/maintenance tax — see ADR 025 Rejected) and an
ephemeral-keyed encrypted volume (superseded — nothing to encrypt once no file is
written).

### S7 — Retire the allowlist (ADR 026; lands before S4)
- Delete the `model_allowlist` table + `AthenaStore` methods
  (`listModelAllowlist`/`defaultModelAllowlistID`/`addModelAllowlist`/
  `removeModelAllowlist`/`setModelAllowlistDefault`/`modelAllowlistCount`).
- Remove `/api/models/allow` GET/POST/DELETE + `/api/models/allow/default` PUT
  routes + handlers + WebUI mirrors (`/ui/allowlist`, `/ui/api/allowlist*`); the
  `athena allowlist` CLI **including M43.5 offline `--data-dir`**; allowlist DTOs;
  `model.allow.{add,rm,default}` audit actions.
- Re-point selection at the store: each module's `allowedModelIds()` returns store
  dirs that `ModelSupport` (ADR 021) classifies as its modality; the gate becomes
  store-presence + class match via the retained `canonicalByStoreIdentity`
  (case-insensitive basename); miss ⇒ 400 `model_not_available`. No on-request
  download (unchanged).
- Default in TOML: generalize `athena default` → `athena default --module M
  <name>` (per-class keys; LLM keeps `model`). The `--*-model` load flags become
  first-boot config-default seeds, not allowlist seeds; `athena init` aux-pull
  follows the configured defaults.
- Ambiguity rule (MLX-free, unit-pinned): omit `model` + >1 of class + no
  configured default ⇒ **400 `ambiguous_model`**; exactly 1 ⇒ use it; 0 ⇒
  `model_not_available`.
- M41 `/api/models/{load,unload,resident}` + inference rebind stay, re-pointed at
  store models; `model.rebind` audit + governor reconcile unchanged.
- Move `ModelAllowlistTests` coverage to the store-identity helper; update
  `e2e-rbac.sh` (drop allowlist editing #11/M43.5, WebUI console M44.1) with the
  new ambiguity/default e2e; update docs.

### S6 — Surface + thesis reconciliation
- Update CLAUDE.md "Stable `/v1/*` endpoints" list (remove vectors/store/queue);
  spec↔routes drift-guard green.
- Amend ADR 011 (drop queue/vectors as governor tenants) and ADR 013 (remove the
  three native surfaces) — same edit window as the surface change.
- Full e2e gate.

## Open forks for review
1. **Model-ops replacement** (S2): synchronous+SSE vs in-memory tracker. (ADR 025
   defers this to implementation.)
2. **Consumer confirmation:** the the consuming application integration contract (memory) lists
   a "schema-less-queue" expectation and vectors as opt-in. Operator states no
   consumer uses either now — confirm the consuming application has dropped both before S1/S2
   land (removal is breaking).

_(Resolved: allowlist default → per-module TOML keys; ambiguity → 400
`ambiguous_model` — see ADR 026, S7.)_

## Sequencing
S1 → S3 are independent leaf removals (safe first). S2 waits on fork #1. **S7
(retire the allowlist)** lands before **S4**, since S4's unconditional stateless
mode depends on the allowlist no longer needing the DB. S4 is the behavioral
change (most review-sensitive). S5 is independent (Option D in-memory decode) and
could land first as a standalone data-at-rest win, after its ½-day spike. S6 closes
the surface/thesis loop.
