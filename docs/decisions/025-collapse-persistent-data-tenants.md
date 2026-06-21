# ADR 025 — collapse the persistent data tenants (queue + vector DB) and add a stateless loopback mode

**Status:** Accepted — **in progress** (M80). Shipped: S5 in-memory decode
(v0.10.199), S1+S3 vector DB + `/v1/store*` removal (v0.10.201), S7 allowlist
retirement (v0.10.202, [ADR 026](decisions/026-retire-allowlist-store-is-registry.md)),
**S2 queue removal + model-ops → synchronous + SSE (v0.10.203)**. Remaining: S4
(stateless loopback), S6 (surface/thesis reconciliation). See the change plan
(`docs/collapse-persistence-plan.md`).
Motivated by ADR 024's threat model (a co-resident adversary; minimize
data-at-rest): the fewer places the daemon persists request content, the smaller
the scraping/forensics surface. Operator-confirmed: **no current downstream
consumer uses the queue or the vector DB** — both can be handled consumer-side.

## Context

A full persistence audit (see the plan doc) found the daemon's data-at-rest lives
in **one** SQLite file — `<data-dir>/athena.sqlite` (+ `-wal`/`-shm`) owned by the
`AthenaStore` actor — holding **8 tables across 4 domains**, plus transient temp
upload files:

| Table / artifact | Holds | Domain |
|---|---|---|
| `jobs` | **request prompts + completion results** (BLOBs) | async queue |
| `vectors` | **embeddings + metadata** (BLOBs) | vector DB |
| `auth_users`/`auth_tokens`/`auth_user_roles` | credential hashes, token digests, roles | auth/RBAC |
| `audit_log` | principal/action/target/result (no content) | audit |
| `usage_counters` | per-principal token counts | usage |
| `model_allowlist` | operator model registry | lifecycle (M42) |
| temp upload files | **raw audio/video/image bytes** in `NSTemporaryDirectory()` | transcription/vision/video |

Key facts that shape the decision:

- **Queue + vectors are the only content-bearing *removable* tenants.** Inference
  (chat/embeddings/transcription/diarization/vision/video) does **not** read or
  write either — they are leaf data tenants. The **one** coupling is that
  operator model lifecycle ops (`pull`/`convert`/`prune`) are enqueued via
  `enqueueModelOp()`.
- **The SQLite container itself is load-bearing** for auth/audit/usage/allowlist,
  so the container is not deleted — only the `jobs`/`vectors` tables and their
  surface come out.
- **The temp upload files are the sharpest content-at-rest risk and are NOT
  removable** — transcription/vision need bytes on disk (`AVAssetReader`, whisper,
  image decode). They are `defer`-deleted on success but **survive a crash**, in a
  world-readable temp dir. This is hardened, not collapsed.

## Decision

1. **Delete the async queue and the built-in vector DB entirely** — code, tables,
   `/v1/queue*` + `/v1/vectors*` routes, DTOs, handlers, `athena queue`/`athena
   vectors` CLI, `queue_*`/`vector_*` config, RBAC permissions, tests, docs.
   *Delete, not disable:* a default-off toggle still ships the persistence
   capability in the binary; deletion removes it (strongest threat-model outcome,
   and matches "collapse out of the codebase").

2. **Delete the `/v1/store/export` + `/v1/store/stats` endpoints** — they exist to
   dump/stat the queue+vectors container; with those gone they have no purpose.
   `AthenaStore` (the type) stays for auth/audit/usage/allowlist.

3. **Add a stateless loopback mode.** When auth is off (loopback dev mode, no
   seeded users), the daemon creates **no `athena.sqlite`**: the model allowlist is
   read from config/in-memory, and audit/usage do not persist. With nothing to
   authenticate, audit, meter, or queue, there is **zero request-related
   data-at-rest** except the transient (hardened) temp upload files.

4. **Make `audit_log` + `usage_counters` opt-out / ephemeral.** In authed/installed
   mode they persist as today (operational necessity for a multi-tenant authed
   daemon); in stateless mode they are inert. A config switch controls persistence
   independent of mode.

5. **`model_allowlist` is removed entirely — see [ADR 026](decisions/026-retire-allowlist-store-is-registry.md).**
   Model availability becomes the model-store contents (classified by `ModelSupport`,
   ADR 021); the per-module default moves to TOML config; the validation gate
   becomes a store-presence check. This **supersedes** M42's "allowlist in SQLite"
   decision (the operator reversed it) and makes the stateless-loopback mode
   unconditional — with no allowlist in SQLite, the DB is needed only for
   auth/audit/usage.

6. **Eliminate the upload temp files by decoding from memory (Option D).**
   **[SHIPPED v0.10.199 — `InMemoryAsset.swift`; one shared `AVAssetReader` core
   over an `athena-mem://` resource-loader asset; all 6 serve-path call sites pass
   `Data`, temp staging deleted, boot sweep added. Decode-parity note: the
   resource-loader asset does not byte-sniff like `AVAudioFile`, so a magic-byte
   `sniffExtension` recovers the container for unnamed uploads.]** The
   audio/video upload bytes are written to `NSTemporaryDirectory()` today only
   because the *current* call sites use the file-based AVFoundation APIs
   (`AVAudioFile(forReading:)`, `AVAssetReader` over a file URL). AVFoundation is
   **not** actually limited to files: an `AVURLAsset` backed by an
   **`AVAssetResourceLoaderDelegate`** (custom URL scheme serving byte ranges from
   the in-memory upload `Data`) lets `AVAssetReader` demux **audio and video — and
   future video frames** — with **no file on disk**; Audio File Stream Services
   (`AudioFileStreamParseBytes` + `AudioConverter`) is the audio-only alternative.
   We feed Apple's own decoders from memory rather than writing a decoder. This
   *removes* the upload temp files (the data-at-rest win), keeps the same Apple
   decoders (no format-coverage or PCM-parity risk, hardware decode retained), and
   adds **no new dependency**. A startup sweep of any legacy `athena-*` files stays
   as migration insurance. Separable — can land before S1–S4.

### Amendments required on acceptance (recorded now, applied with implementation)

Per the architectural-decision-discipline rule, the conflicts are surfaced here
and resolved in the same edit window as implementation:

- **ADR 011 (thesis)** names queue + vectors as the *tenants proving the unified
  governor across modalities*. On acceptance, ADR 011 is amended to drop them as
  tenants — the multi-tenant story narrows to audio/embeddings/vision/video. The
  governor thesis stands on those modalities; queue/vectors were the weakest
  tenants (CPU bookkeeping, no Metal budget) and their removal does not weaken the
  Metal-governor claim.
- **ADR 013 + the CLAUDE.md "Stable `/v1/*` endpoints" list** mark
  `/v1/vectors*`, `/v1/store*`, `/v1/queue*` as **stable [native]**. Removal is a
  **breaking API change**; OpenAPISpec.swift, the surface list, and the spec↔routes
  drift-guard all update in the same edit (canonical-pipeline rule).
- **M42 (allowlist→SQLite)** is **superseded** by [ADR 026](decisions/026-retire-allowlist-store-is-registry.md):
  the allowlist is retired, not narrowed — availability = store contents, default
  = config. Removes the last non-auth/audit/usage reason for the DB to exist.

### Honesty boundary (binding)

- **Deletion removes the persistence *capability* from the binary** — the
  strongest possible outcome for queue/vectors. Nothing can re-enable on-disk job
  or vector storage without re-adding code.
- **Upload temp files are eliminated for the decode paths (Option D), but the
  bytes still live in process RAM.** Feeding Apple's decoders from the in-memory
  upload `Data` removes the on-disk staging entirely — so the crash-residue /
  forensic surface for uploads is *gone*, not merely hardened. What remains is the
  upload `Data` + decoded PCM in process memory during the request, which is the
  **live co-resident read** threat owned by ADR 024 Tier 1, not by this ADR.
  Stated, not papered over.
- **In authed mode, auth/audit/usage still persist by necessity** — a
  multi-tenant authed daemon cannot authenticate, audit, or meter without state.
  Stateless mode is the only configuration with zero such persistence.

### Rejected / deferred

- **Default-off toggle instead of deletion** — rejected; the capability (and thus
  the latent persistence path) would still ship. The operator chose deletion.
- **Replacement for async model ops** (`pull`/`convert`/`prune` formerly
  enqueued) — **RESOLVED at S2 implementation (v0.10.203): synchronous + SSE
  progress.** `POST /api/models/{pull,convert,prune}` now runs the op in-process
  and streams Server-Sent Events directly on the request — `data:
  {"event":"progress","fraction":F}` frames during the download, a terminal
  `{"event":"done","result":{…}}` or `{"event":"error","error":{message,type,code}}`
  frame, then `[DONE]` (long silent tails kept alive with `: keep-alive`
  comments). **No job id, no persistence** — the op's lifetime is the request's.
  The non-persisted in-memory job tracker (option b) was **rejected**: with the
  queue gone there is no durability story to preserve, and a tracker reintroduces
  per-op state (status-by-id, restart-loss semantics) for no benefit over a
  blocking stream. The `.modelWrite` RBAC gate and the M30 audit record are
  retained; the WebUI console (EventSource is GET-only) calls the same handlers
  synchronously and renders the terminal result. The CLI (`athena pull/convert/
  prune [--host …] [--follow]`) consumes the SSE stream; `--wait` is removed
  (there is no job to poll), `--follow` now means "print progress as it streams."
- **Deleting `AthenaStore` wholesale** — rejected; it is the auth/audit/usage/
  allowlist persistence layer.
- **A bespoke in-memory decoder in the Rust shim (symphonia + rubato)** —
  **rejected** for this daemon. It would replace Apple's decoders to achieve
  in-memory decode, but there is **no cross-platform driver** (the daemon is
  Apple-Silicon-only; only the client is cross-platform and it does not decode).
  It would take on an owned codec-coverage tax (AVFoundation tracks OS codec/HW
  support for free), a **PCM-parity re-validation gate** on every dep bump (a
  different decoder changes the bytes feeding the models), a decoder CVE/fuzzing
  surface, and Swift→C→Rust debugging — ~2.5–4 weeks vs Option D's ~1–1.5 weeks
  with none of that maintenance. Rust earns its place for the grammar shim
  (`llguidance` has no Apple equivalent); a media decoder is the opposite case.
  Revisit only if AVFoundation's in-memory APIs prove insufficient for our
  container/codec set (a half-day spike settles that).
- **Ephemeral-keyed encrypted volume for the temp files** (crypto-erase crash
  residue via an `hdiutil` AES image keyed only in RAM) — **superseded for the
  upload path** by Option D: with no file written, there is nothing to encrypt.
  The technique is retained on the shelf should any *future* path be forced to
  persist request bytes to disk.

## Consequences

- Breaking removal of three documented `/v1/*` native surfaces; consumers that
  used queue/vectors (none currently) must self-host them.
- The single-consumer/loopback deployment (the current case) persists **nothing**
  about requests except hardened transient temp files.
- The thesis narrows but its core (one Metal memory budget across audio/
  embeddings/vision/video) is unaffected.
- `AthenaStore` shrinks to the auth/audit/usage/allowlist tables; SQLCipher
  at-rest coverage (M34) now applies to a smaller, credential-and-metadata-only DB.

### Validation (on implementation)

- Decision logic (mode selection: when is the DB created? when does audit/usage
  persist?) is MLX-free and unit-pinned (ADR 008/009).
- e2e: stateless loopback run creates no `athena.sqlite`; authed run still does.
- Drift-guard: removed routes are absent from both OpenAPISpec.swift and the live
  router; RBAC permissions for the removed features are gone.
- Regression: a crash mid-transcription leaves no readable temp upload file after
  the startup sweep.

Plan + slices: `docs/collapse-persistence-plan.md` (on approval).
