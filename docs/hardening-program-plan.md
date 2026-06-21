# Hardening program — data-at-rest + data-in-use (ADRs 024 / 025 / 026)

**Status:** Proposed program, awaiting go. Umbrella over three interlocking,
already-written ADRs. This doc sequences them into one dependency-ordered execution
plan; the per-decision detail lives in the ADRs and `collapse-persistence-plan.md`.

## Goal

Turn Athena into a **near-stateless oracle**: minimize what the daemon persists
about requests, and protect what is unavoidably resident.

- **Data-at-rest** (ADR 025 + 026): delete the persistent data tenants (queue,
  vector DB, allowlist) and the temp-upload residue; in loopback/no-auth mode the
  daemon persists **nothing** about requests.
- **Data-in-use** (ADR 024): raise the bar against a **co-resident malicious
  process** scraping the address space (binary lockdown + side-channel hygiene),
  with the honest ceiling documented.

Net target posture: *loopback ⇒ no `athena.sqlite`, no upload temp files; the
binary is non-root-unscrapable; the only residual is the active working set in
RAM, owned by the ADR-024 honesty boundary.*

## Inputs (read these first)

- [ADR 024](decisions/024-in-memory-data-protection-coresident-threat.md) — co-resident threat, defense ladder T1–T3.
- [ADR 025](decisions/025-collapse-persistent-data-tenants.md) — delete queue + vectors + `/v1/store`; stateless mode; Option D temp-file elimination.
- [ADR 026](decisions/026-retire-allowlist-store-is-registry.md) — retire allowlist; store-is-registry; TOML defaults; ambiguity → 400.
- [collapse-persistence-plan.md](collapse-persistence-plan.md) — slices S1–S7.

## Open decisions — RESOLVED at kickoff (2026-06-20)

1. **ADR 024 tier scope → T1 + T2 committed; T3 deferred** as a stretch (gated on
   whether the deployment runs the prompt cache with sensitive idle entries).
2. **Async model-ops replacement → synchronous + SSE progress** (no persistence;
   matches passive-oracle). Applied at S2.
3. **Consumer confirmation → confirmed dropped.** The downstream consumer no
   longer uses the queue or the vector DB; the breaking removals S1/S2 are cleared.

## Phase 0 spike results (2026-06-20) — both green

- **Spike A (S5 in-memory decode): PASS.** An `AVAssetResourceLoaderDelegate`-backed
  `AVURLAsset` (custom `athena-mem://` scheme) feeds `AVAssetReader` with no file on
  disk, decoding wav/mp3/m4a-aac/flac/ogg-opus/mp4/mov to 16 kHz mono Float32 with
  RMS parity (3–4 decimals) vs the daemon's current file-based path; sample-count
  deltas are codec priming/padding only. No format regressed → no Audio File Stream
  Services fallback needed. Note: the audio path migrates `AVAudioFile` →
  `AVAssetReader` (file-only API can't take a custom scheme); video already uses
  `AVAssetReader`, so both chokepoints unify on one in-memory mechanism.
- **Spike B (T1 process lockdown): PASS.** Re-signing the real `athena` with the
  Hardened Runtime + `get-task-allow=false` flips a debugger attach from "reaches
  the task" (baseline adhoc) to `error: attach failed: Not allowed to attach to
  process` (AMFI denial), and MLX's `default.metallib` still loads under the
  Hardened Runtime (27B model loaded + `/v1/chat/completions` returned) — clearing
  the M43.3 risk. The binary is effectively statically linked (one weak platform
  dylib), so library validation is a non-issue; no `disable-library-validation` /
  `allow-jit` entitlement is granted. Notarization can't be exercised locally (no
  signing identity) → operator-side release step (`NOTARIZE=1` + `CODESIGN_IDENTITY`
  + `NOTARYTOOL_PROFILE` in `deploy/build.sh`).

## Phases (dependency-ordered)

Each slice is a test-pinned commit. House rules: pre-commit pipeline **Tests →
Security → Quality → Refactor**; `appVersion` bump **in** the slice commit;
direct-to-main + annotated semantic tag per slice (release workflow); all route
changes update `OpenAPISpec.swift` in the same edit; errors use the `{"error":…}`
envelope; MLX-free decision logic unit-pinned (ADR 008/009).

### Phase 0 — De-risk & unblock (spikes + decisions; ~1–2 days)
- **Spike A (S5):** confirm an `AVAssetResourceLoaderDelegate`-backed `AVURLAsset`
  feeds `AVAssetReader` across the real container/codec set (mp3/m4a-aac/wav/flac/
  ogg/mov+mp4). Fallback: Audio File Stream Services for bare audio. (~½ day)
- **Spike B (T1):** build a **hardened-runtime + notarized + no-`get-task-allow`**
  `athena` and confirm MLX's metallib bundle still loads (resolve the M43.3
  breakage). This is T1's gating risk. (~½–1 day)
- **Resolve** the three open decisions above.

### Phase 1 — Independent hardening wins (low-risk, parallelizable)
- **S5 — Option D in-memory decode.** Feed Apple decoders from the in-memory upload
  `Data`; delete the ~7 temp-write/`defer` sites; preserve the MLX-free bounds +
  error taxonomy. Removes the upload crash-residue surface.
- **T1 — process lockdown.** Hardened runtime, notarization, drop `get-task-allow`,
  no debugger entitlement; `deploy/build.sh` + signing/notarization step.
- **T2 — side-channel hygiene.** `setrlimit(RLIMIT_CORE,0)`, selective `mlock`,
  `PT_DENY_ATTACH`, best-effort zeroize-on-evict. **Coordinate with S5** — the
  decode-buffer no-swap-spill items overlap; land them once.

### Phase 2 — Persistence removal (leaf removals)
- **S1 — Vector DB out.** Delete `VectorStore`, `vectors` table, `/v1/vectors*`,
  CLI, config, RBAC, tests, docs.
- **S3 — Store endpoints out.** Drop `/v1/store/export` + `/v1/store/stats`; keep
  `AthenaStore` for auth/audit/usage.
- **S7 — Retire allowlist (ADR 026).** Delete `model_allowlist` + CRUD +
  `/api/models/allow*` + WebUI mirrors + `athena allowlist` (incl. M43.5 offline);
  availability = store contents via `ModelSupport`; default → per-module TOML
  (`athena default --module M`); ambiguity → 400 `ambiguous_model`. **Before S4.**
- **S2 — Queue out** (after decision #2). Re-home model ops, then delete
  `RequestQueue`/`QueueWorkerService`, `jobs` table, `/v1/queue*`, CLI, config,
  RBAC, `/ui` dashboard hook, docs.

### Phase 3 — Stateless mode + reconciliation
- **S4 — Stateless loopback mode + audit/usage opt-out** (depends on S7). Auth-off ⇒
  create no `athena.sqlite`; audit/usage persistence switch; doctor reports mode.
- **S6 — Surface + thesis reconciliation.** Update the CLAUDE.md "Stable `/v1/*`"
  list + OpenAPI drift-guard; **amend ADR 011** (drop queue/vectors as governor
  tenants) and **ADR 013** (remove the 3 native surfaces) in the same edit window;
  full e2e gate.

### Phase 4 — Stretch (only if decision #1 includes them)
- **T3 — encrypt idle prompt-cache KV at-rest-in-RAM** (AES-GCM/SEP, decrypt-on-
  restore). Collapses the plaintext window to the entry currently decoding.
- **Build attestation** — signed/inspectable build manifest + exposed binary
  measurement (the PCC-shaped, code-not-memory piece). Separable.

## Coordination points / hazards
- **S5 ↔ T2:** decode-buffer no-swap-spill / `RLIMIT_CORE` overlap — implement once.
- **T1 ↔ M43.3:** hardened runtime previously broke metallib lookup; Spike B must
  clear this before T1 ships.
- **S7 before S4:** S4's unconditional "no DB in loopback" depends on the allowlist
  no longer needing the DB.
- **S2 breaking + consumer:** do not land S1/S2 before decision #3.
- **Honesty boundary (binding, ADR 024):** none of this defends a kernel/SIP-off
  adversary or the live RAM working set beyond T1's bar; say so, don't overclaim.

## Done = 
Loopback run creates no `athena.sqlite` and no upload temp files; a non-root
debugger-entitled process cannot `task_for_pid` the daemon; `/v1/queue*`,
`/v1/vectors*`, `/v1/store*`, `/api/models/allow*` are gone from the spec and the
router; model selection works off the store with TOML defaults and a 400 on
ambiguity; ADR 011/013 amended; full e2e green.
