# Backlog hitlist — unfinished / forgotten / quietly-dropped work

**Audit date:** 2026-06-19 · **Code state:** appVersion `0.10.186` (tags through `v0.10.186`)
**Method:** reconciled every ADR (001–023), plan/milestone doc (m46–m73, api-surface,
governor-truth, etc.), readiness ledger, and memory topic file against the actual
`Sources/`, `Tests/`, `deploy/`, and git tags. **Status labels were not trusted** — each
verdict cites a doc claim *and* a code path. This is for operator review; nothing here is
auto-actioned, and anything "substantial" routes through the brownfield change gate
(design + ADR), not "just done".

**Counts:** DO NOW 1 · DO SOON 12 · DEFER 23 · DROP 11.
**Shipped since audit (v0.10.193–197):** #3 K7 argv-secret leak, #7 `athena show` vision
line, #11 allowlist offline `--data-dir`, #12 auth-deny `hint`, #13 K2 SSE status check.
**Top 3 DO-NOW / first-to-schedule:** (1) reconcile stale ADR/plan status labels — three
docs claim "Proposed / no code yet" for *shipped* work, one is a dependency of a later
shipped ADR; (2) the governor accounting-truthfulness work (ADR 023) is the project's own
stated #1 priority but sits Proposed while 8 later feature ADRs shipped ahead of it; (3) the
`security … -w <value>` argv secret leak (K7) — the exact footgun ADR 005 set out to kill,
still live on an adjacent code path.

---

## 1. Summary table (DO NOW → DO SOON → DEFER → DROP)

| # | Item | Source | Verified state | Class | Effort | Risk | Gate / dependency |
|---|------|--------|----------------|-------|--------|------|-------------------|
| 1 | Reconcile stale ADR/plan status labels (016, 017, 007, 001, 008, 009; m50/convert-quant plans) | ADRs + plans | Docs say "Proposed/no-code/Not started"; code shipped | **DO NOW** | S | none | — (doc-only) |
| 2 | Governor accounting truthfulness (ADR 023): bound serve MLX cache, reconcile admission vs real allocator, measure per-model footprint | ADR 011→023; memory `heartbeat-rss-undercounts-gpu`, `2026-06-05-watchdog-panic` | Admission still meters RSS; serve sets no `cacheLimit`; #2 conflicts w/ M5.5 | **DO SOON** | M–L | med–high | ADR 023 review + amendment (RSS-vs-activeMemory conflict) |
| 3 | K7 — `security … -w <value>` argv secret leak | ADR 005; `audit-remediation-plan.md:128` | **DONE v0.10.193** — secret fed on stdin (`-w` at end), never argv | **DONE** | S–M | low | — |
| 4 | `diarized_json` response_format on `/v1/audio/transcriptions` (#4a) | ADR 013 §3; `m35-readiness:67` | Absent — enum lacks it, falls through to `{text}` 0-spk | **DO SOON** | M | low | none (shape published) |
| 5 | Honor `logprobs`/`top_logprobs` on greedy path (stop 400-ing) | ADR 013 §4 | Code does the **opposite** — still 400s | **DO SOON** | M | low | none (ratified) |
| 6 | Native `/api/embed` per-principal metering | ADR 007 #8; `m35-readiness:54` | `handleNativeEmbed` drops `meter()` | **DO SOON** | S | low | none |
| 7 | `athena show` vision-capability line (M71.3) | ADR 010; `m71-vision-input-plan.md:86` | **DONE v0.10.194** — `show` prints `vision: yes/no` from `hasVisionConfig` | **DONE** | S | low | — |
| 8 | M70.3 CI test-debt backfill (ND14/ND15 whisper-lang/RMS/clustering) | `audit-remediation-plan.md:305` | Unchecked, no test file | **DO SOON** | S–M | low | none |
| 9 | Finish ADR 013 §5 error reclassification (rebind catch-alls + queue-submit) | ADR 013 §5 | Audio/template legs shipped; rebind/queue legs unconfirmed | **DO SOON** | S | low | none |
| 10 | Substrate fork+pin to a remote (release blocker R1) | `release-distribution.md:17` | `Package.swift:71` still `path: "../mlx-swift-lm"`; un-pushed delta grew | **DO SOON** | M | low | user creates org fork repo |
| 11 | Allowlist offline `--data-dir` path (usability #3 / M43.5) | `usability-audit:52`; `m43` memory | **DONE v0.10.195** — macOS `allowlist` verb opens AthenaStore directly when `--data-dir` given | **DONE** | M | low | — |
| 12 | Auth-deny envelopes carry no recovery `hint` (usability #5) | `usability-audit:85` | **DONE v0.10.196** — middleware 401/403 hint already shipped (M43.4); filled the in-handler `deny403` gap | **DONE** | S | low | — |
| 13 | K2 — client SSE readers don't check HTTP status (lone "High" in M71 tail) | `audit-remediation-plan.md:312` | **DONE v0.10.197** — fixed the two queue-events SSE followers; `RemoteLogs` already guarded | **DONE** | S | low | — |
| 14 | Token-budget quotas (#9) | ADR 007 #9; `m35-readiness:61` | Absent; counters exist, no admission check | DEFER | M | low | commercial-host posture / consumer; ADR-007 milestone never scheduled |
| 15 | Parakeet-as-default transcription | ADR 020 open-decision #2 | Whisper default; **no WER A/B harness exists** | DEFER | M | low | a real WER+throughput A/B on a labelled set |
| 16 | M78.2 video descriptions (`POST /v1/video/descriptions`) | ADR 022 | Genuinely absent (route + frame sampler) | DEFER | L | med | downstream consumer requirements |
| 17 | Remove deprecated `/api/chat`+`/api/embed` (breaking) | ADR 013 §1(d) | Both still served, marked `deprecated:true` | DEFER | S | med | deprecation window + consumer migration |
| 18 | `reasoning_effort` alias | ADR 013 §7 | Absent (silently dropped) | DEFER | S | low | a consumer drives it |
| 19 | Audio-in-chat (issue #5) | ADR 010; `audio-input-chat-plan` | Audio tower stripped; no `input_audio` part | DEFER | L | high | upstream `mlx-swift-lm#207` + consumer |
| 20 | DFlash structured output (M63.4) / temp>0 / kernels / GDN targets | ADR 001 | Engine unguided/greedy/pure-mx only | DEFER | M–L | med | Guide peek-advance shape; perf follow-up; no demand |
| 21 | M51 symmetric Guide-mask + M52.C sampled structured | `m51-plan`, `m52-plan` | Absent; `samplingEligible` gates on `schemaJSON==nil` | DEFER | S | low | only if sampling precondition relaxed |
| 22 | M52.A TurboQuant #909 perf kernels | `m52-plan:53` | Absent; `kv_compression` default `.none` | DEFER | L | med | TurboQuant-as-default / long-context workload |
| 23 | M52.B TriAttention calibrated-trig scoring | `m52-plan:131` | Norm-only; no `athena calibrate` | DEFER | L | med | aggressive-eviction long-multi-turn workload |
| 24 | M59.5 prompt-cache disk persistence | `m59-prompt-prefix-cache:153` | No disk tier in `PrefixKVCache.swift` | DEFER | M | low | cross-restart reuse pattern |
| 25 | M59.6 scoped prompt-cache "shred" | `m59-6-prompt-cache-shred:1` | Spec-only, not built | DEFER | M | med | data-minimization / erasure requirement |
| 26 | M61 prefix-affinity queue scheduling | `m61-prefix-affinity-queue:3` | Spec-only; jobs table lacks cache_key/model | DEFER | M | low | multi-tenant queue contention + client fan-out |
| 27 | At-rest key rotation (#7, `store rekey`) | `m35-readiness:65` | No `store rekey`; rekey only in vendored sqlite3.c | DEFER | M | med | operator need; no consumer demand |
| 28 | Streaming-to-disk upload decode | ADR 017 | Handlers buffer whole body in memory | DEFER | L | med | observed memory pressure from concurrent large uploads |
| 29 | Client `https` (K1/K8) | ADR 004 | Client hardcodes `http://` (`DaemonOptions.swift:22`) | DEFER | M | low | untrusted-network exposure milestone |
| 30 | Fail-closed TLS / `--insecure` | ADR 004 | Warn-only by design | DEFER | M | low | accepted posture; same milestone as 29 |
| 31 | VLM constrained-decoding compose; VLM/LLM governor double-count; cold-load slot-outside-gate | ADRs 010, 015 | Open "if it bites" follow-ups | DEFER | M | med | the named symptom actually observed |
| 32 | GPU/Metal watchdog-wedge host recovery | `2026-06-05-watchdog-panic` | No host-level GPU-hang recovery | DEFER | L | high | live reproduction to pin |
| 33 | swift-huggingface temp fork revert | `swift-huggingface-fork:26` | `Package.swift:96` still `path:` fork | DEFER | S | low | upstream PR #50 merge + new release tag |
| 34 | Port-7447 cross-repo contract (an internal codename/an internal codename/an internal codename) | `athena-port-contract-followup` | an internal codename `environment.md` still says Ollama 11434 | DEFER | S | low | user-driven, cross-repo |
| 35 | DiffusionGemma + "one governor, two embodiments" ADR | `diffusiongemma-support-investigation` | No ADR, no runtime | DEFER | L | n/a | research / upstream; can't do structured output |
| 36 | bf16 unquantized decode cliff (~2.8 tok/s) | `decode-throughput-gap-analysis` | No fix; substrate/MLX kernel issue | DEFER | L | low | Metal GPU capture; upstream. Use 8-bit |
| 37 | Release pipeline R2–R7 (version-from-tag, .pkg, GH Actions, cask, `athena update`, signing) | `release-distribution:54` | `appVersion` hardcoded; no workflow | DEFER | L | med | blocked on #10 (R1) + user repos + Apple acct |
| 38 | File MLX upstream issue draft | `mlx-issue-draft-…:8` | "Not yet posted" | DEFER | S | none | external action (relates to #36) |
| 39 | Quantized-embedder convert route | ADR 016 | Rejected-for-now | DEFER | M | low | consumer needs quantized embedders staged |
| 40 | TTS `/v1/audio/speech` | ADR 013 don't-build | Absent (correct) | DEFER | M | low | a consumer wants audio output |
| 41 | GPU-compute-extension governance model | ADR 011 | Explicitly unresolved | DEFER | L | n/a | a second accelerated compute workload arrives |
| 42 | NF3 TriAttention numeric parity CI | `audit-remediation-plan.md:304` | Unchecked | DEFER | S | low | MLX-in-CI limitation (ADR 009) — gated by design |
| 43 | M60.4 fan-management spike | `m60-plan:114`; `fan-management-idea` | Never acted | **DROP** | — | — | value gone (root cause was sleep, fixed); root-only/fragile |
| 44 | K13 — `--key` in argv | `audit-remediation-plan.md:128` | **Non-existent** — only help text refs | **DROP** | — | — | no live argv vector; close tracker line |
| 45 | ADR 007 #8 native `/api/chat` metering | ADR 007 status | **Done** (`AthenaServer.swift:3759`, v0.10.140) | **DROP** | — | — | superseded by code; embed remains (#6) |
| 46 | ADR 008 M70.1b extraction | ADR 008 §Consequences | **Done** (`AthenaDeploy` target + tests) | **DROP** | — | — | shipped; doc lag only |
| 47 | Dangling HF-symlink doctor check | `hf-symlink-dangling-store-entries` | **Done** v0.10.142 (`Doctor.swift:119`) | **DROP** | — | — | shipped; startup-warning half intentionally skipped |
| 48 | System-sleep power assertion (M60.2) | `m60-throughput-decay-investigation` | **Done** (`PowerAssertion.swift`, committed) | **DROP** | — | — | shipped |
| 49 | ADR 019 Parakeet mel int16 shortcut | ADR 019 | Superseded by ADR 020 S2 reference-exact L1 mel | **DROP** | — | — | closed |
| 50 | Usability #1/#7/#8 (config-set reinstall, /healthz signals, /ui/allowlist) | `usability-audit` | **Done** (M43.4 / M44.1) | **DROP** | — | — | shipped; stale audit lines |
| 51 | `structured-rss-rootcause` index-cache | `structured-rss-rootcause.md` | Superseded by M53 llguidance swap | **DROP** | — | — | engine replaced |

---

## 2. Per-item cards — DO NOW / DO SOON

### #1 (DO NOW) — Reconcile stale ADR/plan status labels
- **What:** Several docs assert a status contradicted by shipped code.
- **Why it matters:** ADR 016 reads *"Proposed (M73) — awaiting operator approval"* yet it shipped
  (v0.10.166) **and is a dependency of the shipped ADR 021** ("S2 subsumes … ADR-016 pattern") — a
  reader could re-debate an approved, load-bearing decision. Stale labels are exactly what this audit
  exists to stop resurfacing.
- **Evidence (doc → code):**
  - ADR 016 `:3` "awaiting operator approval" → shipped; `ModelClass`/redirect live, ADR 021 builds on it.
  - ADR 017 `:3` "Proposed — pending operator review (no code yet)" → fully shipped (`AthenaServerKit/ExpectContinueHandler.swift`, `UploadLimit.swift`, `maxAudioUploadBytes` config, 413 in spec).
  - ADR 007 status "Not yet implemented" → `/api` metering shipped (`AthenaServer.swift:3759`, v0.10.140); only quotas remain.
  - ADR 001 §Status lists "M63.1–M63.4 SHIPPED" → M63.4 *structured-output* deliverable did **not** ship (`DFlashGeneration.swift:23`).
  - ADR 008 §Consequences describes M70.1b as pending → landed (`AthenaDeploy` target).
  - ADR 009 status "Implementing" → all nine CI seams + tests exist.
  - `m50-plan.md:8` "Not started" → M50.1–.5 shipped (v0.10.80); `convert-quant-arch-gate-plan.md:3` "do not implement yet" → shipped v0.10.165; `m72-vision-convert-plan.md:88` 500-wart → now 400 (`AthenaError.swift:135`).
- **Proposed first slice:** Edit the status line / §Status of each ADR to Accepted+version (016, 017, 007-partial, 008, 009), correct ADR 001's M63.4 line, and mark the three plan headers Shipped. **No code change.**
- **Test:** none (doc). Optionally extend the existing OpenAPI/route drift-guard mentality with a lint that an ADR marked "Proposed" has no shipped version stamp — out of scope here.
- **Effort:** S · **Risk:** none · **Deps:** none.

### #2 (DO SOON) — Governor accounting truthfulness (ADR 023)
- **What:** Make the Metal governor's view true and bound the serve-path MLX buffer cache.
- **Why it matters:** ADR 011 names this the **next milestone, "promoted above feature work"** — yet 8
  later feature ADRs (015–022) shipped while it stayed Proposed. Field evidence: `phys_footprint` hit
  96 GiB (= whole budget), cache peaked **over** budget, plausibly linked to the 2026-06-05 watchdog
  reboot. It is the load-bearing wall of the project's stated moat.
- **Evidence (doc → code):**
  - Admission reconciles against an RSS-delta probe, not the allocator: `MemoryGovernor.swift` reconcile + `Load.swift:473-475` (`processResidentBytes()`).
  - Serve path sets **no** `MLX.Memory.cacheLimit`; only `ModelConvert.swift:240-242` caps it (256 MB, convert-only). No `mlx_cache_limit_bytes` config key.
  - `mlxActiveBytes`/`mlxCacheBytes` are exposed for *observability* (`AthenaServer.swift:1565,6098`) but the control loop never consumes them.
  - **CONFLICT to resolve first:** `Load.swift:459-470` (M5.5) *deliberately* uses RSS because `activeMemory` "MISSES file-backed mmaps … delta is ~120 MB even though the process holds ~8 GB of weights." ADR 023 fix #2 ("count activeMemory for admission") and fix #3 ("measured load-time activeMemory delta") therefore **contradict the standing M5.5 rationale** for mmap-backed tenants (embedder/whisper/diarizer/speaker).
- **Proposed first slice:** (a) Approve/amend ADR 023 to reconcile the RSS-vs-activeMemory conflict —
  specify joint accounting of reclaimable cache (one global number) + mmap'd weights; then (b) ship the
  cheap, safe lever first: configurable `mlx_cache_limit_bytes` bounding the serve cache (the seam
  already exists in convert). Defer #2/#3 admission changes until the amendment lands.
- **Test:** MLX-free unit pins per ADR 008/009 (cache-limit resolution + admission decision algebra); heavy numeric path stays env-gated.
- **Effort:** M (cache-bound) → L (admission) · **Risk:** med–high (governs admission; the moat scenario) · **Gate:** brownfield change gate — ADR 023 review + amendment.

### #3 (DO SOON) — K7: `security … -w <value>` argv secret leak
- **What:** The Keychain writer passes the secret as an argv flag to `/usr/bin/security`.
- **Why it matters:** It is the precise footgun ADR 005 set out to eliminate (`--password` in argv),
  surviving on an adjacent code path. The secret is visible to any local `ps`/process audit during the spawn.
- **Evidence (doc → code):** `audit-remediation-plan.md:128` "K7 … `security -w <value>` argv leak …
  deferred to M71" → still live: `Credentials.swift:57` (`add-generic-password … -w value`) and `:107`
  (`sudo -u … security add-generic-password … -w value`). (Line 46 is a `-w` *read* flag with no value —
  not a leak.) M71 shipped vision (ADR 010/012), so K7 lost its assigned home.
- **Proposed first slice:** Replace the `security add-generic-password -w <value>` spawns with the
  Keychain `SecItem*` API (no argv), or feed the value via stdin. Local exposure only, so S-sized.
- **Test:** unit/integration round-trip on store/read of a secret; assert no secret appears in a captured argv.
- **Effort:** S–M · **Risk:** low · **Deps:** none. Re-home the tracker off the dead "M71".

### #4 (DO SOON) — `diarized_json` response_format (#4a)
- **What:** Add `diarized_json` to `POST /v1/audio/transcriptions` `response_format`.
- **Why it matters:** ADR 013 §3 ratified it with a published shape; the documented consumer
  (the consuming application) sends it and **silently falls through to plain `{text}` with 0 speakers**.
- **Evidence (doc → code):** ADR 013 §3 / `m35-readiness:67` → `OpenAPISpec.swift:170` enum is
  `["json","text","srt","vtt","verbose_json"]`; `AthenaServer.swift:1827,2042` switch has no
  `diarized_json` case; **0 hits** for `diarized_json` in `Sources/`; `the consuming application-integration.md:282`
  confirms the silent fall-through.
- **Proposed first slice:** Add the enum value + a switch case that runs the existing `diarize=true`
  segment+speaker path and serializes the published shape; mark native in the op `description` per the `/v1` rule.
- **Test:** route test asserting `response_format=diarized_json` returns speaker-tagged segments (not `{text}`); OpenAPI drift-guard picks up the new enum.
- **Effort:** M · **Risk:** low · **Deps:** none.

### #5 (DO SOON) — Honor `logprobs`/`top_logprobs` on the greedy path
- **What:** Stop 400-ing `logprobs`/`top_logprobs`; emit the OpenAI logprobs response object.
- **Why it matters:** ADR 013 §4 ratified the reversal of M31's reject-behavior ("the greedy/structured
  path already computes the logits … stop 400-ing them"). Code still does the **opposite** — a direct
  contradiction of a ratified decision.
- **Evidence (doc → code):** ADR 013 §4 → `OpenAIDTO.swift:242-248` `unsupportedParameter()` still returns
  `"logprobs"/"top_logprobs"` → 400 at `AthenaServer.swift:907`; `OpenAPISpec.swift:92,994,995` still
  documents "Rejected with 400".
- **Proposed first slice:** Remove the two params from `unsupportedParameter()`; on the greedy path,
  capture top-k logits already computed and serialize the OpenAI `logprobs` object; update the spec text.
- **Test:** route test: `logprobs:true` → 200 with a populated `logprobs` object; `n>1`/`logit_bias` still 400.
- **Effort:** M · **Risk:** low · **Deps:** none.

### #6 (DO SOON) — Native `/api/embed` per-principal metering
- **What:** Meter `/api/embed` like `/api/chat` and `/v1/embeddings`.
- **Why it matters:** ADR 007 #8 committed native metering; `handleNativeChat` was fixed (M59.3) but the
  embed twin was missed, so embed traffic is invisible to usage counters (and to any future quota).
- **Evidence (doc → code):** `m35-readiness:54` → `handleNativeEmbed` (`AthenaServer.swift:3820`) calls
  `governedEmbed` and returns with **no `meter()`** (contrast `handleNativeChat` at `:3759`).
- **Proposed first slice:** Compute `TokenUsage` from the embed batch and call `meter(principal:usage:)`
  with the resolved principal, mirroring chat.
- **Test:** route test asserting `usage_counters` increments after an `/api/embed` call. **Effort:** S · **Risk:** low.
- **Note:** `/api/embed` is deprecated (ADR 013), so low urgency — but it's a one-handler fix and a #14 prereq.

### #7 (DO SOON) — `athena show` vision-capability line (M71.3)
- **What:** Surface `servesVision` in `athena show`.
- **Why it matters:** Request-side gating works, but an operator has no way to see whether a resident
  model serves images — the last unshipped slice of the vision-input milestone.
- **Evidence (doc → code):** `m71-vision-input-plan.md:86` / ADR 010 `:3` "M71.3 … pending" → `RmShow.swift:46-82`
  prints model/path/size/type/support, **no vision line**, though `servesVision` exists on the module.
- **Proposed first slice:** Add a "vision: yes/no" row driven by `servesVision`/`ModelSupport`.
- **Test:** CLI snapshot test for a VLM vs a text-only model. **Effort:** S · **Risk:** low.

### #8 (DO SOON) — M70.3 CI test-debt backfill (ND14/ND15)
- **What:** Add the unbuilt MLX-free decision-seam tests: whisper language round-trip, RMS gate, clustering branches.
- **Why it matters:** Cheap coverage on already-shipped correctness fixes; consistent with the M70 stub-CI discipline (ADR 009). Not numerically gated — buildable today.
- **Evidence (doc → code):** `audit-remediation-plan.md:305` `[ ] ND14/ND15 …` → no matching test file.
- **Proposed first slice:** Add unit tests against the existing pure-Swift seams. **Test:** the tests are the deliverable. **Effort:** S–M · **Risk:** low.
- *(NF3 TriAttention numeric parity is intentionally MLX-gated — see Don't-touch #42.)*

### #9 (DO SOON) — Finish ADR 013 §5 error reclassification
- **What:** Confirm/finish 4xx cause-naming for the 4× per-module rebind catch-alls and queue-submit passthrough.
- **Why it matters:** ADR 013 §5 reclassifies client/precondition faults 500→4xx; the audio + chat-template legs landed (`AthenaError.swift:125-169`) but the rebind/queue legs were not separately verified.
- **Evidence (doc → code):** ADR 013 §5 → audio/template mapped+wired; rebind/queue legs unconfirmed (no cause-naming case traced).
- **Proposed first slice:** Audit the embed/transcribe/diarize/speaker rebind handlers + queue submit; route failures through the classify seam. **Test:** route tests asserting 4xx + named code on a bad-model rebind. **Effort:** S · **Risk:** low.

### #10 (DO SOON) — Substrate fork+pin (release blocker R1)
- **What:** Push the local `mlx-swift-lm` clone to a remote and pin `Package.swift` to it.
- **Why it matters:** The build is **not reproducible off this machine** — `Package.swift:71` path-deps a
  local clone carrying un-pushed deltas (3 TurboQuant + the C11 seeded-sampling commit; the delta has
  *grown* since the memory was written). Blocks the entire release pipeline (#37) and any CI.
- **Evidence (doc → code):** `release-distribution.md:17` → `Package.swift:71` `.package(path: "../mlx-swift-lm")`; cf. the already-correct `AppleSiliconMetrics` git-URL pin at `:100`.
- **Proposed first slice:** Operator creates the org fork repo; push the deltas; switch the dep to `url+revision`. **Test:** clean-clone build on a second checkout. **Effort:** M · **Risk:** low · **Gate:** user must create/authorize the fork repo.

### #11 (DO SOON) — Allowlist offline `--data-dir`
- **What:** Let `athena allowlist` edit the SQLite allowlist directly when the daemon is down / no token.
- **Why it matters:** Operability — an operator with a wedged daemon currently cannot fix the allowlist (HTTP-only). This is also the deferred M43.5 follow-up.
- **Evidence (doc → code):** `usability-audit:52` → `RemoteAllowlist.swift:168-220` is HTTP-only, no `--data-dir`/AthenaStore path.
- **Proposed first slice:** Add a `--data-dir` branch that opens AthenaStore directly (reuse the offline pattern). **Test:** CLI test mutating the allowlist with the daemon stopped. **Effort:** M · **Risk:** low.

### #12 (DO SOON) — Auth-deny `hint`
- **What:** Add a recovery `hint` to 401/403 envelopes.
- **Why it matters:** Operator legibility — a denied call names the missing permission/recovery path instead of a bare 403.
- **Evidence (doc → code):** `usability-audit:85` → **stale**: the route-middleware deny
  (`AthenaServerKit/Auth.swift`) already emits `hint` on 401 **and** 403 (M43.4); the genuine
  gap was the in-handler `AthenaServer.deny403` RBAC/safety guards, which routed through
  `error()` with no hint. **Done v0.10.196**: added optional `hint` to the error envelope +
  `deny403` default, e2e-asserted on both the middleware 403 and the last-admin guard.
- **Proposed first slice:** Extend the deny envelope with an optional `hint` (no schema break — additive). **Test:** route test asserting the hint on a permission-denied call. **Effort:** S · **Risk:** low.

### #13 (DO SOON) — K2: client SSE readers check HTTP status
- **What:** Ensure every client SSE reader checks the HTTP status before parsing the stream.
- **Why it matters:** The lone remaining "High" in the M71 audit tail — a non-200 (e.g. 503/4xx) is currently mis-parsed as an event stream, so the client silently swallows real errors.
- **Evidence (doc → code):** `audit-remediation-plan.md:312` (K2) → `RemoteLogs.swift:102` already checks status (partial); audit the other SSE consumers (`RemoteQueue`/events) for the same guard.
- **Proposed first slice:** Add the status check to each SSE follow path. **Test:** client test feeding a 503 to an SSE call → surfaces an error, not empty stream. **Effort:** S · **Risk:** low.

---

## 3. Drift & contradictions (doc ↔ code mismatches)

Highest-value output. Recommended reconciliation in **bold**.

1. **ADR 016 status** says "Proposed — awaiting operator approval" but it shipped (v0.10.166) **and ADR 021 (shipped) depends on it**. → **Flip to Accepted+version.** (#1)
2. **ADR 017 status** says "Proposed — pending review (no code yet)" but the *entire* ADR is implemented. → **Flip to Accepted+version.** (#1)
3. **ADR 007 status** "Not yet implemented" but `/api` metering (#8) shipped (`AthenaServer.swift:3759`); only quotas (#9) remain. → **Change to Partially-Implemented (metering done, quotas open).** (#1, #6, #14)
4. **ADR 013 §4 vs code (hard contradiction):** ADR says *stop* 400-ing `logprobs`; code still 400s and the spec still documents the rejection. → **Implement §4 (#5) or, if reversing the decision, amend ADR 013.**
5. **ADR 013 §3 vs code:** `diarized_json` ratified but never built; consumer silently degraded. → **Implement (#4).**
6. **ADR 023 fix #2/#3 vs M5.5 rationale (design conflict):** "count `activeMemory` for admission/footprint" contradicts the deliberate RSS choice at `Load.swift:459-470` (activeMemory misses mmap'd weights). → **Amend ADR 023 to specify joint cache+mmap accounting before building (#2).**
7. **ADR 001 §Status** lists M63.4 SHIPPED, but the M63.4 structured-output deliverable is still deferred (`DFlashGeneration.swift:23`). → **Correct the status line (#1).**
8. **ADR 005 "tracked in M71"** is stale: K7 still live (`Credentials.swift:57,107`); K13 is *non-existent* (no `--key` argv). → **Re-home K7 (#3); delete the K13 claim (#44).**
9. **ADR 008 §Consequences** describes M70.1b as pending; it landed (`AthenaDeploy` + tests). → **Update §Consequences (#1).**
10. **Memory `m35-readiness` says `handleNativeChat` unmetered** → now metered (M59.3); only `/api/embed` remains. → **Update memory; ship #6.**
11. **Memory `heartbeat-rss-undercounts-gpu` implies all-RSS-blind** → the prompt-cache *relief* path now uses `phys_footprint` (M60.6); only *admission* still RSS. → **Note the partial fix in ADR 023.**
12. **Memory `2026-06-05-watchdog-panic` / `m60` say PowerAssertion uncommitted** → committed and live (`PowerAssertion.swift`). → **Update memory.**
13. **Memory `hf-symlink-dangling-store-entries` "not yet built"** → shipped v0.10.142 (`Doctor.swift:119`). → **Update/retire memory.**
14. **Memory `release-distribution` "3 un-pushed commits"** → now 3 + C11 sampling delta (grew). → **Update memory; ship #10.**
15. **Plans:** `m50-plan` "Not started", `convert-quant-arch-gate-plan` "do not implement yet", `m72-vision-convert` 500-wart → all shipped/fixed. → **Update plan headers (#1).**
16. **Possible duplicate:** `Sources/athena/Commands/ConfigEditor.swift` co-exists with `Sources/AthenaDeploy/ConfigEditor.swift` (post-M70.1b). → **Confirm the former is a thin re-export, not a forked copy** (not a hitlist item; a 5-min check).

---

## 4. Don't-touch list (confirmed correctly deferred/dropped)

So a future audit doesn't re-flag these. Each notes the gate that would change the verdict.

- **Token-budget quotas (#9)** — committed by ADR 007 but *no consumer* needs them; counters already exist. **Un-defer when:** a commercial-host deployment or a consumer requires per-principal ceilings.
- **Parakeet-as-default** — Whisper default is correct and intentional; **no WER A/B harness exists**. **Un-defer when:** a labelled-set WER+throughput A/B is actually run.
- **M78.2 video descriptions** — correctly waiting on consumer requirements (frame rate / summary / contract). Do **not** build speculatively. **Un-defer when:** a downstream consumer specifies the shape. (Must be tagged `[native]`.)
- **`/api/chat`+`/api/embed` removal** — staged exactly per ADR 013 (deprecated, still served). **Un-defer when:** the deprecation window elapses + consumer migration confirmed.
- **`reasoning_effort` alias** — explicitly on-demand. **Un-defer when:** a consumer drives the field.
- **Audio-in-chat (#5)** — audio tower stripped in substrate. **Un-defer when:** upstream `mlx-swift-lm#207` lands **and** a consumer needs audio *reasoning* (distinct from `/v1/audio/*` analysis).
- **DFlash structured/temp>0/kernels/GDN** — perf/scope follow-ups, no demand. **Un-defer when:** a Guide peek-advance shape exists or a perf milestone is opened.
- **M51 / M52.A / M52.B / M52.C** — all inert today (precondition-gated / workload-gated). **Un-defer when:** the sampling precondition is relaxed (M51/C) or a long-context/aggressive-eviction workload appears (A/B).
- **M59.5 disk cache / M59.6 shred / M61 affinity-queue** — workload/erasure-gated; M61's own "Stage 0" (raise `max_entries`/`idle_ttl`) is the cheaper first move. **Un-defer when:** the named workload/legal pattern is observed.
- **At-rest key rotation (#7)** — no consumer demand. **Un-defer when:** an operator needs to rotate a store key.
- **Streaming-to-disk uploads** — bounded by concurrency cap. **Un-defer when:** concurrent large uploads cause real memory pressure.
- **Client `https` / fail-closed TLS (ADR 004)** — accepted warn-only posture; client stays http-only. **Un-defer when:** the appliance moves toward untrusted-network exposure.
- **VLM constrained-decoding / VLM-LLM double-count / cold-load slot-outside-gate** — "if it bites" follow-ups. **Un-defer when:** the symptom is actually observed.
- **GPU/Metal watchdog-wedge recovery** — forensics exhausted. **Un-defer when:** a live reproduction exists to pin it (don't chase blind).
- **swift-huggingface fork revert** — **Un-defer when:** upstream PR #50 merges + a new release tag exists. (Until then it must still be remote-pinned for the release pipeline, like #10.)
- **Port-7447 cross-repo** — user-driven, outside Athena. **Un-defer when:** the user updates an internal codename/an internal codename/an internal codename.
- **DiffusionGemma / "two embodiments" ADR** — research; cannot do structured output (bounds value). **Un-defer when:** a Mac in-process runtime is viable and the strategic question is worth an ADR.
- **bf16 unquant decode cliff / MLX issue** — substrate/upstream; 8-bit is the operational answer. **Un-defer when:** a GPU capture pins the kernel (then file #38).
- **Release pipeline R2–R7** — gated on #10 (R1) + user repos + Apple Developer account.
- **Quantized-embedder convert route / TTS / GPU-compute-extension governance** — all consumer/workload-gated per their ADRs.
- **NF3 TriAttention numeric parity CI** — intentionally MLX-gated (ADR 009: `MLXArray` can't eval under `swift test`); validate via the Release-binary/manual tier. **This is by-design, not test-debt** (unlike ND14/ND15, which *are* buildable — see #8).
- **Fan management (M60.4)** — **DROP:** value evaporated once the overnight crater was traced to sleep (fixed by PowerAssertion); Apple-Silicon fan control is undocumented/root-only/fragile. thermalState back-off (shipped on `/healthz`) is the cheaper answer.
- **K13 `--key` argv** — **DROP:** no `--key` Option exists in code; only help text references it. Close the tracker line.
- **Speaker identity (ADR 014)** — deliberate non-extension; daemon unchanged by decision. **Un-defer when:** the roster grows past ~hundreds of voices (tripwire).

---

*Generated by the backlog-discovery audit (5-way fan-out over ADRs 001–023, ~30 plan/analysis
docs, the readiness/audit ledgers, and ~90 memory topic files), with every verdict verified
against `Sources/`/`Tests/`/`deploy/`/git tags rather than status labels.*
