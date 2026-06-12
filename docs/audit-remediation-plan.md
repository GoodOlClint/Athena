# Audit Remediation Program — M65–M71

Remediates the combined findings ledger: `AUDIT_FINDINGS.md` (2026-06-08 baseline, IDs `A1`…`L11`) and `AUDIT_FINDINGS_V2.md` (2026-06-09 re-audit, IDs `NA1`…`NL4`). Both docs stay frozen as the findings record; **this file is the live tracker** — check items off here, with the closing version noted.

Scope counts at plan time: 1 Critical + ~49 High + ~55 Medium confirmed across both audits, plus a Low tail (~100). Already closed before this plan: C8 (v0.10.107). Refuted, no action: B9 (behavioral half — help-text drift folded into M71.2), B14, E8.

## Principles

- **Order = blast radius**: remote attack surface → data loss/lockout → silent inference corruption → concurrency → perf/operability → CI blindness → polish tail.
- Each slice is independently shippable in the usual style: commit + annotated tag to main, `Athena.appVersion` bump in the slice commit, e2e gate green.
- Every fix that *can* carry a unit test does — but the structural "make Server/Commands testable at all" work is its own milestone (M70), so earlier slices test what's testable today and M70 backfills.
- When a slice touches a module, it also **re-verifies that module's `not-refound` baseline items** (listed per milestone below) and closes or re-confirms them in this tracker — they're cheap to adjudicate with the code already open.
- Items marked **DECISION** change a contract or policy; they need explicit sign-off before implementation and a CLAUDE.md / ADR edit in the same window.

---

## M65 — Remote attack surface (security)

The only Critical lives here (G1), plus everything an unprivileged or low-privileged remote caller can trigger.

### M65.1 — FFI hardening (rust-shim) ✅ v0.10.117
- [x] G1 (Critical) per-entry `catch_unwind` via `ffi_guard`; crate stays `panic="unwind"` (abort path is parked DECISION #4, not taken) (v0.10.117)
- [x] G2 reject oversized token ids before `max_id+1` dense alloc (`MAX_VOCAB` cap) (v0.10.117)
- [x] G3 cap schema bytes/depth/repetition before compile (`validate_schema`: 1 MiB / depth 64 / 100k reps) (v0.10.117)
- [x] G6 cap caller-supplied `n`/`l` in `build_words` (`MAX_VOCAB`/`MAX_TOKEN_BYTES`) (v0.10.117)
- [x] G10 `id < size` guard; mask short-write fails loud; `set_err` interior-NUL + reentrancy safe (v0.10.117)

> Note: the 7 `StructuredGuideTests` regex cases + 1 `StructuredShimTests` ordering artifact are pre-existing reds on the dead regex path (engine is schema-only since M53) — that's **NG1**, scheduled for **M70.2**. Untouched by M65.1; new hostile-schema Swift test passes.

### M65.2 — WebUI & login surface — 5/6 ✅ v0.10.118 (A3 deferred)
- [x] NA1 (High) escape `t.label`/`t.username`/`t.scope` (+ users/roles fields) in /ui users page — shared JS `esc()` + `data-u` delete button (v0.10.118)
- [x] A11 HTML-escape config values via shared `esc()` (replaces `"`-only escape; also path + error sinks) (v0.10.118)
- [ ] A3 rate-limit `POST /ui/login` by IP + backoff — **DEFERRED**: no client-IP plumbing today; correct IP source (peer via `RemoteAddressRequestContext` vs trusting `X-Forwarded-For`) overlaps DECISION #1's TLS/proxy trust model — needs sign-off
- [x] A10 generic login failure + always-run dummy PBKDF2 (`Passwords.dummyVerify`) — closes enumeration timing oracle (v0.10.118)
- [x] A12 `Secure` cookie attribute when the daemon serves TLS (`tlsEnabled`) (v0.10.118)
- [x] NA7 percent-decode fallback keeps `+`→space normalization (v0.10.118)

> Server/ login/session/XSS logic isn't unit-testable until M70 (NA2); validated via build + e2e-rbac 494/0 (login/cookie path) + binary inspection.

### M65.3 — Untrusted input caps (DoS) ✅ v0.10.119
- [x] ND1 (High) finite-check + Double-domain clamp before `Int()` in speaker embeddings → 400 audio_segment_invalid (v0.10.119)
- [x] D4 `AudioDecode` decode-sample ceiling (~4h) + clamped/guarded header reserve (v0.10.119)
- [x] A9 multipart `firstRange(of:)` split + `maxParts=256` cap (v0.10.119)
- [x] NI4 per-input embedding token ceiling (32768) → 400 input_too_long (new AthenaError) (v0.10.119)
- [x] G4 + NC2 fail closed: `structuredRequestError()` 400 on missing/unserializable schema (sync+queued); MLXLLMModule throws structured_output_unavailable on unresolvable vocab / uncompilable schema (v0.10.119)

### M65.4 — Path confinement (file ops on caller-influenced paths)
- [ ] A1 (High) confine store export under an export dir; reject symlinks/overwrite
- [ ] C5 validate id before `removeItem` in ModelPull (`..` wipes store parent)
- [ ] C6 validate `outName` in ModelConvert
- [ ] C7 confine `copy` src under store root
- [ ] C20/C24 symlink-resolution asserts in size/cp/prune; aux-copy allowlist
- [ ] D6 confine `load(directory:)` under store root
- [ ] NC12 validate allowlist id at model-load path resolution (defense-in-depth)

### M65.5 — AuthZ gaps & info leaks
- [ ] A5 (High) single caller/permission resolution in AuthMiddleware (4 drifted copies today) — *structural, do first in this slice; it underpins the rest*
- [ ] NA6 ownerless queue jobs: `owner==nil` → admin-only under enforced auth
- [ ] H6 optional owner filter at store layer for jobs/usage (defense-in-depth)
- [ ] A21 + NE7 stop reflecting raw `\(error)` / substrate detail into responses; log full, return generic
- [ ] A22 fail closed on world-readable auth-keys file

**DECISION items parked at M65 (need sign-off, then ADR):**
- A2/K1: require TLS (or explicit `--insecure`) for non-loopback auth-on daemons; add https to the portable client. Contract change for remote deployments.
- H5: per-owner `owner` column on vectors (schema migration + cross-principal semantics).
- B2/K7/K13: stop accepting secrets via argv (breaking CLI change; `--password-stdin` pattern).

Re-verify while here (not-refound): A2, A9*, A10*, A12*, A13, A14, A16–A22, A26, A27, G-module (none), D12, D13. (*already being fixed above — confirm the rest.)

---

## M66 — Data integrity & credential lifecycle

Local data loss, lockouts, and the config parser silently lying.

### M66.1 — Store integrity
- [ ] NH1 (High) recoverable `migrateToEncrypted` swap: plaintext → `.bak`, move, delete `.bak` only on success; startup adopts orphaned `.enc-migrate`
- [ ] H2 transactional allowlist default clear-then-set (`BEGIN IMMEDIATE`)
- [ ] H11 `deleteUser` cascade in one transaction, errors surfaced
- [ ] NH2 don't clear vector cache when persisted delete failed
- [ ] H13 `trimJobs` excess computed against terminal rows
- [ ] H14 failed audit write: log + health signal
- [ ] H15 zero at-rest key buffer after keying

### M66.2 — Token/user lifecycle
- [ ] NA4 server-side rotate: put new token BEFORE deleting old
- [ ] B4 CLI rotate: same ordering fix
- [ ] B5 `user add` on existing user requires `--force`/confirm
- [ ] NB11 `guardLastAdmin` fails closed on query error
- [ ] H12 `listTokens` returns truncated hash for display
- [ ] B17 document/default TTL for install-seeded admin token

### M66.3 — Install & client store safety
- [ ] B3 install seed: `O_NOFOLLOW`/`fchown` before writes (symlink TOCTOU)
- [ ] B7 `fchmod`/`fchown` on fds throughout Install
- [ ] B6 validate `data_dir`-style paths under allowed root before root chown
- [ ] NB1 invalid `--label` must not bypass the root-daemon hard-fail
- [ ] NK1 client `store import`: back up existing DB; abort on liveness-probe timeout
- [ ] B8 privileged ops read `ATHENA_STORE_KEY` from service Keychain, not sudo env

### M66.4 — Config parsing truth (AthenaDeploy + ConfigEditor)
- [ ] J1 (High) truthy-set bool parsing (`1`/`True`/`yes`), ParseError otherwise — `encrypt_store`/`preload` silently off today
- [ ] NJ1 (High) strip `\r` from CRLF config lines
- [ ] J2 inline-`#` only outside quotes
- [ ] NB2 escape strings in `setScalarThrowing` (TOML injection via /ui/api/config)
- [ ] NB8 validate enum-ish keys (`engine`, `kv_compression`) at `config set` time
- [ ] B15 appended bare key lands before first `[section]`
- [ ] NJ2 + NB9 non-default `--prefix`: plist/daemon/CLI agree on config path; stop hardcoding /usr/local

Re-verify while here: B6*, B7*, B8*, B10–B13, B15*, B18–B23, H6*, H14*, J5, K3, K6.

---

## M67 — Inference correctness (silent wrong output)

### M67.1 — TriAttention / models
- [ ] NF1 (High) RoPE offset must keep absolute position after eviction (not `keepIndices.count`)
- [ ] NF2 request-scoped eviction policy passed as parameter, not shared model state (subsumes baseline C2)
- [ ] F1 Metal kernel `else { y = 0 }` on masked path
- [ ] F4 detect & split fused `gate_up_proj` on MTP-MoE
- [ ] NF9 validate `fromState`/`applyMetaState` restored geometry
- [ ] NF7 reject/round non-multiple-of-8 `linear_key_head_dim` instead of silent truncate

### M67.2 — Embeddings
- [ ] NI1 (High) pass attention mask to pooling (subsumes baseline I1's id==pad case — mask from real row lengths)
- [ ] I4 reject empty input (NaN vector today)
- [ ] NI3 governor reload honors the rebound model, not the default
- [ ] NI2 stub/MLX model-id resolution parity
- [ ] I5/I6 pooling flags from model config; produced-dim validation

### M67.3 — Sampling & structured determinism
- [ ] C1 (High) stable top_k/top_p sort (index tie-break) — seed reproducibility
- [ ] C11 per-request sampler RNG instead of global `MLXRandom.seed`
- [ ] C12 nil-EOS fabricated id no longer drops a real token
- [ ] C13 StopStreamFilter matches on unicode scalars across chunk splits
- [ ] G7 `JSONValue.integer` preserves >2^53 schema constants
- [ ] G9 distinguish out-of-range vs disallowed in FFI `advance`

### M67.4 — Whisper / transcription
- [ ] D5 + ND3 full 99-language table (forced language AND auto-detect reporting)
- [ ] ND2 derive special-token ids from tokenizer/config, not hardcoded large-v3
- [ ] D8 empty PCM early-return
- [ ] D9 seed `segStart` from `lastTs ?? 0`
- [ ] D12 clamp inverted segments
- [ ] ND5 eviction id-match parity with load/rebind (needless multi-GB reload)
- [ ] D10 validate clustering threshold; expose minClusters

### M67.5 — LLM module odds
- [ ] NC1 (High) remove/lock `currentModelDirectory` global (queued convert vs cold-load race)
- [ ] NC3 preflight OOM guard tokenizes the same input as generation
- [ ] NC11 `residentDirectory` no longer traps on empty `modelDirectories`
- [ ] NC13 stub governor reservation released on allowlist empty
- [ ] NE5 LLM accepts full HF org/name ids like the aux modules (canonicalization parity)

Re-verify while here: C16–C19, C21, C23, F6, I7, I8, NE-uncertains.

---

## M68 — Concurrency & lifecycle

### M68.1 — MemoryGovernor
- [ ] NE1 (High) `.unloading` slot blocks concurrent reload (drain before load)
- [ ] E1 re-read state after suspension in `ensureLoaded`
- [ ] E2 task-keyed `inFlight` cleanup
- [ ] E6 no double-subtract on failure path
- [ ] NE2 admission failure recorded in `lastLoadError` (no perpetual 503)
- [ ] E5 validate budget > 0
- [ ] E12 learned-footprint reconcile under concurrent teardown

### M68.2 — LLM & structured guide ownership
- [ ] C3 (High) memoize vocab/structured-index builds as per-model `Task` (actor-reentrancy)
- [ ] C10 single `dropResidentModel()` for the 5 duplicated invalidation sites (prereq for C3 being safe)
- [ ] G5 enforce single-owner StructuredGuide invariant (assert/document; never `@unchecked Sendable`)
- [ ] G8 FFI error path returned per-call, not via cross-thread `thread_local`

### M68.3 — Transcription & embedding modules
- [ ] D1 (High) Sortformer `generate` runs on the actor (no detached escape)
- [ ] D2 rebind/unload drains in-flight generate
- [ ] ND8 streaming diarization observes request cancellation
- [ ] I2 (High) capture model handle pre-await; serialize embed per slot
- [ ] I3 gate `clearCache` on large working sets
- [ ] H4 vector upsert cap-check/cache-mutation across await; memoized load Task
- [ ] H10 authoritative dim; reject zero-length vectors

### M68.4 — Cancellation bridge
- [ ] A8 + E3 streamed disconnect and deadline expiry both reach `cancelGeneration()` (M60.5 added the decode-loop early-break; this wires the remaining paths and verifies end-to-end)
- [ ] NE6 deadline-truncated callback not fired on downstream cancel
- [ ] E13 `@TaskLocal` decode counter visible to the loop's task
- [ ] A23 queued→canceled as single conditional UPDATE

Re-verify while here: E15, E17, D14, A26, K3 (client SSE teardown — fix lands M71 unless trivial here).

---

## M69 — Performance & operability

### M69.1 — Queue/store access patterns
- [ ] A4 (High) `nextPending()` minimal-column query (drain stops loading all blobs)
- [ ] H9 `INDEX jobs(status, created)`
- [ ] A15 `depth()` = `SELECT COUNT(*)`
- [ ] NA5 /ui state uses a no-blob recent-N summary query
- [ ] A24 explicit `ORDER BY created ASC`
- [ ] A16 `/v1/models/:id` returns mtime without full-store walk

### M69.2 — Admission-control & metrics truth on streams
- [ ] NA3 concurrency slot + inflight gauge + latency timer live until the streamed body terminates (M29.2 currently doesn't bound streams; /healthz lies during decode)
- [ ] NA8 reuse resolved principal for metering (drop second auth resolution)
- [ ] A25 nearest-rank percentile

### M69.3 — Vector store
- [ ] H3 (High) cache resident normalized matrix (no per-query rebuild)
- [ ] H7 tie-break top-k on id
- [ ] H8 min-heap top-k; binary-insert; id→index map

### M69.4 — Diarization & whisper hot paths
- [ ] D3 (High) O(n²) clustering (NN-chain/PQ UPGMA)
- [ ] D7 vDSP gram-matrix fill; union-find relabel
- [ ] ND4 batch GPU→host syncs in predsToSegments/trimSilence
- [ ] ND10 O(k²) subwordSplit re-decode
- [ ] ND11 skip logprob syncs when avg_logprob unused
- [ ] ND9 move heavy sync decode off the cooperative pool (DECISION-lite: executor strategy)

### M69.5 — Decode-path costs (benchmark-gated; skip any that don't pay)
- [ ] C4 (High) `guidedArgmax` reusable buffer / O(allowed) mask / on-device
- [ ] F2 on-device eviction scoring (no per-layer host sort)
- [ ] F3 hoist `clearCache` out of per-layer eviction
- [ ] F5 chunked KV pre-alloc; C22 prefix-hit tie-break; C14 partial-select sampling; NF8 prefix gather skip

---

## M70 — Test debt (CI blindness)

### M70.1 — Make the executable target testable
- [ ] NA2 (High) + NB4: extract Server/ + Commands/ pure logic into a testable library target (or `@testable import` the executable target); then table-driven tests for `constantTimeEqual`, `resolve()` expiry, `AuthPolicy.required`, `RateLimiter.take`, `ConcurrencyLimiter`, `Session.validate`/CSRF, `MultipartForm`, `AthenaMetrics.prometheus`
- [ ] L-cross-cutting: stub-model CI tier so the MLX/inference/vector surface isn't only validated by env-gated manual xcodebuild

### M70.2 — Re-arm the env-gated load-bearing invariants
- [ ] L1 bit-identical-greedy speculative parity on a CI stub model
- [ ] L2 acceptance-rate floor observer test
- [ ] L3 VectorStore ranking on CI fixtures; L4 dim-mismatch `[]` test
- [ ] NG1 StructuredGuide tests rebuilt off the dead regex path onto real schema compilation
- [ ] NC4 PrefixKVCache bit-identity/scope-isolation/eviction tests
- [ ] NC5 cancellation early-break tests for all four decode loops
- [ ] NC6 StructuredVocab + GuidedDecoder unit tests

### M70.3 — Remaining coverage gaps
- [ ] L5–L11 (schema-mask seam, stop-filter cases, multi-seed sampling, embed-vector asserts, governor coalesce count, basename-collision, tautological asserts)
- [ ] NL1 governor load-transition/coalescing; NL4 Counter race in test harness
- [ ] NF3 TriAttention compress + gatedDeltaOps parity CI coverage
- [ ] ND14/ND15 whisper language round-trip; RMS gate; clustering branches
- [ ] NI6 bucketing pack/reassembly; NJ4 config-key parse tests; NE8 governor hooks + proxy redaction

---

## M71 — Conformance & polish tail (batch, low individual risk)

- [ ] API-surface drift: A17/A18 OpenAPI tags + cached_tokens/usage reconcile; A20 `top_logprobs:0`; NA9 queued-vs-sync param parity; NA10 transcription response_format enum; NA11 `encoding_format` honored-or-rejected; A14 403-vs-redirect
- [ ] Client robustness: K2 (High) SSE follows check HTTP status; K4/K9 optional DTO fields; K5/K11 terminal-status exit codes; K10 Run.swift status/error handling; K12; NK2 portable rotate/--ttl parity; K3 SSE teardown; K6 idempotency key on queue submit
- [ ] CLI consistency: B9 help-text default fix; B12 user arg style; B13 `--json` + footers to stderr; B16 status exit code; B18 `\r` trim; B19 prune error detail; B21/B22 exit-code & verb collisions; NB3 `default` vs installed daemon; NB10 `--revision` honored; NB12 pid 0/negative guard; NB13 status honors TLS
- [ ] Dedup/dead code: NA12, NB7, NC9, NC10, ND12, ND13, NK6, NK7, NG4, NH3, NF5, C15–C18, E11, E16, B24
- [ ] Larger refactors held to last (each its own slice if taken): A6 server god-object split; C9 `runSpeculative` extraction; E10 protocol unification; A19/A27 AuthPolicy single source

---

## Standing decisions queue (blockers for their items, not for the program)

All five decisions are now **ruled and recorded as ADRs** (003–007); see `docs/decisions/`.
Implementation lands in the milestones below.

| # | Decision | Affects | Ruling (ADR) |
|---|---|---|---|
| 1 | Non-loopback auth-on TLS posture + login-IP source | A2, K1, K8, A12, A3 | **Warn-only** — loud startup warning + doctor finding, no `--insecure` gate, client https deferred; A3 keys on TCP **peer IP** (no XFF trust), now unblocked. [ADR 004] |
| 2 | Secrets-in-argv | B2, K7, K13 | **Hard-remove `--password`** — prompt / `--password-stdin` / env only (breaking). [ADR 005] |
| 3 | Vector-store `owner` column | H5 | **Add owner-scoping** — nullable `owner`, owner-filtered query/get/delete, admin sees all, legacy NULL = admin-only (additive migration). [ADR 006] |
| 4 | rust-shim panic strategy | G1 | **Per-entry `catch_unwind`**, crate stays `panic="unwind"` — DONE M65.1 v0.10.117. [ADR 003] |
| 5 | `/api` metering + token-budget quotas | NA8, backlog #8/#9 | **Bring #8/#9 into the program** as a new milestone; NA8 single-resolution folds in. [ADR 007] |

**Resulting plan deltas:**
- M65.5 — A2/K1 downgraded to a warn-only startup-warning + doctor check (not fail-closed); **A3 moves here** (peer-IP login limiter via a new `RemoteAddressRequestContext`).
- M66 — `--password` removal (ADR 005) + vector `owner`-scoping migration (ADR 006) become first-class slices.
- New milestone **M69.6 (or M72)** — native `/api` metering (#8) + token-budget quotas (#9) per ADR 007; NA8's cheap half still lands in M69.2.

## Tracking

- Closing a checkbox: note the version, e.g. `[x] NF1 … (v0.10.118)`.
- If a slice's re-verification shows a not-refound baseline item is already fixed or invalid, mark it here with a one-line reason rather than fixing.
