# Athena code audit + remediation blueprint — 2026-07-01

**Baseline:** v0.10.236, commit `a603e0a`, branch `main`. All line numbers reference this commit.
**Method:** six parallel specialized review passes (correctness, security, concurrency, AI-slop, code-smell/over-engineering, half-implemented-features), each verified against source — no grep-only findings. Unit suite at baseline: **783 pass / 39 skipped (heavy, model-gated)**.
**Audience:** this document is a work blueprint for follow-up agents. Each work package (WP) is self-contained: files, exact change, definition of done. Execute WPs in phase order; WPs within a phase are independent unless noted.

---

## Executive summary

The codebase is in unusually good shape for its velocity (~85 milestones in 6 weeks): no P0s anywhere, **no security P0/P1** (auth, SQL, SSRF, FFI, crypto all verified clean), route⇄spec parity clean, no orphaned config keys, layering (ADR 008/009 MLX-free seams) holding. The real findings cluster in four places:

1. **The InferenceGate (ADR 029) has three remaining holes**, all on the *operator/governor* side rather than the request path: governor eviction/relief calls `MLX.Memory.clearCache()` and prompt-cache `flushIdle` **off-gate concurrently with live decodes** (the strongest candidate yet for the 2026-06-05 GPU wedge), `POST /api/models/load` rebinds off-gate, and in-daemon `convert` runs quantize kernels off-gate.
2. **ADR 030 Part 2 remains the one surviving whole-daemon-abort path**: an async-eval `[metal::malloc]` OOM still `fatalError`s the daemon, dropping every in-flight request.
3. **The ADR 036 "one pump" seam is half-built**: two SSE pumps plus a third non-streaming copy re-implement stop/tool/finish semantics, and dialect drift has already shipped.
4. **A batch of genuine but low-risk debt**: one G4 structured-output contract breach, a world-readable credential DB, ~50 LOC of provably dead ADR-031 leftovers, 4× copy-pasted upload preambles, and a doc-rot batch.

**Stale-knowledge correction** (repeated in several trackers): "ADR 029 wiring pending" is wrong — the gate is wired into all five module-class forwards, vision/video indirectly, and the request-path rebind (`auditedRebind`, `AthenaServer.swift:3761`). The M1 cancellation backstop is present in all four decode loops. Only the three holes in WP1/WP3 remain.

| Phase | Theme | WPs | Severity ceiling |
|---|---|---|---|
| 1 | Stability (Metal-pool exclusivity + daemon-abort) | WP1–WP3 | P1 |
| 2 | Contract & security hygiene | WP4–WP6 | P2 |
| 3 | Consolidation (dedupe, dead code, god-file split) | WP7–WP10 | debt |
| 4 | Docs, ledger, memory | WP11–WP12 | cosmetic |

House rules for whoever executes this: each slice = commit + annotated semantic tag straight to `origin/main` with `Athena.appVersion` bumped in the slice commit; decision logic MLX-free + unit-pinned (ADR 008/009); error strings name structural causes, never model repos (ADR 021); errors use the `{"error":{message,type,code}}` envelope; routes change only together with `OpenAPISpec.swift`.

---

## Phase 1 — Stability (do first)

### WP1 · Gate governor-initiated Metal frees under the InferenceGate  — **P1, highest value**

**Finding (concurrency review, confidence high):** the InferenceGate serializes tenant *forwards*, but the `MemoryGovernor` frees Metal memory outside it. Three ungated entry points run MLX ops concurrently with a gated decode:

- `evictSync` (`Sources/AthenaCore/MemoryGovernor.swift:834`) → detached `Task { await module.unload(); onUnloaded() }` where `onUnloaded = { MLX.Memory.clearCache() }` (`Sources/athena/Commands/Load.swift:622`).
- `makeRoom` rung 1 → `reclaimCache?()` = `clearCache()` (`Load.swift:639`) and `promptCacheRelief?()` = `flushIdle(); clearCache()` (`Load.swift:588`).
- `relievePromptCachePressureIfNeeded` (post-generation, `AthenaServer.swift:1615`) → same relief hook. `flushIdle` → `demoteToDisk` (`Sources/AthenaLLM/PrefixKVCache.swift:463–500, :684`) does `cache.copy()`, `eval(...)`, `KVByteCodec.encode` — MLX ops; the cache's NSLock does not exclude a forward holding the (different) gate.

**Interleaving:** request A holds the gate mid-decode; request B's admission (different module class) trips the high-water mark → `evictSync`/`reclaimCache` free the shared buffer pool under A's in-flight kernels. The codebase itself documents this class as forward-corrupting (`MLXDiarizationModule.swift:41`). Strong candidate for the 2026-06-05 GPU wedge.

**Fix:** run every governor-initiated MLX-touching body inside `InferenceGate.shared.withExclusiveExecution`. Keep admission *math* ungated (correct per ADR 029). Concretely: the teardown task in `evictSync`/`unload` becomes `Task { await InferenceGate.shared.withExclusiveExecution { await module.unload(); hook?() }; await self.markUnloaded(id) }`; wrap the `reclaimCache`/`promptCacheRelief` hook closures in `Load.swift:588/622/639` the same way (gating at the hook site keeps `MemoryGovernor` MLX-free).

**DoD:** unit test pinning the decision (a stub gate recording acquisition order proves eviction cannot interleave a fake forward); `./deploy/test.sh` green; `deploy/e2e-rbac.sh` green; a soak note that cross-tenant load near the memory ceiling no longer submits concurrent MLX work (log-verifiable via gate acquisition logging). Amend ADR 029 in the same commit (its wiring list gains "governor eviction/relief").

### WP2 · ADR 030 Part 2 — stop aborting the daemon on recognizable Metal OOM — **P1**

**Finding (correctness review, confidence high; deferral documented):** `MLX.setErrorHandler` (`Sources/athena/Commands/Load.swift:462–467`) unconditionally re-`fatalError`s on any async-eval fault. `AthenaError.isMetalOOM` (`Sources/AthenaCore/AthenaError.swift:329–341`) already classifies the thrown-path; async faults on MLX's worker thread only reach this handler. One oversized request that slips past the ADR-030-P1 *prefill* ceiling (e.g. long *generation* growing the O(seq²) score buffer, or a wide vision tensor) aborts the whole daemon → launchd restart drops every in-flight request. This is the last whole-daemon-abort path.

**Fix:** in the handler closure, branch on the recognized OOM needles (`[metal::malloc]`, "maximum allowed buffer size" — reuse `isMetalOOM`'s needle set): fail the in-flight decode with the classified 503 (`metal_oom` / `.metalOutOfMemory`) and keep the process alive; re-fatal only for unrecognized faults (the code comment at `Load.swift:460–461` already names this exact upgrade path). The plumbing challenge is signalling the in-flight generation from the handler thread — the `DecodeLoopControl` cancellation counter is the existing seam.

**DoD:** unit-pin the needle→degrade decision (MLX-free); a gated heavy test or manual repro (oversized decode) showing 503 + daemon stays up; ADR 030 status updated to Part 2 shipped.

### WP3 · Close the two remaining ungated operator paths — **P2**

**Findings (concurrency + half-implemented reviews, confidence high):**
- `POST /api/models/load` (`AthenaServer.swift:3881`, `handleModelsLoad`) calls `sel.rebind(to:)` **without** the gate — contrast the request path's gated `auditedRebind` at `:3761`. Rebind = drop + load new weights on the pool, racing any in-flight forward (double-residency + co-execution; the H3 hazard reachable via the control plane).
- In-daemon `POST /api/models/convert` (`AthenaServer.swift:3121` → `Sources/AthenaLLM/ModelConvert.swift`) runs full MLX weight-load + quantize kernels with no `InferenceGate` reference anywhere in the file. Unlisted omission in ADR 029 (neither wired nor carved out).

**Fix:** route `handleModelsLoad` through `auditedRebind` (or wrap in `withExclusiveExecution`); wrap `ModelConvert`'s Metal-executing span in the gate (a convert can hold the gate for minutes — acceptable for an operator op, but say so in the ADR amendment; alternatively an explicit ADR carve-out if blocking inference on convert is deemed worse — decide, don't leave it implicit).

**DoD:** both paths gated (or ADR-carved); ADR 029 amended in the same commit; e2e model-ops scripts green.

---

## Phase 2 — Contract & security hygiene

### WP4 · G4 structured-output contract breach — **P2, silent wrong output**

**Finding (correctness review, confidence high):** with `tools` + default `tool_choice:"auto"` + a broken `response_format:{type:"json_schema",...}`, `structuredRequestError()` (`Sources/athena/Server/OpenAIDTO.swift:409–422`) short-circuits to nil because the "tools take precedence" check at `:413` fires on `advertiseMenu` (true for auto+tools) — but `effectiveSchema()` (`:376–400`) never engages a tool schema for un-forced auto, falls to the response_format branch, gets nil → **unconstrained generation with a 200** instead of a 400. A caller who asked for schema-constrained output silently gets free text.

**Fix:** short-circuit only when a tool is actually *forced* (`!forcedTools().isEmpty`), not merely advertised. Add the unit test (this combination is currently uncovered).

**DoD:** new test: auto+tools+broken-schema → 400 `invalid_request_error`; existing tool-choice tests green; `deploy/e2e-tool-choice-auto.sh` green.

### WP5 · Security hygiene batch — **P2 + P3**

1. **World-readable credential DB (P2, confidence high):** `athena.sqlite` + `-wal`/`-shm` + `.migrate-bak` are created with no `posixPermissions` (verified; `Sources/AthenaStore/AthenaStore.swift:130–171`, migration at `:304–307`), and `athena install` sets the data dir to 0755 (`Sources/athena/Commands/Install.swift:284`). The DB holds token SHA-256 hashes, PBKDF2 password hashes, usernames/roles, audit, usage — offline-readable by any co-resident non-root user, squarely inside the ADR-024 threat model. **Fix:** chmod 0600 on DB + sidecars after open and after the migration swap; data dir 0700 (model-store dir stays 0755 — weights aren't secret). Mirror the existing pattern in `KVSnapshotStore.swift:81–82, 165–168`.
2. **Raw store-error strings to client (P3):** `AthenaServer.swift:4531–4533` (token create; similar ~`:4247` user-create, ~`:4658`) return `"\(error)"` raw, bypassing the `Self.classified` suppression boundary (`:5558–5572`). Route through the classifier; detail to os_log only.
3. **`/healthz` + `/openapi.json` unauthenticated on non-loopback (P3, posture decision):** `Auth.swift:304–307`; healthz exposes model ids, memory footprint, GPU clock, thermal. Either gate healthz *detail* behind `.metricsRead` (bare liveness 200 stays open), or record it as an accepted posture ADR-004-style. **Decide + document; don't leave implicit.**

**DoD:** perms verified by a unit/e2e assertion (stat the created DB in `e2e-rbac.sh`); error-hygiene paths through classifier; posture decision recorded (ADR or ADR-004 amendment).

### WP6 · ADR 029 residual: wrong-model window on concurrent different-model chat — **P3, correctness not crash**

**Finding (concurrency review; ADR 029 itself defers this):** LLM rebind and decode are two separate gate acquisitions (`AthenaServer.swift:803–809/3733–3736` vs `:1053/1077/1106`). A rebinds→X, releases; B rebinds→Y; A decodes against Y (wrong-model output, possibly mislabeled id). **Fix:** fold rebind + container-capture + decode into one gated span, mirroring the embedding module's `embedInFlight` chain (`MLXEmbeddingModule.swift:242–258`). Schedule after WP1/WP3 (same code region).

---

## Phase 3 — Consolidation

Line numbers in this phase reference the pre-split `AthenaServer.swift`; execute WP7–WP9 **before** WP10 (the split) so they stay valid.

### WP7 · Finish the ADR 036 pump seam — **the drift machine**

**Finding (code-smell P1 + AI-slop, corroborated by correctness review):** `pumpTokens` (`AthenaServer.swift:4965–5197`) and `pumpAnthropic` (`:5236–5382`) each independently re-implement StopStreamFilter latching, ReasoningChannelFilter, tool-buffer + `parseToolCall` fallback, finish-reason resolution, and usage recording; the non-streaming Anthropic path (`:1720–1745`) is a **third** copy of the stop-truncate + tool-precedence algebra that `chatChoice`/`toolCallChoice` (`:1179–1228`) implement for OpenAI. ADR 036 promised these live *once*. Drift already shipped: pumpAnthropic drops reasoning (`:5312`) and fakes stop attribution (`stopHit = stops.first`, `:5298`); Anthropic ignores the per-request timeout the OpenAI path honors (`:1690/:1707` vs `:1020–1022/:1102–1104`).

**Fix:** one generic `foldGenChunks(events, stops, isToolCall, sink: ProtocolEncoder)` where the encoder is a small struct of closures (`emitText/emitReasoning/emitToolCall/finish`); plus one shared `resolveToolCallOutcome(detected:text:isToolCall:) → enum` that all four encode sites switch on (the tool-call precedence is currently copied at `:1207–1217`, `:1736–1745`, `:5157–5165`, `:5330–5340`). ~150 LOC deleted; the drift class closes structurally. While there: thread the per-request timeout uniformly, and either surface Anthropic reasoning as thinking-blocks or keep the documented drop — but as an encoder decision, not a pump fork.

**DoD:** `deploy/e2e-tool-choice-auto.sh` + the Anthropic e2e (5/5) green; unit tests for `resolveToolCallOutcome`; a diff showing both pumps reduced to encoder structs.

### WP8 · Dead-code deletion batch — all provably dead, verified zero callers

| Item | Location | LOC |
|---|---|---|
| `governedPreflight(messages:...)` — ADR 031 orphan | `AthenaServer.swift:3369–3390` | ~20 |
| `streamNDJSON(tokens:...)` — /api/chat transport orphan | `AthenaServer.swift:3394–3425` | ~30 |
| `SepKEK` throws-only stub (+ its test) — keep the `KEKType.sep` header raw value | `Sources/AthenaCore/KVSnapshotEnvelope.swift:105–112`, `KVSnapshotEnvelopeTests.swift:59` | ~25 |
| `canonicalCaseInsensitive(_:)` | `Sources/AthenaCore/ModelAllowlist.swift:52` | ~10 |
| Test-only String `generate` overload family (3 in protocol + conformers; stub carries a 4th not even in the protocol) — first migrate tests onto `generateMetered` with a small `.text`-collecting helper | `Sources/AthenaLLM/StubLLMModule.swift:15–35,130–165`, `MLXLLMModule.swift:691–703` | ~80 |
| 9× `guard case .ok … fatalError()` unreachable-branch unwrap → one helper on `Outcome` (or replace `Outcome` with `Result`, see WP9) | `AthenaServer.swift:3465, 3616, 3674, 3843, 3916, 4019, 4230, 4476, 4618` | net ~-30 |

**Also decide here — `/api/embed`:** deprecated, ADR 013 verified zero callers, removal pre-cleared but "optional" per ADR 031. Either (a) remove it ADR-031-style (route `:442` + `handleNativeEmbed:3461–3494` + DTO `NativeAPIDTO.swift:37` + spec `OpenAPISpec.swift:308–312` + drift-guard, one commit), or (b) keep it and fix the real divergence the AI-slop review found: `/v1/embeddings` (`:1776–1807`) does `auditedRebind` inline while `governedEmbed` (`:3432–3457`) omits it, so `/api/embed` rebinds **unaudited**, and the doc comment at `:3427` falsely claims the helper is shared by both routes. Recommendation: (a) remove — it executes existing ADR intent and deletes the divergence instead of repairing it. Fold `auditedRebind` into `governedEmbed` and route `/v1/embeddings` through it either way.

**DoD:** `./deploy/test.sh` + `deploy/e2e-rbac.sh` green; grep proves zero references to each deleted symbol.

### WP9 · Dedupe batch

1. **Multipart upload preamble ×4:** `handleTranscriptions` (`:1841–1882`), `handleVideoTranscriptions` (~`:2088–2120`), `handleDiarizations` (`:2230–2266`), `handleSpeakerEmbeddings` (~`:2560–2592`) repeat the same ~40-line boundary-check → cap → collect-with-413 → parse → `first("file")` block (already drifting: video vs audio cap). Extract `extractUploadFile(request:cap:) async → Outcome<(MultipartForm, MultipartForm.Part)>`.
2. **Transcription response encoder ×2:** `handleTranscriptions` (`:1929–2020`) vs `handleVideoTranscriptions` (`:2180–2224`) duplicate `plain()`, `words()`, and the response_format switch verbatim (video = strict subset, no diarize). Extract `encodeTranscription(result, format, wantWords, diarize:)`. ~70 LOC.
3. Minor (same commit): `uiQuery` hand-parses the query string (`:3964–3975`) — Hummingbird ships `request.uri.queryParameters` (keep `safeModelName` downstream); `iso(_:)` (`:3496`) allocates an `ISO8601DateFormatter` per call on request paths → `static let`; `Outcome<T>` (`:3344–3347`) is `Result` re-invented — fold into WP8's helper decision.

**DoD:** audio/video/diarization/speaker e2e green; upload-cap 413 behavior byte-identical (unit-pin the extracted helper).

### WP10 · Split `AthenaServer.swift` — mechanical, last

5,725 lines, 40% of the module: 76 route registrations in a 567-line `run()` (`:141–708`), both dialect handlers, all media handlers, RBAC admin, ~25 `ui*` cookie wrappers, SSE machinery, formatters. Split along the seams the code already names: `AthenaServer+ChatOpenAI.swift`, `+Anthropic.swift`, `+Audio.swift`, `+Admin.swift`, `+UI.swift`, `+SSE.swift`; move registration blocks with their handlers; ~30 `private`→`internal` tweaks. While in there, optionally table-drive the `ui*` wrapper family (`(perm, mutating, handler)` registered in a loop — `:3977–4030, 4755–4815`). Zero behavior change; e2e suites pin it. Consider the same treatment later for `Load.run()` (`Commands/Load.swift:318–958` — a 640-line dual-personality function that is both a client rebind command and the daemon composition root; extract the rebind branch, then named phases).

---

## Phase 4 — Docs, ledger, memory

### WP11 · Doc-rot + spec-honesty batch (one commit)

**Spec/doc over-claims (the self-describing contract, so not merely cosmetic):**
- `OpenAPISpec.swift:114` + `AnthropicDTO.swift:11–13`: claims `cache_control`/`thinking` are refused with 400 — actually Codable **silently drops** them (neither key appears anywhere in Sources); and "x-api-key alias next slice" is stale (shipped status: not needed, bearer works). Fix the descriptions (or actually refuse the keys — decide; silent-ignore is OpenAI-adapter-conventional, but say so).
- `docs/kv-cache-disk-snapshots.md:105`: claims passphrase KEK exists; only `keyfile:PATH` is parsed (`Load.swift:542–571`; `KEKType.passphrase` marked "not yet provided", `KVSnapshotEnvelope.swift:23`).
- KV-snapshot restore observability (correctness P2-3): `KVSnapshotStore.swift:91–106` coerces *decrypt/auth* failures to silent go-cold via `try?` — add a `notice` log distinguishing "no file" from "file present but unrestorable" so snapshot-dir corruption is visible.

**Stale comments (verified against code):** `AthenaServer.swift:1395, 3365, 3379, 3725` (/api/chat + queue references); `AthenaDeploy/AthenaConfig.swift:26` + `DefaultConfig.swift:104` ("vectors + queue + jobs" — DefaultConfig ships in the operator template); `AthenaCore/AthenaError.swift:116` ("allowlisted"); `AthenaServerKit/Auth.swift:469` + `AppRequestContext.swift:19` (`queuePrincipal` never exists); `clients/Sources/AthenaClient/RemoteModels.swift:173, 215, 540` ("allowlist"); `AthenaModels/AthenaQwen35MTP.swift:13` ("NOT yet wired" — long wired; ADR 033 deletes the file anyway); `governedEmbed` doc `:3427` (covered by WP8); `SpeculativeStats.swift:17–29` ("future metrics surfaces" promise — strike or wire).
**File rename:** `Sources/AthenaCore/ModelAllowlist.swift` → `ModelIdentity.swift` + rewrite the 30-line retired-allowlist header (live helpers: `modelStoreIdentity`, `canonicalByStoreIdentity`, `dedupedCaseInsensitive`).
**Repo hygiene:** commit ADR 033 + `docs/qwen35-mtp-substrate-cutover-plan.md` (currently untracked — the decision exists only in the working tree); archive/delete `docs/tool-call-streaming-handoff.md` (describes the bug fixed in v0.10.230/ADR 034/036 — verified fixed); **keep** `docs/substrate-jinja-bugs-handoff.md` (upstream swift-jinja bugs still latent; Athena workaround shipped v0.10.236).

### WP12 · Ledger (no code now; calendar items)

- **Revert knobs** (repo is 6 weeks old, none removable yet — revisit ~Sep 2026 if unexercised): `cold_load_wait_secs=0` legacy-503 (ADR 015, Jun 17 — oldest), `governor_admission_mode="estimate"` (ADR 023 G2, Jun 21), `inference_gate_enabled=false` / `ATHENA_INFERENCE_GATE=0` (ADR 029, ~Jun 28).
- **TriAttention retire-tripwire:** default-off, Qwen3.5-only (`Load.swift:729–738` warns it's inert otherwise), ~700 vendored lines, zero e2e, fleet serves Gemma4. ADR 028 deliberately retained it — add an explicit tripwire in ADR 028's style ("if no Qwen3.5 target is served by <date>, park upstream") rather than deleting now. ADR 033, if executed, changes this calculus (it drops TriAttention per the operator decision recorded in the cutover plan).
- **`prompt_cache_*` knob surface** (13 keys, all wired, feature off-by-default): don't delete; add one interaction table to `docs/kv-cache-disk-snapshots.md` and consider deriving `persist_max_*` defaults from the RAM-tier values.
- **Anthropic dialect gaps, on record as intentional:** reasoning dropped on both paths (consistent; "first cut"), per-request timeout not honored (no wire field). Revisit if a consumer asks for thinking blocks.

---

## Appendix — verified clean (do not re-review without cause)

- **Security:** RBAC route coverage complete; bearer tokens SHA-256 + constant-time + fail-closed expiry; login limiter peer-IP-keyed, no XFF trust, timing-oracle-safe dummy verify; session cookie HMAC-SHA256/HttpOnly/SameSite=Strict + CSRF header on all UI mutations; SQL fully parameterized; SSRF blocked (`data:`-only images, model ids confined to store dirs); upload caps bounded with stateless 100-continue; rust-shim 15/15 exports `catch_unwind`-guarded with vocab/id/byte caps; idle-KV + snapshot crypto sound (fresh keys, random nonces, AAD binding, 0600/0700); TLS fails closed; deploy scripts loopback-only, ADR-005 argv rule test-asserted.
- **Concurrency:** `AthenaStore` actor (FULLMUTEX, no bypass); `RateLimiter`/`ConcurrencyLimiter` release-exactly-once incl. streamed bodies; `InferenceGate` FIFO/cancellation-correct, no leak on cancel; `PrefixKVCache` in-request paths (clone-on-hit, refcount vs LRU, T3 ciphertext-at-rest invariant holds); SSE cancellation bridges client disconnect → `cancelGeneration()` with no orphaned tasks.
- **Correctness:** streaming tool-call content leak (the 2026-06-30 handoff) **fixed** — both pump paths emit `tool_calls` deltas + `finish_reason:"tool_calls"`, no double-emit; swift-jinja workaround applied on both dialects' lowering paths (`OpenAIDTO.swift:367`, `AnthropicDTO.swift:156`); governor/prefill/cache-limit math clamped and unit-pinned; `StopStreamFilter`/`ReasoningChannelFilter` holdback logic correct.
- **Structure:** OpenAPISpec ⇄ registered-route parity clean; all ~55 config keys read; RBAC/store/config/spec layers free of removed-feature (queue/vectors/allowlist/DFlash/TurboQuant) remnants; no MLX imports in AthenaServerKit/AthenaCore/AthenaDeploy; module protocols carry exactly the deliberate MLX+Stub conformer pairs; hand-rolled TOML parser, `MultipartForm`, WebUI-HTML-in-Swift, one-file OpenAPI spec all deliberate and fine.
