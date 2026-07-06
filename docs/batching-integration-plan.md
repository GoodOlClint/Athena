# Continuous-batching integration — change plan (ADR 038 milestone, minimal core)

**Status:** proposed, awaiting operator approval (brownfield gate).
**Decision record:** ADR 039 (this plan's architecture).
**Executes:** ADR 038 Decision 2 ("continuous batching inside one gated span"). Preserves ADR 011 (one Metal budget, never compose at inference), ADR 029 (one gated execution span), ADR 023 (truthful budget accounting).

## Intent (operator-chosen, 2026-07-05)

Build the **minimal core** now — the operator is directing the build ahead of the ADR-038 demand trigger, with capacity de-risked this session (~3× at N=8, plateau ~18×/13.5× N=64 dense/MoE; committed KV small; useful batch width ~64). Not the full milestone: per-row structured output (substrate gap — `RowSampler` has no grammar hook) and scheduler polish are **explicitly deferred**. Batching is **default-off, config-flag opt-in**. KV admission is **conservative worst-case**. The substrate re-pin lands as a **standalone slice first**.

## What batches, what doesn't

Only **plain unstructured chat** batches. The existing `runSpeculative` fork ([MLXLLMModule.swift:740](../Sources/AthenaLLM/MLXLLMModule.swift#L740)) already routes **structured output, MTP, speculative, and logprobs** requests down a separate single-sequence path — those stay single-row for free, bypassing the batch (the same rule mlx-lm ships: batch *or* speculate). The batch only intercepts the `beginGeneration` unstructured path ([:761](../Sources/AthenaLLM/MLXLLMModule.swift#L761), drive at [:1270](../Sources/AthenaLLM/MLXLLMModule.swift#L1270)).

## Slices (each: small, test-pinned commit; house pre-commit pipeline)

### S0 — Standalone substrate re-pin
Bump [Package.swift:27](../Package.swift#L27) `5f17df48…` → the #263 integration revision; regenerate `Package.resolved`. **No batching code.**
- **Prerequisite:** confirm the target revision is pushed to `github.com/GoodOlClint/mlx-swift-lm.git` (the local integration clone is `9f777fe`; the remote must carry the pinned SHA — push it if absent). Do **not** pin to a SHA that exists only locally.
- **DoD:** full `./deploy/test.sh` (unit) + `./deploy/e2e-rbac.sh` + `./deploy/build.sh Release` all green against the new substrate — proves the daemon-wide bump is regression-safe *in isolation*, so any later red is unambiguously batching, not the substrate.
- Commit + tag (own version bump).

### S1 — Governor per-sequence KV accounting (the load-bearing gap)
Add per-request KV reservation to the governor. Today `MemoryReservation` is keyed by `ModuleID` in flat bytes ([ModuleID.swift:51](../Sources/AthenaCore/ModuleID.swift#L51)) with **zero per-request granularity**.
- New MLX-free decision logic (in `GovernorMemory`/`AthenaCore`, unit-pinned per ADR 008/009): given a candidate row's `maxTokens × perTokenKVBytes` ([MLXLLMModule.swift:334](../Sources/AthenaLLM/MLXLLMModule.swift#L334) already computes `perTokenKVBytes`), decide admit/deny against the **ADR-023 truthful denominator** (`budget − max(committed, reserved)`), reserving worst-case up front.
- **No behavior change yet** — batching is off; this is the admission primitive S2 calls. A serve path with batching off is byte-unchanged.
- **DoD:** unit tests — a row whose worst-case KV fits is admitted; one that would exceed the truthful budget is **denied** (returns a cause-naming decision, not an OOM); reservations release on row completion. Pure functions, no Metal.

### S2 — Batch scheduler + route plain chat + per-row fan-out (behind `batching_enabled`, default off)
- New `batching_enabled` knob (static `nonisolated(unsafe) var`, boot-set from TOML + env, mirroring `InferenceGate.enabled`); **default false**.
- New **batch scheduler** (serializing actor owning one `BatchGenerator`): plain-chat requests `insert()` their prompt (admitted via S1), the scheduler holds **one ADR-029 gated span** and drives `next()` across all live rows, fanning each row's tokens into that request's existing `AsyncStream<GenChunk>`. Same-model only; a `rebind` drains the batch first (batch barrier — the `container` swap at [:663](../Sources/AthenaLLM/MLXLLMModule.swift#L663) can't land mid-batch). Leave-not-cancel on client disconnect (`BatchGenerator.cancel(uid:)`).
- Per-row metering: each row reports its own `TokenUsage` (the `GenChunk.usage` path is already per-stream).
- **When the flag is off:** the existing per-request `withExclusiveExecution` path ([:726](../Sources/AthenaLLM/MLXLLMModule.swift#L726)) runs unchanged — proven by the OpenAI e2e byte-parity pins.
- **DoD (discriminating, end-to-end):** `deploy/e2e-batching.sh` — with `batching_enabled=on`, fire N concurrent `/v1/chat/completions`; assert (a) all N return correct completions, (b) batched greedy output matches the serial single-request output for the same prompt (close-to-single-stream bar, substrate #9 precedent), (c) `/metrics` shows the batch ran (>1 row concurrent), (d) governor never exceeds budget. With `batching_enabled=off`, the existing e2e pins stay byte-identical. The DoD **fails before S2** (no batching path) and passes after.

## Deferred (not in this pass, tracked)
Per-row structured output (needs substrate `RowSampler` grammar hook — ADR 038 substrate gap #4), cross-batch shared-prefix dedup (#16), paged KV (#12/#13 — de-risked academic on large unified memory this session), MoE-batch efficiency tuning. Speculative/MTP batching stays out (batch-or-speculate).

## S2 implementation spec (mapped 2026-07-05, ready to code)

Overnight run shipped S0 (v0.10.264) + S1 (v0.10.265); S2 deferred to a co-verified session (hot-path concurrency, thin unattended verification). Everything below is mapped and locked so S2 is a code-and-verify job, not a design job.

**Locked design decisions:**
- **Fixed-batch, not mid-flight join** (minimal core): drain the queue → admit a batch → drive to completion → next batch. The ADR-039 "join/leave" continuous batching is the deferred polish. Simpler, far less concurrency hazard for the first cut; a request arriving mid-batch waits for the next batch (accepted latency tradeoff behind the default-off flag).
- **Text-only chat on any model (incl. a VLM's text path); image requests excluded.** Batch iff `batching_enabled && schemaJSON==nil && (tools==nil||empty) && logprobs==nil && speculative != true && no message carries images`. **VERIFIED end-to-end on the served gemma-4-26b-a4b VLM** (6 concurrent chats, all coherent + correct, no cross-contamination, 6-in-4s). A VLM container's `.model` *does* batch its text path — the earlier "non-vision only" guard was over-conservative and would have made batching never engage on any loadable chat model (Qwen3.5 checkpoints don't load; gemma-4 is the only loadable family). Image-bearing requests still take the serial VLM route (the batch engine decodes text tokens only). Everything else → the existing per-request path, unchanged.

**Seams (file:line):**
- Route branch: in `generateMetered` ([MLXLLMModule.swift:716](../Sources/AthenaLLM/MLXLLMModule.swift#L716)), *before* the `withExclusiveExecution` block — batchable requests go to the scheduler (which owns the gate) instead of each taking the gate. Non-batchable fall through unchanged.
- Prompt tokens: reuse `UserInput → container.prepare(input:) → lmInput.text.tokens.asArray(Int.self)` ([:1244-1247](../Sources/AthenaLLM/MLXLLMModule.swift#L1244)) so tokenization/chat-template is byte-identical to serial. Enforce `enforcePromptCeiling` ([:1250](../Sources/AthenaLLM/MLXLLMModule.swift#L1250)).
- Drive: `BatchGenerator(model: ctx.model, …)` inside `container.perform`; `insert(prompts:maxTokens:samplers:)`; `while hasWork { for r in next() {…} }`. Per-row `NaiveStreamingDetokenizer` (`append(token:)`/`next()`) → `GenChunk.text`; on `r.finishReason` emit `.usage` (prompt = promptLen, completion = `r.allTokens?.count`) + `.finish` (`"length"`→.length else .stop) + `continuation.finish()`.
- Per-row sampler: `makeRowSampler(temperature:topP:topK:seed:)` from the request's temp/topP/seed.
- Admission: `SequenceKVLedger` (S1, shipped) — `sequenceKVReservation(maxTokens, perTokenKVBytes)` then `admit(uid:rowKVBytes:denominator:budget:)`; denied rows stay queued for the next batch.
- Rebind barrier: `rebind`/`dropResidentModel` ([:657](../Sources/AthenaLLM/MLXLLMModule.swift#L657)/[:420](../Sources/AthenaLLM/MLXLLMModule.swift#L420)) must drain the active batch before reassigning `container`.

**The one open plumbing decision (resolve first):** `MLXLLMModule` has **no** governor reference and there is **no** `MemoryGovernor.shared`. The scheduler's admission needs the live `(denominator, budget)`. Proposed: add `public func admissionInputs() async -> (denominator: Int, budget: Int)` to `MemoryGovernor` (reuse the private `admissionDenominator()` + `totalBudgetBytes`), and set a `@Sendable () async -> (Int, Int)` provider on the scheduler at daemon boot in `Load.swift` (where both are already wired) — avoids threading a governor ref through the module. Confirm this vs. passing a governor ref into the module at construction.

**Actor worker pattern (the concurrency crux):** the drive loop must NOT hold the `MLXLLMModule` actor while decoding (other requests must enqueue). It runs in a detached `Task` and does its heavy work inside `await container.perform { … }` (a *different* actor), so `MLXLLMModule` is free to accept enqueues during decode. Worker: `while let batch = drainQueue(); !batch.isEmpty { await runOneBatch(batch) }`; set `batchWorkerRunning=false` when the queue drains. Mutable per-uid maps (detok/continuations) declared *inside* the `container.perform` closure (it's `@Sendable`); `BatchPending` must be `Sendable` (continuations + `[Int]` + `@Sendable` sampler all are).

**Flag:** `batching_enabled` static `nonisolated(unsafe) var enabled = false`, boot-set from TOML (`tomlCfg?.batchingEnabled`) + env, mirroring `InferenceGate.enabled` in `Load.swift`.

**DoD:** `deploy/e2e-batching.sh` — spin a loopback daemon on `gemma-4-12B-8bit` (dense text, clean batch path) with `batching_enabled=on`; fire N concurrent `/v1/chat/completions`; assert all N correct + coherent, batched greedy ≈ serial (close-to-single-stream), governor never over budget; then `batching_enabled=off` → existing `e2e-rbac`/OpenAI pins byte-identical. Fails before S2, passes after.

## Test bar & rollout
- Every slice lands with its regression test; MLX-free decision logic unit-pinned (ADR 008/009), MLX numerics gated behind `ATHENA_RUN_MODEL_TESTS`.
- Rollout: ship default-off. The **shipped observability slice (v0.10.263)** is the trigger instrument — enable `batching_enabled` when `/metrics` shows routine `gateWaiters ≥ 2` / rising `gateWaitP95Ms`. Revert = flip the flag.
- Correctness bar: close-to-single-stream, not exact token parity (substrate convention).
