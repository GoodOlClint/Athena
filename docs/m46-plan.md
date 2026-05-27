# M46 — consumer integration shake-out

Planning doc captured from the the consuming application end-to-end test findings against
v0.10.56. Five sub-slices plus a pre-tag hotfix. Not yet executed — saved so
the work can be picked up when ready.

## Operator decisions (locked)

| # | Decision | Choice |
|---|---|---|
| 1 | Per-request timeout field name on `ChatCompletionRequest` | `timeout` (OpenAI-flavored, not `request_timeout`) |
| 2 | Case-insensitive model matching strictness | Simple `lowercased()` compare; document the ASCII-only limit |
| 3 | `reservedBytes` deprecation window | Just drop the field — no compat shim, no consumer is reading it |
| 4 | M46.3 substrate fallback for `chat_template_kwargs` | **TBD** — resolve during M46.3 execution after checking what mlx-swift's `UserInput` API exposes. If it forwards template kwargs ⇒ direct passthrough. If it doesn't ⇒ choose between pre-rendering the prompt with the template Jinja ourselves vs. fail-closed 400. Default lean: pre-render, since fail-closed makes the field useless on every supported Qwen model |

---

## M45.7 — the consuming application unblock hotfix (working tree, ready to commit)

| File | Change |
|---|---|
| [deploy/athena.toml:213-220](../deploy/athena.toml#L213-L220) | Comment out `request_timeout_secs` — matches daemon's documented opt-in posture |
| [Sources/athena/Server/AthenaServer.swift:4320-4341](../Sources/athena/Server/AthenaServer.swift#L4320-L4341) | `Self.error()` warning-logs every 5xx response (silent-kill fix) |
| [Sources/athena/Commands/Load.swift:64-68](../Sources/athena/Commands/Load.swift#L64-L68) | `--speculative` help text reflects M40 sampling-speculative support |

Test plan: existing e2e suite (must stay 481/0). No new tests; logging side-effect not unit-testable without standing up an AthenaServer test target.

Ship as a single commit + annotated tag before kicking off M46.1.

---

## M46.1 — Operator legibility II

**Goal**: close the matching silent-failure gaps M45.7 didn't reach.

**Changes**:
- [Sources/AthenaCore/InferenceDeadline.swift:63-83](../Sources/AthenaCore/InferenceDeadline.swift#L63-L83): `deadlineBoundedNanos` logs warning when the deadline timer fires before the pump finishes (stream truncation). Needs a race-free "timer or pump won?" guard — small `NSLock`-protected bool inside the closure.
- [Sources/athena/Server/AthenaServer.swift:687-689](../Sources/athena/Server/AthenaServer.swift#L687-L689): install pre-ServiceLifecycle `signal()` trap that logs signum + any available sender context (`siginfo_t.si_pid` on POSIX) before graceful drain. Closes finding #4 — next time the daemon walks into the exit handler, the log says *why*.
- **Long-generation heartbeat** (added 2026-05-27 from the consuming application's "10-min wall time, syslog silent" observation): in [Sources/athena/Server/AthenaServer.swift:collectMetered](../Sources/athena/Server/AthenaServer.swift#L926) emit a periodic notice during a decode that exceeds N seconds (default 10 s) — log every M tokens (default 64) or every K seconds (default 5), whichever fires first. Includes request id, module, elapsed seconds, tokens emitted, current tokens/sec. Closes the third silent-failure gap: "actively decoding" is no longer visually indistinguishable from "process hung."

**Tests**:
- Extend `InferenceDeadlineTests` to assert log capture on stream truncation.
- New `SignalHandlerTests` — likely needs a unit-level fake since macOS signal delivery is awkward in test harnesses.
- New test asserting the heartbeat fires (count of notice-level log lines under a synthetic slow generator).

**Risk**: Low. The race guard in `deadlineBoundedNanos` is the only subtle bit.

---

## M46.2 — Preload all configured modules

**Goal**: close finding #2 — first request to embedding/transcription/diarization/speakerEmbedding modules no longer returns `module_loading` 503.

**Changes**:
- [Sources/athena/Server/AthenaServer.swift:657-671](../Sources/athena/Server/AthenaServer.swift#L657-L671): extend `--preload` from "warm `.llm`" to "warm every `ModuleID` whose persisted allowlist has an `is_default=1` entry". Use `governor.ensureLoaded()` per module, kicked off in parallel as detached tasks. HTTP surface still comes up immediately; warms still best-effort.
- Doc: update `--preload` help text + `deploy/athena.toml` preload comment + README.

**Tests**: integration test seeding allowlist with one LLM + one embedding default, asserting `--preload` warms both.

**Risk**: Low.

---

## M46.3 — Per-request overrides on ChatCompletionRequest

**Goal**: close findings #1 (per-call timeout) and #3 (`enable_thinking`).

**Changes**:
- [Sources/athena/Server/OpenAIDTO.swift:60-98](../Sources/athena/Server/OpenAIDTO.swift#L60-L98) — add two fields:
  - `timeout: Int?` — per-request inference deadline override in seconds. Overrides the daemon-wide `request_timeout_secs`. `nil` ⇒ inherit daemon default; `0` ⇒ disable for this call.
  - `chat_template_kwargs: [String: JSONValue]?` — passthrough dict reaching the chat-template application (path depends on substrate API check — see Decision #4 above).
- Plumb both through the sync, stream, and queue paths (`collectMetered`, `deadlineBounded`, `RequestQueue` worker).
- Update OpenAPI spec at [Sources/athena/Server/OpenAPI.swift](../Sources/athena/Server/OpenAPI.swift); drift-guard test catches misalignment.

**Tests**:
- `OpenAIDTOTests` for decoding both new fields.
- Integration test asserting `enable_thinking=false` suppresses `<think>` on plain (no `response_format`) chat against Qwen3.5.
- Integration test asserting `timeout: 600` survives past a daemon configured with `request_timeout_secs=120`.
- Integration test asserting `timeout: 1` against a slow request returns 504 even if daemon default is unbounded.

**Risk**: Medium. Substrate's `UserInput` API for chat-template kwargs needs verification. The pristine-substrate rule (memory) forbids forking — so the fallback is pre-rendering the prompt ourselves with the Jinja template + kwargs.

---

## M46.4 — Model identity hygiene

**Goal**: close finding #5 — `Qwen/...4b` and `Qwen/...4B` can't both appear in the allowlist; requests with a different case match the configured entry.

**Changes**:
- Case-insensitive matching at request time:
  - [Sources/AthenaLLM/MLXLLMModule.swift:305](../Sources/AthenaLLM/MLXLLMModule.swift#L305) — replace `allowed.contains(target)` with case-folded compare.
  - [Sources/AthenaEmbedding/MLXEmbeddingModule.swift:92,161](../Sources/AthenaEmbedding/MLXEmbeddingModule.swift#L92) — same.
  - Preserve canonical case in storage so list responses don't lie.
- SQLite migration on `model_allowlist`:
  - Detect case-divergent `(module, lower(id))` pairs.
  - Dedupe to the first-declared row (preserve `is_default` / `declared` ordering).
  - Log a notice per collapsed pair at migration time.
  - Schema version bump in [Sources/AthenaStore/AthenaStore.swift](../Sources/AthenaStore/AthenaStore.swift) so downgrades are detected.
- Error-response builder at [Sources/AthenaCore/AthenaError.swift:100-112](../Sources/AthenaCore/AthenaError.swift#L100-L112) — dedupe `available` list before joining (defense-in-depth; should be redundant post-migration).

**Tests**:
- Migration test against a seeded fixture with case-divergent rows.
- Resolution test for `4b` vs `4B` request paths.
- Downgrade-guard test asserting schema-version bump is honored.

**Risk**: Medium — data migration on existing installs. Guard with the schema-version bump and add a one-shot rollback note in the operator log.

---

## M46.5 — /healthz schema polish + module-lifecycle observability

**Goal**: close findings #6 (reservedBytes naming), #8 (unload reasons), and the late-arriving "slow vs. hung" legibility ask from the consuming application (2026-05-27).

**Changes**:
- Rename `reservedBytes` → `residentBytes` in `/healthz` response. Drop the old field outright (Decision #3 — no compat shim, no consumer reads it).
- Add `unloaded_reason: idle_evict | memory_pressure | operator_unload | crash_restart` field to module state in [Sources/AthenaCore/MemoryGovernor.swift](../Sources/AthenaCore/MemoryGovernor.swift).
- Surface `unloaded_reason` in `/healthz` module snapshot + emit an audit-log row on every state transition.
- **"Slow vs. hung" signals** — add to `/healthz`:
  - `inflightRequests` — count of currently-active generations across all module classes (LLM, embedding, transcription, diarization, speakerEmbedding). Driven off an actor-tracked counter in `AthenaServer` that increments on enter and decrements on exit/error of every metered handler.
  - `generationTokensPerSec` — rolling-window throughput across recent decode chunks. Sourced from the same `GenChunk.usage` events `collectMetered` already drains; aggregate over a configurable window (default 30 s) per module class. Reports `null` when no recent activity.
- Update OpenAPI spec + drift-guard test.

**Tests**:
- Governor state-machine tests asserting the right `unloaded_reason` flows through each unload path (idle evict, memory pressure, operator unload, crash restart).
- `/healthz` schema test asserting `residentBytes` present, `reservedBytes` absent, `inflightRequests` + `generationTokensPerSec` shape.
- Inflight counter test: spawn N concurrent slow requests, assert `inflightRequests == N` mid-flight, drops to 0 after.

**Risk**: Low — observability layer, no decision-loop coupling.

---

## Sequencing rationale

```
M45.7  ← ship first (timeout + silent-kill unblock; already coded)
  │
M46.6  ← embedder per-call buffer leak fix (RE-SEQUENCED 2026-05-27).
  │      Promoted ahead of M46.1 because it's the actual QUALITY unblock
  │      for the consuming application — the 30 GiB embedder leak is what causes
  │      unified-memory thrash and 10-min wall times. Small, mechanical
  │      fix once the eval/release boundary is identified.
  │
M46.1  ← legibility II + long-generation heartbeat (small, low-risk;
  │      pairs with the M45.7 silent-kill fix conceptually; the heartbeat
  │      complements M46.6 because once the leak is fixed but generations
  │      are still slow under other load, consumers still need to know
  │      "alive but slow" vs "dead")
  │
M46.2  ← preload-all (small, isolated; helps every cold-start consumer)
  │
M46.3  ← per-request overrides (the consuming application's biggest "nice to have")
  │
M46.4  ← model name hygiene (data migration — sequence after the bigger
  │      refactors land so any model-table touches don't conflict)
  │
M46.5  ← /healthz polish + slow-vs-hung signals (last among feature
         slices; touches OpenAPI which other slices also touch, so
         sequencing it late means one OpenAPI write per slice not five)
```

Smallest-risk-first within the feature slices, with the heaviest data-migration slice (M46.4) buffered by simpler work on either side. **M46.6 jumps the queue because the consuming application's quality unblock depends on it.** If the consuming application prioritizes M46.3 (per-request overrides) over M46.2 (preload-all), swap them — both are independent.

---

## Definition of done for M46

- [ ] M45.7 + all six M46 sub-slices: commit + annotated semantic tag pushed straight to origin/main (per release workflow).
- [ ] e2e suite green on every tag (currently 481/0 from M45.6 — stay 0-failure).
- [ ] OpenAPI spec drift-guard green on every tag.
- [ ] the consuming application's `make eval-corpus` run completes through `synthetic-test-corpus-mini` with no 503 retries, no thinking-channel leakage on plain chat, no deadline kills.
- [ ] `MEMORY.md` updated with one entry per shipped slice (`project_m46-1-legibility-ii.md`, `project_m46-2-preload-all.md`, …, `project_m46-6-embedder-memory.md`).

---

## M46.6 — Embedder per-call buffer leak fix

**Status updated 2026-05-27 from the consuming application follow-up data.** Root cause is now identified, not investigative. Promoted to **highest-priority M46 slice after M45.7** because it's likely the *actual* unblock for the consuming application's slow extraction — the 30 GiB embedder leak is what pushes total memory to ~88/96 GiB and triggers unified-memory thrash, which is what's giving 10-min wall times on 361-byte inputs.

**Confirmed root cause**: candidate #1 from the original investigation list. The embedder line went **157 MiB (initial probe of mmap'd weights — M5.5 expected) → 37.9 GiB (after warm calls)**. That delta isn't initial-load measurement noise; it's per-call attention/projection scratch that's allocated during `embed()` and never released. A 4B model's working memory should be in the hundreds of MiB at most for a normal input, not 30 GiB.

The other two candidates from the original list are ruled out by the data:
- **#2 monotonic accounting** — would explain growth but not the *magnitude* (30 GiB working memory for a 4B model is implausible regardless of how it's accounted; the leak is real, not measurement).
- **#3 mmap double-counting** — would predict LLM and embedder sums exceed total process RSS; 50.3 + 37.9 = 88.2 GiB vs ~80 GiB process RSS in the consuming application's snapshot is close enough that it's not the dominant effect.

**Fix shape**: explicit `MLX.eval()` + autorelease drain at the end of [Sources/AthenaEmbedding/MLXEmbeddingModule.swift](../Sources/AthenaEmbedding/MLXEmbeddingModule.swift)'s `embed()` path. Mirror whatever the LLM module does on the analogous decode-exit boundary — the LLM line item's 50.3 GiB matches its bf16 weight footprint, so it's not leaking, which means the substrate's eval/release primitives work; the embedder just isn't calling them.

**Diagnostic step before coding** (cheap):
- Read [Sources/AthenaEmbedding/MLXEmbeddingModule.swift](../Sources/AthenaEmbedding/MLXEmbeddingModule.swift) and compare its eval/release boundaries to [Sources/AthenaLLM/MLXLLMModule.swift](../Sources/AthenaLLM/MLXLLMModule.swift). The delta tells you exactly where to insert the drain.
- If the LLM module does the same thing the embedder does and *doesn't* leak, the eval/release primitive is wired elsewhere (substrate-internal) and the fix is different — but the prior strongly favors a missing-eval bug in the embedder path.

**Tests**:
- Embedding regression test: load embedder, embed long-sequence input, assert `reservedBytes` returns to a steady-state baseline within ±10% after the call completes (not before — eval/release is async).
- Repeat-call test: 10 consecutive embed calls, assert `reservedBytes` doesn't grow monotonically.

**Risk**: Low. The fix is mechanical (insert an eval boundary). The only subtlety is making sure the boundary is placed *after* the result tensor's bytes have been materialized into the response, not before — otherwise the result itself gets reclaimed prematurely.

**Sequencing**: **First M46 sub-slice to ship.** Promoted ahead of M46.1 because it's the actual quality unblock for the consuming application. Estimated 1-2 hours of work + test if the diagnostic confirms the hypothesis.

---

## Out of scope for M46

Items surfaced in the audit that aren't worth folding in:
- "Daemon walked into exit handler" (#4 in original report) — partly closed by M46.1's signal logging, but the root cause might still be external (launchctl bootout from somewhere). Re-evaluate after the signal log lands and we see one in the wild.
- Eager-load via `?wait_for_module=N` query param on `/v1/chat/completions` and `/v1/embeddings` — superseded by M46.2's preload-all + the documented 503-retry contract.
- LiteLLM client-side timeout investigation — that's a the consuming application / consumer-side concern, not Athena's.
- **Model-name quantization hint** (surfaced 2026-05-27): the consuming application ran "Qwen3.5-27b-mlx" expecting a 4-bit build and got bf16. Athena's model store doesn't encode quantization tier in names; `athena show` already reports the per-arch info via M23 but doesn't surface quantization explicitly. Worth a small UX pass to surface quantization in `athena show` / `/api/models` later — but it's a one-time discovery wart, not a recurring quality issue, so it shouldn't gate M46.

---

## Status

Saved 2026-05-27. Not started. Plan locked pending:
- Operator decision on whether to swap M46.2/M46.3 ordering.
- Resolution of Decision #4 during M46.3 execution.
- M46.6 diagnostic reads to land before M46.1 starts — outcome determines whether M46.6 is a code fix, governor refactor, or doc-only.

**Late-arriving findings (post-initial plan):**
- 2026-05-27 — the consuming application agent reported embedder `reservedBytes` = 37.9 GiB after warm calls (≈4-5× the 4B model's bf16 weight footprint). Folded in as M46.6.
- 2026-05-27 — the consuming application agent asked for `/healthz` to distinguish "hung" from "slow" via in-flight request count and active tokens/sec. Folded into M46.5.
- 2026-05-27 (follow-up data) — the consuming application confirmed the 157 MiB → 37.9 GiB transition is per-call (not initial-load measurement noise). Promotes M46.6 candidate #1 from "investigative" to "confirmed root cause." Re-sequenced M46.6 ahead of M46.1 since it's the actual quality unblock for the consuming application's slow extraction.
- 2026-05-27 (same follow-up) — Athena's syslog was silent during a 10-min generation. Added long-generation heartbeat to M46.1 (periodic notice every K seconds or M tokens during a long decode) so consumers can tell "actively decoding" from "process hung."
- 2026-05-27 (same follow-up) — `Qwen3.5-27b-mlx` model-store name doesn't encode quantization tier; the consuming application ran it expecting 4-bit and got bf16 (50.3 GiB resident matches bf16 weight footprint). Logged as a follow-up UX wart, not folded into M46.
