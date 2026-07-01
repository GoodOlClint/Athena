# 030 — Default-on prefill ceiling + degrade recoverable MLX allocation faults

**Status:** Accepted. **Part 1 (default-on derived prefill ceiling) implemented**
— `GovernorMemory.defaultPromptTokenCeiling` (MLX-free, unit-pinned) + enforced
at both `container.prepare` chokepoints in `MLXLLMModule` (`runSpeculative` AND
`beginGeneration` — the substrate-stream path the gemma4-MoE abort took, which
previously bypassed the guard). **Part 2 (degrade recognized MLX faults to 503)
IMPLEMENTED (WP2, v0.10.238).** The H2 `isMetalOOM` needle fix it depends on
already shipped.
**Date:** 2026-06-26

**Part 2 implementation note (WP2, 2026-07-01).** The global `MLX.setErrorHandler`
(`Load.swift`) now, for a **recognized** allocation/buffer-size fault (matched by
the new `AthenaError.isMetalOOMMessage` — the same needle set as the 503
classification), **records** it into a process-global `MetalFaultLatch` and
**returns** instead of `fatalError`ing — keeping the daemon alive. Returning from
the handler is not experimental: it is the exact mechanism mlx-swift's own
`withError`/`withErrorHandler` use to convert MLX faults into Swift throws, and a
rejected device-cap `malloc` leaves the allocator intact (nothing was
half-written). The offending tenant's gated span
(`InferenceGate.withExclusiveExecution`, which ADR 029 guarantees is
single-tenant) clears the latch on entry and, on exit, converts a set latch into
a classified `metalOutOfMemory` → **503 `metal_oom`**; the decode loops break on
the latch (`DecodeLoopControl`) first so they never touch the faulted arrays. Any
**unrecognized** fault still re-`fatalError`s. Behind a default-on revert knob
`metal_fault_degrade` (`ATHENA_METAL_FAULT_DEGRADE`) — and, because the degrade
needs the gate to attribute+consume the fault, it no-ops (falls back to
`fatalError`) if either that knob or `inference_gate_enabled` is off.

MLX-free decision logic unit-pinned (latch algebra, needle match, gate→503
conversion, latch-preferred-over-raw-throw, no-fault-no-degrade). A **gated heavy
test** (`MetalFaultDegradeE2ETests`, `ATHENA_RUN_MODEL_TESTS=1`) triggers a real
`[metal::malloc]` device-cap fault (a single ~93 GiB fp16 buffer > the 80.64 GiB
M5-Max cap) through the actual handler→latch→gate path and asserts 503 + a
correct post-fault eval; it needs a metallib-bundled Metal test host (the
`swift test` tier has none — ADR 009). **Empirical finding on the M5-Max host:**
the *prompt-driven* O(seq²) score-buffer abort the ADR originally observed is now
**additionally mitigated by 512-token chunked prefill** — a 114,918-token prompt
(`max_prompt_tokens=0`, KV cap raised past its own guard) prefilled to a clean
`200`, and a 600k-token prompt returned a clean `503 prompt_cache_cap_exceeded`,
in both cases **with the daemon surviving**. So Part 2 is confirmed as the
*safety net* for residual single-buffer faults (a wide vision tensor, a future
non-chunked path) rather than a frequently-hit path.
**Milestone:** TBD (daemon-resilience follow-up)
**Related:** ADR 011 (governor), the H2 `isMetalOOM` needle fix (shipped this
batch), the `max_prompt_tokens` guard (v0.10.219).

## Context

A full code review (2026-06-26) found a **request-reachable whole-daemon abort**
on the LLM chat path:

- `runSpeculative` tokenizes the prompt and calls
  `promptExceedsCap(count, cap: params.maxPromptTokens)`. The cap defaults to
  `nil` (`max_prompt_tokens` is an operator calibration knob, **default-off**),
  so the guard is skipped out-of-the-box.
- Prefill attention is **O(seq²)**. Past a model/hardware-specific length a
  single fp16/bf16 score buffer exceeds Metal's `maxBufferLength` (observed:
  gemma4-MoE at ~61k tokens → a 111 GiB buffer vs an 80.6 GiB device cap). The
  substrate allocator **throws** `[metal::malloc] … greater than the maximum
  allowed buffer size …` on MLX's `default-qos.cooperative` worker thread.
- That throw never reaches any Swift `catch` (it fires off the Swift task), so
  it routes to the global `MLX.setErrorHandler` installed in `Load.swift`, which
  **re-`fatalError`s** — killing the daemon for all tenants. The handler comment
  already documents this as deliberate-for-now with a named upgrade path
  ("degrade recoverable errors to a request 500 once the captured messages show
  which are safe").

Asymmetry: the **embedding** path has an always-on hard `maxInputTokens` ceiling
(NI4); the **LLM** path does not. The `isMetalOOM` classification gap (the needle
list missing `[metal::malloc]` / `maximum allowed buffer size`) was a separate
finding (H2) **already fixed** in the localized batch — its fix is a prerequisite
for part 2 below.

## Decision

Two parts, shippable in sequence.

### Part 1 — Default-on derived prefill ceiling

Derive a conservative **default** `maxPromptTokens` when the operator has not set
one, from the loaded model's context window **and** the device `maxBufferLength`
(the sequence length below which a single attention score buffer cannot exceed
the device cap). Apply it as the existing pre-prefill check, refusing an
over-ceiling prompt with the current `inputTooLong` **400** before any MLX eval —
mirroring the embedding NI4 ceiling. The operator `max_prompt_tokens` override is
**retained** (it can raise or lower the derived default; it is the calibration
knob the hardware/model specificity demands).

### Part 2 — Degrade recognized MLX allocation faults instead of re-fataling

Change the global `setErrorHandler` so that a **recognized** allocation/buffer
fault (matched via the now-expanded `AthenaError.isMetalOOM` needles, incl.
`[metal::malloc]` and `maximum allowed buffer size`) is **recorded** into a
per-thread / `@TaskLocal` "last MLX fault" slot that the request path checks
after `eval`/`asyncEval` and converts to a classified **503 `metal_oom`** (or
**400** for a request-shape cause). The handler `fatalError`s **only** on
genuinely-unrecognized faults (MLX state is undefined for those → clean launchd
restart with a logged message, as today).

## Consequences

- A single oversized request **no longer aborts the daemon** out-of-the-box
  (Part 1 catches it pre-eval; Part 2 is the safety net for anything that slips
  past the heuristic).
- The derived ceiling is **heuristic** (hardware + model specific). The operator
  knob stays — *ponytail: leave the tuning knob, don't pretend the formula is
  exact.*
- Part 2 narrows the "MLX state undefined after an error" risk to a deliberately
  **small scope** — allocation/buffer-size faults, which are pre-submission
  rejections (no partial Metal state mutated) for the dominant case. Anything
  outside the recognized set still fatals.
- The ceiling derivation and the needle classification are **MLX-free decision
  logic**, unit-pinned (ADR 008/009). The numeric prefill behavior stays in
  `AthenaLLM`, gated.

## Open questions (resolve in review)

- **Derivation formula:** context-window bound vs `maxBufferLength`-derived
  seq-length bound — which dominates, and the safety factor. Needs one
  measurement per model class to calibrate the default.
- **Staging:** ship Part 1 (default ceiling) first and keep the handler fataling
  with the better message; ship Part 2 (degrade) once Part 1's logs confirm
  which fault strings are safe to recover from. Recommended.
- **Scope of Part 2:** strictly allocation/buffer faults, or also other
  recoverable substrate faults surfaced on the worker thread? Lean strict.
