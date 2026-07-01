# 029 — Execution-exclusive inference slot (one Metal-executing tenant at a time)

**Status:** Accepted — **IMPLEMENTED** (primitive + M1 in v0.10.223; call-site
wiring follows). Wiring: every module's leaf Metal execution now runs inside
`InferenceGate.shared.withExclusiveExecution` — LLM decode (the `generateMetered`
stream Task, held across the streamed decode; `AsyncStream` unbounded buffering
keeps it decode-bound so a slow SSE consumer never holds the gate), embeddings
(`embedSerialized` via the `embedInFlight` chain), transcription (`transcribePCM`
via an actor self-hop worker), diarization (`runOffline`/`runStreaming`/`segment`),
speaker-embedding (`embed`/`windowEmbeddings` via self-hop workers) — plus the
warm rebind (`auditedRebind` → `sel.rebind`), so a swap can't load a second
model's weights while a decode holds the slot (closes the H3 double-residency +
co-execution). Cold-load `awaitLoad`/`performLoad` stays UNgated (rule 1). Revert
knob wired from `inference_gate_enabled` (TOML) / `ATHENA_INFERENCE_GATE` (env),
default ON.

**WP1 (v0.10.237) — governor-initiated Metal frees now gated too.** The 2026-07-01
audit found the gate serialized tenant *forwards* but the `MemoryGovernor` still
freed Metal memory OUTSIDE it: `evictSync`/`unload` teardown (`module.unload()` +
the `clearCache` unload hook), the `makeRoom` rung-1 reclaim
(`promptCacheRelief` + `reclaimCache`), and `relievePressure` /
`relievePromptCachePressureIfNeeded` all ran `clearCache`/`flushIdle` concurrently
with a gated decode — request B's admission near the high-water mark could tear
down the shared buffer pool under request A's in-flight kernels (the strongest
candidate yet for the 2026-06-05 GPU wedge). Fix: the two eviction teardown Tasks
wrap their `module.unload()` + hook span in `withExclusiveExecution`; the
`ReclaimCacheHook`/`PromptCacheReliefHook` are now `async` and run their MLX frees
under the gate at the serve seam (`Load.swift`), with the governor **awaiting**
them so the admission re-gate still sees the reclaim's effect — it just waits for
any in-flight forward first. The admission *math* stays ungated (correct per
rule 1). MLX-free + unit-pinned (`testReclaimRunsUnderInferenceGateWP1`).

**Residual (follow-up):** the narrow concurrent-DIFFERENT-model wrong-MODEL
*selection* window (A requests X, B requests Y at the same instant; A may decode
Y because the server-side rebind and the gated decode are two separate gate
acquisitions). This is a pre-existing logical mis-selection — NOT worsened by the
gate, and with NO crash/co-execution/double-residency (those are closed). Closing
it fully needs the rebind folded into the gated decode Task (or a pinned-container
hand-off through `runSpeculative`); deferred as a small slice.

Earlier (v0.10.223): the
`InferenceGate` actor (`Sources/AthenaCore/InferenceGate.swift`) — a FIFO,
cancellation-aware async semaphore with `withExclusiveExecution`, default-on
`enabled` revert knob, MLX-free + unit-pinned (4 tests: mutual exclusion, revert
knob, error-propagation/release, cancelled-waiter drain); and **M1** — a
`Task.isCancelled` backstop in `DecodeLoopControl.isCancelled()` (the one
chokepoint all four decode loops share) so cancellation no longer depends solely
on the TaskLocal-counter inheritance.

**Pending wiring (next step), with the scoping decisions resolved during
implementation:**
1. Each module's Metal execution (LLM decode, embeddings forward, transcription,
   diarization, speaker-embedding) wraps its work in
   `InferenceGate.shared.withExclusiveExecution { … }` — gives H5 cross-tenant
   exclusivity.
2. **Cold-load exclusion:** the gate is acquired around *execution* only, NOT the
   governor's `awaitLoad` cold-load wait (ADR 015) — waiting on a background load
   is I/O, not Metal execution, and must not block other tenants for up to
   `cold_load_wait_secs`.
3. **H3 atomicity:** gating the LLM warm-rebind and the decode as *separate*
   sections still leaves a wrong-model window (B rebinds between A's rebind and
   A's container capture). So the LLM rebind must be **folded into the gated
   execution** — atomic rebind+capture+decode — mirroring the embedding module's
   existing `embedInFlight` chain. This is the refactor of the server
   preflight→generate flow that the wiring step carries.
4. Wire the `InferenceGate.enabled` revert knob from config at boot (alongside
   `governor_admission_mode`).
**Date:** 2026-06-26
**Milestone:** TBD (governor hardening follow-up to ADR 011/023)
**Supersedes/amends:** clarifies ADR 011's "single slot" wording.

## Context

ADR 011 positions the **unified Metal memory governor** as Athena's reason to
exist: audio/embeddings/vision/video are *tenants* sharing one Metal budget, and
the rule is **never compose at the inference layer** — "two uncoordinated
allocators on one Metal pool defeats the governor."

A full code review (2026-06-26) found that the "single slot" ADR 011 implies is
enforced **only as one memory reservation per module class**, not as
execution exclusivity. Concretely:

- Each MLX module (`MLXLLMModule`, `MLXTranscriptionModule`,
  `MLXEmbeddingModule`, `MLXDiarizationModule`, `MLXSpeakerEmbeddingModule`) is a
  distinct Swift actor holding a distinct substrate `ModelContainer`. The
  substrate's `SerialAccessContainer` serializes access **within one container**
  only — it does **not** serialize one module against another.
- The server handlers do `await governor.awaitLoad(<id>)` (memory admission) and
  then run inference. `MemoryGovernor` is purely a memory admission/load/evict
  actor; **nothing acquires an execution-exclusive token** before driving MLX
  kernels.

Two consequences, both confirmed against the code:

1. **Cross-tenant co-execution (H5).** A concurrent `/v1/chat/completions` +
   `/v1/audio/transcriptions` both pass admission (both reservations fit the
   budget) and submit MLX eval graphs to the one Metal device simultaneously —
   the exact hazard ADR 011 forbids, and a plausible cause of the unexplained
   "GPU/Metal wedge during sustained decode" in the 2026-06-05 watchdog note.

2. **Warm-rebind double-residency (H3).** Request A binds the LLM `container`
   locally then suspends at an `await` (prepare/structured-vocab); Request B
   (different `model=`) drives a warm `rebind` → `dropResidentModel` →
   `loadModel` during A's suspension. A resumes and runs `container.perform` on
   the *old* container it still holds via ARC, while B decodes the new one — two
   weight sets resident, two eval graphs running, and the governor's single
   fixed `.llm` reservation makes the transient double-residency invisible to
   admission (→ possible Metal OOM under pressure; `servedLLMModel()` can also
   echo the other request's model). The embedding module already serializes this
   via `embedInFlight`; the LLM module does not.

## Decision

Introduce **one process-global inference execution gate** — a single
serializing primitive (an `AsyncSemaphore(value: 1)` or a dedicated
serializing actor) — that **every Metal-executing operation acquires for the
duration of its execution**, independent of and composed with the
`MemoryGovernor`:

- LLM decode (`MLXLLMModule.runSpeculative`/`generateMetered` →
  `container.perform`), embeddings forward, transcription (`transcribePCM`),
  diarization, speaker-embedding.
- **Model rebind / resident-model swap** (`rebind` / `dropResidentModel` /
  `loadModel`). Because a rebind acquires the *same* gate, it cannot run while a
  decode holds it — **this closes H3** without a module-local serializer.

Properties:

- The gate is acquired at **execution time**, not at admission — so cold-load
  blocking (ADR 015), the streaming `: loading` keepalives, and the memory
  governor's admission math are unaffected.
- Hold for the execution span **only**, never across the whole HTTP request.
- **FIFO** ordering; released via `defer` on every path including throw and
  task cancellation (so a cancelled/disconnected request cannot leak the gate).
- The governor stays the memory admission/accounting authority (ADR 023). The
  gate adds *execution* exclusivity; the two are orthogonal.

This also resolves **M1** (the streaming-cancellation backstop) as a rider: add
a `Task.isCancelled` check in the decode loops alongside the existing
`DecodeProgress.counter` poll, plus a regression test, so correctness no longer
depends solely on the TaskLocal-inheritance placement.

## Consequences

- **Cross-tenant requests serialize at the Metal level.** Added latency when two
  tenants are busy at once. Acceptable: one Metal pool cannot truly parallelize
  two large eval graphs anyway, and ADR 011 already mandates non-composition at
  inference. Throughput within a single tenant is unchanged.
- **H3 fixed structurally** (no module-local `generateInFlight` needed) and the
  governor's accounting stops being undercut by transient double-residency.
- The gate primitive's decision logic is MLX-free and **unit-pinned** (ADR
  008/009): FIFO ordering, single-holder invariant, release-on-throw/cancel.
- **Honesty boundary:** the gate guarantees only one Athena tenant submits work
  at a time; it does not make MLX's internal worker threads cooperative. That is
  sufficient — exclusivity at submission means one eval graph in flight.

## Open questions (resolve in review)

- **Primitive choice:** `AsyncSemaphore(value: 1)` vs a serializing actor. Lean
  semaphore (simpler, FIFO, cancellation-aware) unless an actor composes better
  with the existing module actors.
- **Granularity:** one global gate (only one Metal device exists) — confirmed
  sufficient; no per-device split.
- **Reentrancy:** a single request acquires once for its decode; rebind+decode
  within one logical operation must not self-deadlock (rebind happens before the
  decode acquires, or they share one acquisition).
- **Revert knob:** ship behind a config flag (default-on) mirroring
  `governor_admission_mode`, so the gate can be disabled if it regresses a
  workload.
