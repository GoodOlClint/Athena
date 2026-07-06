# ADR 039 — Per-sequence KV accounting + batch-within-the-span scheduler

- **Status:** Accepted — S0 (re-pin v0.10.264) + S1 (governor per-seq KV v0.10.265) + S2 (scheduler + routing + fan-out v0.10.266) SHIPPED; batching default-off, verified end-to-end (`deploy/e2e-batching.sh`, 6 concurrent on gemma-4-26b-a4b-it-8bit — all coherent, no cross-contamination, 6-in-4s)
- **Date:** 2026-07-05
- **Deciders:** operator + agent
- **Executes:** ADR 038 Decision 2 (continuous batching inside one gated span). **Extends:** ADR 011 (unified budget), ADR 023 (truthful accounting), ADR 029 (one gated span). **Plan:** `docs/batching-integration-plan.md`.

## Context

ADR 038 sanctioned continuous batching *inside one gated execution span* as the upgrade path once the demand trigger fires, and named two Athena-side blockers: (1) **the governor has zero per-request memory accounting** — `MemoryReservation` is keyed by `ModuleID` in flat bytes ([ModuleID.swift:51](../../Sources/AthenaCore/ModuleID.swift#L51)), so N-row `BatchKVCache` growth is invisible to admission (the load-bearing gap); (2) no scheduler multiplexes queued requests into the substrate's `BatchGenerator`.

This session de-risked the capability (throwaway `bench-batch`, removed): batching delivers ~3× aggregate at N=8 and plateaus ~18×/13.5× at N≈64 on dense/MoE; **committed KV is small (~0.1 GB/row)** and the alarming memory swing was reclaimable MLX cache already bounded by ADR-023 G1. The useful batch width is **~64** (throughput saturates; past it is wasted memory). The operator is directing the build now, ahead of the (unfired) demand trigger, as **minimal core, default-off**.

The current serialization point is `InferenceGate.shared.withExclusiveExecution` wrapping each request's *whole* decode ([MLXLLMModule.swift:726](../../Sources/AthenaLLM/MLXLLMModule.swift#L726)); the single-sequence drive is `container.generate(...)` ([:1270](../../Sources/AthenaLLM/MLXLLMModule.swift#L1270)); structured/MTP/speculative/logprobs already fork off via `runSpeculative` ([:740](../../Sources/AthenaLLM/MLXLLMModule.swift#L740)).

## Decision

1. **Extend the governor with per-sequence KV accounting (conservative worst-case).** A batched row reserves `maxTokens × perTokenKVBytes` ([perTokenKVBytes already computed](../../Sources/AthenaLLM/MLXLLMModule.swift#L334)) **up front**, admitted against the ADR-023 truthful denominator (`budget − max(committed, reserved)`). A row that would exceed the budget is **denied a batch seat** (falls back to queue/serial or a cause-naming 4xx), never admitted into an OOM. Reservations release on row completion. Decision logic is MLX-free and unit-pinned (ADR 008/009). Rationale for conservative-over-optimistic: per-row KV is small (measured ~0.1 GB) so worst-case reservation is cheap, throughput plateaus at ~64 regardless, and OOM-safety dominates squeezing batch occupancy — optimistic incremental growth reintroduces exactly the ADR-023 budget-blowout this accounting exists to prevent.

2. **One batch scheduler owns one `BatchGenerator` and holds one ADR-029 gated span.** Plain-chat requests `insert()` (after per-row admission) instead of each taking `withExclusiveExecution`; the scheduler drives `next()` across all live rows and fans each row's tokens into that request's own `AsyncStream<GenChunk>`. This composes with — does not weaken — ADR 029: one batched `next()` is one eval graph in flight, so "one gated span" holds; and ADR 011: still one model, one Metal budget, no second allocator. `rebind`/load-swap is a **batch barrier** — it drains the current batch before reassigning `container` ([:663](../../Sources/AthenaLLM/MLXLLMModule.swift#L663)), subsuming the WP6 in-gate rebind. Same-model only. Client disconnect = leave-not-cancel (`BatchGenerator.cancel(uid:)`).

3. **Batching is default-off, config-flag opt-in** (`batching_enabled`, boot-set from TOML + env, mirroring `InferenceGate.enabled`). Flag off ⇒ the existing per-request `withExclusiveExecution` path is **byte-unchanged** (pinned by the OpenAI e2e parity checks). The shipped v0.10.263 observability slice is the trigger instrument; enable when `/metrics` shows routine `gateWaiters ≥ 2`.

4. **Scope is minimal core, fixed-batch.** Only **text-only** plain-unstructured chat batches — on **any** model including a VLM's text path (verified on the served gemma-4-26b-a4b; the guard excludes **image-bearing** requests, not vision *models*, since the batch engine decodes text tokens only). Structured output, MTP, speculative, and logprobs stay single-row (existing `runSpeculative` fork — no work). The scheduler is **fixed-batch** (drain → admit → drive → next), not mid-flight join; a different-model request takes the serial path (the gate is the rebind barrier). Per-row structured output (substrate `RowSampler` grammar hook, ADR 038 gap #4), the join/leave scheduler, paged KV (#12/#13), and shared-prefix dedup (#16) are deferred. **Correctness bar: close-to-single-stream, not exact token parity** — dense batched SDPA diverges slightly from single-seq SDPA on open-ended prompts (substrate #9 precedent); all completions stay coherent.

## Rejected alternatives

- **Optimistic incremental KV reservation** (reserve prompt only, grow as rows decode) — packs more rows but risks mid-batch OOM if growth outruns headroom; reintroduces the ADR-023 blowout. Rejected: the measured per-row KV is small enough that worst-case is cheap, and the throughput plateau at ~64 means the extra occupancy buys nothing.
- **Per-request `withExclusiveExecution`, batched underneath** (keep each request acquiring the gate) — two requests can't both hold the exclusive gate, so this can't multiplex; the gate ownership must move to the scheduler.
- **Batching structured/speculative rows in the same batch** — `RowSampler` has no grammar seam and batch-or-speculate is the shipped substrate rule; forcing it now is a substrate change out of scope.
- **Auto-engage batching on contention** (no flag) — implicit daemon-wide behavior change, harder to revert; rejected for an explicit default-off knob.

## Consequences

- The governor gains its first per-request granularity — a genuine extension of the ADR-011/023 model, but additive: with batching off, admission is unchanged.
- ADR 029/011 unamended — this interprets and extends them (one gated span = one batched `next()`; one budget still enforced, now per-row).
- Substrate re-pin (5f17df48 → #263) lands as a standalone slice first (daemon-wide substrate change isolated from batching logic).
- Measured multi-client throughput (~3× at realistic N) becomes available when the operator flips `batching_enabled`; single-client latency (MTP, prompt cache) remains the correct axis for the current workload.
- Deferred items tracked against ADR 038's substrate-gap list and mlx-tracker #12/#13/#16.
