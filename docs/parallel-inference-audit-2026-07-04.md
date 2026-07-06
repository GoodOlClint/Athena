# Multi-inference / parallel-workload audit — 2026-07-04

**Question:** Athena totally serializes inference. Is that still right, and what is the current state of parallel inference on unified memory?
**Method:** two verified passes — (1) a file:line map of every serialization layer in Athena at HEAD with its recorded rationale; (2) field research on parallel inference on UMA (measured numbers, local MLX clones as ground truth, peer systems). Raw reports preserved; sources at bottom. Companion decision record: `docs/decisions/038-serialize-execution-batch-within-span.md`.

## Verdict

**Serialization is the right call for Athena today — but the operator's stated reason ("Metal doesn't have as many compute cores") is the wrong physics, and formalizing the right reason changes what the future path is.**

Batch-1 LLM decode on Apple Silicon is **memory-bandwidth-bound, not compute-bound**: at ~1–4 FLOP/byte arithmetic intensity against an M4 Max ridge of ~60, the GPU sits ~95% compute-idle waiting on DRAM during decode. "Fewer cores" explains why *prefill* is slow relative to NVIDIA and where the batching *ceiling* sits — it is not why serialization is safe. Batched decode measurably yields **2.6–4.3× aggregate throughput at 16 concurrent sequences on M4 Max** (vllm-mlx paper; oMLX) and ~5–10× on offline batch workloads (mlx-lm's own numbers), and the entire peer field shipped it in 2025–26 (mlx_lm.server continuous batching, llama.cpp/Ollama parallel slots, LM Studio 0.4, vllm-metal).

What actually makes serialization right for Athena now:

1. **The substrate can't batch.** mlx-swift-lm has the KV-cache batch *primitives* (`batchSize`/`leftPadding`/`filter(batchIndices:)`, upstream KVCache.swift:1240ff) but **no batched decode loop** — `TokenIterator` is single-sequence ([mlx-swift-lm #42](https://github.com/ml-explore/mlx-swift-lm/issues/42) open; Python's `BatchGenerator` shipped Sept 2025 and its server got continuous batching Dec 2025, so the reference implementation exists).
2. **The only parallelism available today is concurrent *unbatched* decodes — which is strictly bad.** Two independent decodes on separate streams amortize zero weight reads (they split the same 546 GB/s), land on MLX's still-hardening multi-thread eval path (mlx #2133), break governor admission math, and re-open the exact allocator/clearCache hazards behind the 2026-06-05 wedge. ADR 029's gate is the correct posture against that, full stop.
3. **Cross-modality is not batchable at all** (different models — LLM + Whisper + diarization share one Metal pool), so the cross-tenant serialization is right under any future.
4. **Speculative/MTP and batching are in tension**: mlx-lm's server hard-disables batching when a draft model is set — speculation spends the same idle-compute headroom batching wants. Athena's MTP path (ADR 032) is a batch-1 optimization; at one client it's the right one.

**The formalized position (proposed as ADR 038):** serialization stays; the sanctioned future upgrade is **continuous batching *inside* one gated execution span** — which ADR 029's text already permits (its guarantee is "one eval graph in flight"; a batched step is one graph) — gated on substrate batch support, and only if the real workload shows ≥2 routinely-overlapping chat requests. Concurrent *execution* (two spans on one Metal pool) stays forbidden.

## Athena's serialization stack (what exists, why)

| Layer | Mechanism | Why (recorded rationale) |
|---|---|---|
| 0 · HTTP | `ConcurrencyLimiter` global + per-principal caps — **opt-in, default off, bypassed on loopback** (`RateLimit.swift:139–176`) | DoS bound, not inference policy. By default unlimited requests queue behind the gate. |
| 1 · Process | **InferenceGate** (ADR 029): FIFO async semaphore; one Metal-executing span at a time across ALL tenants + rebind + convert + governor frees | The 2026-06-05 wedge; MLX's process-global allocator/cache/error state; ADR 011 "never compose at inference." Honesty boundary: "one eval graph in flight." |
| 2 · Module | Actors + in-flight chains (`embedInFlight`, diarization D1); one `ModelContainer` slot; substrate `SerialAccessContainer` mutex | Wrong-model races; clearCache races between same-module calls. LLM module structurally cannot hold two decodes. |
| 3 · Request | `n>1`/`logit_bias` → 400 (M31.3); no batch API | Deterministic greedy/structured path. |
| 4 · Loops | Vendored decode loops are batch-1 by construction (`[1, seq]` prefill, scalar `.item()` picks, per-token Guide commits); substrate `TokenIterator` likewise | No batch support existed to build against. |

**Two facts worth knowing from the map:**
- **`/v1/embeddings` already does multi-sequence-per-span** — inputs are length-bucketed, stacked into one `[N, maxLen]` tensor, one forward per bucket (`MLXEmbeddingModule.swift:365–403`). The "many sequences, one gated eval" pattern is proven in-house where there's no KV.
- **Cold-load is deliberately ungated** (ADR 029 rule 1: "the governor's load wait, not Metal execution") — but `loadModelContainer` does Metal allocations, so a cold weight load genuinely overlaps an in-flight decode today. Accepted trade (not blocking tenants for `cold_load_wait_secs`), now on record.

**Gap found (actionable now, independent of any decision): the gate has zero production observability.** `waiterCount`/`isHeld` are test-only; no wait-time metric, no queue depth on `/healthz` or `/metrics`. FIFO also means no fairness — a 30-minute transcription or an operator convert (WP3, gate held for minutes by explicit decision) starves every queued chat request, invisibly. Whatever else happens, instrument the gate first: without wait-time data there is no evidence base for ever flipping to batching.

## What the field does (July 2026)

- **Nobody runs N independent decode threads on UMA.** The universal architecture is **one decode loop, one scheduler, continuous batching** (requests join/leave a shared forward pass) + **chunked prefill** interleave (prefill is the compute-bound part; mlx-lm segments prompts at 2048 tokens between decode steps so a long prefill can't stall the batch).
- mlx-lm (Python): `BatchGenerator` merged 2025-09; `mlx_lm.server` continuous batching 2025-12 (decode-concurrency 32, prompt-concurrency 8; batching disabled when speculative draft set or caches non-mergeable).
- llama.cpp/Ollama: parallel slots + continuous batching on Metal for years; Ollama auto-picks 4 or 1 slots by free memory (measured: ~3–4× throughput for 20–40% per-request latency).
- vLLM on Apple Silicon is no longer a "no": `vllm-project/vllm-metal` (official-org plugin, MLX compute backend, paged varlen Metal attention as of v0.2.0) and `vllm-mlx` (the paper system; continuous batching + prefix caching; notably serves an Anthropic-compatible API and works with Claude Code).
- MLX core 2026: a real thread-safety program landed (thread-local streams, per-stream `MTLCommandQueue`s, mutex-guarded allocator) — concurrent submission is *possible*, but it amortizes nothing and remains a hardening area (mlx #2133, #3078); it is not how anyone serves.
- **GB10/CUDA-UMA** (ties to `docs/cuda-port-audit-2026-07-04.md`): same architecture answer (batched-in-one-scheduler), arriving sooner — vLLM/TensorRT-LLM ship continuous batching as the default mode there, and GB10's ~273 GB/s (half an M4 Max) makes batch-1 decode *slower*, so batching pays even earlier on that hardware.

## The three parallelism models, formalized

**(a) Continuous batching in one gated span — the sanctioned future path.**
ADR 029 permits it as written. Blockers, in dependency order: substrate batch decode loop (upstream/substrate-first per ADR 028 — watch/port mlx-swift-lm #42; primitives already upstream), Athena's vendored loops are batch-1 (plain greedy/sampling batches first; **speculative/MTP requests bypass the batch** — same rule mlx-lm ships), a scheduler above `generateMetered` (join/leave, same-model only, WP6 rebind = batch barrier), **per-sequence KV accounting in the governor** (today there is zero per-request memory accounting — N sequences' KV growth would silently eat the ADR-023-truthful budget; PagedAttention's UMA-relevant lesson is exactly KV budget control), per-sequence stream fan-out + leave-not-cancel semantics, prefix-cache per-row snapshot handling, and batch-wide 503 on a latched fault (acceptable, stated). Trigger condition: gate metrics showing routine multi-client contention.

**(b) Static batching for the non-KV tenants — cheap, optional.**
Embeddings already bucket-batch within a request; cross-request coalescing (micro-batch window) is pure Athena-side scheduler work, no substrate change, gate unchanged. Whisper/Parakeet could batch chunks of one file per forward (substrate-local tensor change; real lever for long audio). Low priority given current caller patterns (the consuming application sends per-doc embedding requests).

**(c) Cross-tenant execution overlap — forbidden, and the prohibition is load-bearing.**
MLX-Swift's allocator, buffer cache, fault handler, and `clearCache` are process-global; separate streams give kernel interleaving, not allocator/fault isolation; nothing like CUDA MPS partitioning exists on Metal. The wedge history is direct evidence. Even if safe it buys little: one pool can't parallelize two large graphs, and no batching exists across different models. Would require upstream MLX per-tenant isolation that has no roadmap signal. Stays banned under ADR 029/011; ADR 038 restates it.

## Recommended actions

1. **Now (small, unconditional):** gate observability — queue depth + wait-time histogram on `/metrics`, `gate_waiters`/`gate_held_ms` on `/healthz`; log a notice when wait exceeds a threshold. This is the evidence instrument for every future decision here. Optionally: a fairness note in docs (long transcription/convert starves chat — known, FIFO, by design).
2. **Formalize (this session):** ADR 038 — serialization posture, corrected physics rationale, in-span batching as the sanctioned path with its trigger condition and blocker list, cross-tenant ban restated.
3. **Watch (no code):** mlx-swift-lm #42 (a community Swift `BatchGenerator` port is being discussed; the Python reference is ~800 lines) — file it in the MLX tracker per the new issue-tracking convention when that lands.
4. **Later, evidence-gated:** if gate metrics show routine ≥2-deep chat contention, spec the in-span batching milestone (substrate-first); until then, MTP speculative (batch-1's best friend) remains the right optimization for the actual single-client workload.

## Sources

Athena map: `RateLimit.swift`, `InferenceGate.swift`, ADR 029/011 (quotes at file:line in the preserved raw report), `MLXLLMModule.swift`, `MLXEmbeddingModule.swift:365–403`, `PrefixKVCache.swift`, `MemoryGovernor.swift`, `MetalFaultLatch.swift`, vendored loops (`GuidedGreedy`/`SpeculativeGeneration`), substrate `Evaluate.swift:742`/`KVCache.swift:1240ff`/`SerialAccessContainer.swift`.
Field: [vllm-mlx paper (arXiv 2601.19139)](https://arxiv.org/html/2601.19139v2) · [mlx-lm PR #443](https://github.com/ml-explore/mlx-lm/pull/443) · [mlx-lm #499](https://github.com/ml-explore/mlx-lm/issues/499) · [mlx-swift-lm #42](https://github.com/ml-explore/mlx-swift-lm/issues/42) · [mlx #2133](https://github.com/ml-explore/mlx/issues/2133) · [mlx #3078](https://github.com/ml-explore/mlx/issues/3078) · [llama.cpp batched-bench](https://github.com/ggml-org/llama.cpp/tree/master/tools/batched-bench) · [Ollama FAQ](https://docs.ollama.com/faq) · [LM Studio parallel requests](https://lmstudio.ai/docs/app/advanced/parallel-requests) · [vllm-project/vllm-metal](https://github.com/vllm-project/vllm-metal) · [waybarrios/vllm-mlx](https://github.com/waybarrios/vllm-mlx) · [oMLX](https://github.com/jundot/omlx) · [awni: MLX-LM on DGX Spark](https://gist.github.com/awni/95112d214b7ff6b3fae30a7bb1ec33a9) · [vLLM on DGX Spark](https://vllm-project.github.io/2026/06/01/vllm-dgx-spark.html)
