# Concurrency supplement — DGX Spark vs Apple Silicon, and what it means for Athena

Supplement to `docs/parallel-inference-audit-2026-07-04.md` and ADR 038 (`docs/decisions/038-serialize-execution-batch-within-the-span.md`). Prompted by an operator data point: a YouTube demo (`youtu.be/Ze5XLooTt6g`) plus the operator's own vLLM run at 64 concurrent requests, where a DGX Spark's throughput scaled "dramatically" and a Mac scaled far less, with the presenter concluding "MLX doesn't support the kind of concurrency that CUDA does."

This supplement does three things: verifies the external numbers, **corrects the headline figure**, and separates the two *distinct* causes the presenter's conclusion conflates. It does not change ADR 038's decision — it sharpens the *why* and attaches honest targets to the "next big win."

## TL;DR

- **The "~1500 tok/s" is real aggregate decode, and it resolves cleanly: 64 concurrent requests × ~24 tok/s each ≈ 1,536 tok/s summed across the machine.** The ~24 tok/s per-request rate matches the verified vLLM DGX Spark decode figure (22.7–23.7 tok/s) exactly — so this is genuine aggregate *output* throughput, not prefill. Against a ~50 tok/s single-stream baseline that is a ~30× aggregate multiplier, consistent with the 26–120× range documented elsewhere.
- **The number that actually separates DGX from Mac is not the aggregate — it's per-request-rate *retention* under concurrency.** The DGX held ~24 tok/s per request all the way to 64-way (near-linear aggregate scaling). Apple Silicon can't: it falls off the linear regime early (2.6–4.3× aggregate by ~16-way), so per-request rate collapses much faster as concurrency climbs. That retention gap — not any single headline number — is the real subject of this supplement.
- **The DGX-vs-Mac gap has two compounding causes, and the presenter named only one.** (1) *Software:* mlx-swift-lm has **no shipped batched decode loop** (issue #42 open), so Athena's substrate literally cannot batch today — CUDA-land has mature PagedAttention + continuous batching. (2) *Physics:* Apple Silicon's **roofline ridge (~60 FLOP/byte) is ~8–30× lower than GB10's**, so even with a perfect batching engine the Mac's batching *ceiling* is intrinsically lower. "MLX doesn't support concurrency like CUDA" is the software half; it is not the whole story, and treating it as the whole story sets the wrong target.
- **Apple Silicon's verified batching multiplier today is 2.6–4.3× at 16 sequences** — but that is **kernel-maturity-limited, not physics-limited**. The physics ceiling is ~20–30×; realized batching is well below it because Metal batched-attention kernels are immature (the in-house gemv→steel_gemm tell). As those kernels mature, the Mac number climbs toward its ceiling — off a **~2× higher batch-1 base** than the Spark.
- **Can MLX close the gap? Mostly software, with a rising silicon floor.** Most of the *measured* gap is software (no Swift batch loop + immature Metal batch kernels) — recoverable ~4× → ~15–20× on current silicon. A residual gap is the silicon **ridge** (Apple's compute:bandwidth ratio), which caps the Mac's *multiplier* below the Spark's — but **M5's new GPU Neural Accelerators (tensor cores) raise that ridge**, and they lift the *batched* regime specifically (Apple confirms batch-1 decode stays bandwidth-bound; batching is what makes decode compute-bound). So the floor is climbing on its own. See §6.
- **For the actual goal (concurrent sub-agents on one Mac Studio): batching is the right lever, and even today's 3–4× makes it worthwhile** — one host can serve ~8–16 light concurrent agents at ~3–4× aggregate throughput, trading per-agent latency for count. The path is unchanged: continuous batching **inside one gated span** (ADR 038), substrate-gated on mlx-swift-lm #42 (now with a portable prototype branch to watch/port).

## 1 · Sizing the headline number

The operator's actual measurement: **64 concurrent requests, each holding ~24 tok/s → ~1,536 tok/s aggregate** across the machine. That is the ~1500. It is aggregate machine *output*, not per-request and not prefill — and the ~24 tok/s per-request figure lands exactly on the verified vLLM DGX Spark decode rate (22.7–23.7 tok/s), which confirms it. Against a ~50 tok/s single-stream baseline that is a ~30× aggregate multiplier. Aggregate machine output is precisely the right metric for "many sub-agents on one host," so the operator's framing is the correct one. The verified DGX Spark figures below (larger models, so lower per-request base) show the same shape of near-linear aggregate scaling under vLLM (PagedAttention + continuous batching):

| System (engine) | Model / quant | Batch-1 decode | Peak aggregate decode | Multiplier | Concurrency |
|---|---|---|---|---|---|
| DGX Spark (vLLM) | Nemotron Super 49B NVFP4 | 5.79 tok/s | **695 tok/s** | **120×** | 256 |
| DGX Spark (vLLM) | gpt-oss 120B MXFP4 | 33.5 tok/s | **863 tok/s** | **~26×** | 256 |
| DGX Spark (vLLM official) | (blog, `--max-num-seqs 4`) | 22.7–23.7 tok/s | — | — | capped at 4 |
| DGX Spark (NVIDIA blog) | prompt-processing | — | **1,725 tok/s** (prefill) | — | — |

Two honest observations from this table:
- The multiplier is real and large (26–120×). These *large* models cap at ~700–860 aggregate decode tok/s; the operator's smaller model has a higher per-request base (~24 vs ~3–5 tok/s), so its ~1,536 aggregate at 64-way is consistent. (Only *prefill* stays separate — DGX Spark prompt-processing peaks ~1,725 tok/s — so watch for a *blended* "tok/s" that folds prompt-processing into the output number.)
- Even NVIDIA's own vLLM blog **caps concurrency at 4** for its headline decode figures and notes that beyond ~4 concurrent decode streams you lose to the bandwidth tax — the *per-request* rate falls even as *aggregate* rises. On the Spark that fall is gentle (it held ~24 tok/s to 64-way); on Apple Silicon it is steep. That difference is the point of §2.

## 2 · Why DGX scales 26–120× and Apple Silicon 2.6–4.3× — the two causes, separated

Batching wins because a decode step reads each **weight once** and applies it to N tokens; the aggregate ceiling is where **compute** or **KV-cache bandwidth** catches up to the weight read. How much headroom exists at batch-1 is set by the **roofline ridge point** = peak-compute ÷ bandwidth (FLOP/byte). Batch-1 decode sits at ~2–4 FLOP/byte on either machine, so the fraction of compute left idle — and thus the batching headroom — is governed by the ridge.

| | DGX Spark / GB10 | M4 Max | M5 Max (measured, in-house) |
|---|---|---|---|
| Bandwidth | 273 GB/s | ~546 GB/s | **555 GB/s measured** (614 peak) |
| Peak compute | ~500 TOPS FP4 dense (~125 TFLOP BF16 est.) | ~34 TFLOP BF16 | ~34–40 TFLOP BF16 |
| **Ridge (FLOP/byte)** | **~460 (BF16) – 1830 (FP4)** | **~60** | **~60** |
| Batch-1 intensity | ~4 FLOP/byte | ~2 FLOP/byte | ~2 FLOP/byte |
| **Idle-compute headroom** | **~115–450×** | **~30×** | **~30×** |
| Compute-saturation batch (≈ ridge · bytes/param ÷ 2) | ~hundreds | **~30** | ~30 |

So there are **two independent multipliers stacked** on the DGX-vs-Mac gap, and the presenter's conclusion captures only the first:

**Cause A — software / kernel maturity (the presenter's point, correct as far as it goes).**
- mlx-**swift**-lm has the KV batch *primitives* but **no batched decode loop** — [issue #42](https://github.com/ml-explore/mlx-swift-lm/issues/42) is **open and unmerged** (July 2026). So Athena's substrate cannot batch at all today; the only Swift-side parallelism available is concurrent *unbatched* decodes, which ADR 038 correctly forbids.
- CUDA-land (vLLM/TensorRT-LLM) ships mature PagedAttention + continuous batching that **realizes near-linear scaling deep into the batch** (DGX Nemotron is still ~7.8× at concurrency 8 — barely off linear).
- On Apple Silicon the batching that *does* exist falls off linear **early**: the in-house `decode-throughput-gap-analysis.md` Metal capture shows the tell — the efficient `gemv` kernel is used only at M=1 (420 GB/s); at M≥2 the forward falls through to `steel_gemm` at 278 GB/s, a flat ~1.5× bandwidth-efficiency loss the moment you leave batch-1. Metal batched/paged-attention kernels are the immature layer, not the physics.

**Cause B — hardware ridge ceiling (the presenter's blind spot).**
- Even with a *perfect* batching engine, the Mac's ~60 ridge caps the batching multiplier at ~20–30× (compute-saturation batch ~30), where GB10's ~460–1830 ridge leaves room for 100×+ before compute binds (KV bandwidth binds it first, near concurrency 256).
- **The Mac's higher bandwidth cuts both ways**, which is the counter-intuitive correction to "faster memory ⇒ bigger concurrency win": more bandwidth gives a **better batch-1 latency** but a **lower ridge**, hence a **smaller multiplier**. The Spark's giant multiplier is partly a *symptom* of being bandwidth-starved relative to its compute — it has more idle compute to reclaim precisely because its bandwidth is half the Mac's.

Net: DGX-vs-Mac gap ≈ (lower physics ceiling: ~30× vs ~450×) × (lower realized fraction: immature Metal kernels vs mature PagedAttention). Both are real; only closing Cause A is in our control, and even fully closed it lands at the ~30× ceiling, not 120×.

## 3 · Apple-Silicon batching reference points (verified)

| Source | Model | Multiplier | Notes |
|---|---|---|---|
| vllm-mlx paper (arXiv 2601.19139), M4 Max | across sizes | **2.6× – 4.3× @16** | peak ~525 tok/s text; 21–87% faster than llama.cpp |
| mlx-lm continuous batching (Python), via same paper | Qwen3-0.6B | 441 → 1,642 tok/s **@16 (3.7×)** | up to 4.14× at 8× concurrency |
| mlx-lm | Qwen3-8B | ~2.6× | bandwidth-saturated (bigger model → lower multiplier) |
| mlx-swift-lm #42 **prototype branch** | Llama-3.2-3B-4bit, M5 | 62 → 118 → **344 tok/s** (batch 1→2→32) | **5.5× at batch 32** — a *Swift* MLX batch path, prototyped, **not merged** |

The #42 prototype (rudrankriyam) is the most Athena-relevant line in the table: it is a working *Swift* batch-generation branch, which is exactly the substrate-first port target ADR 038's blocker list points at. Its 5.5× at batch-32 on a 3B is consistent with the physics: small model, tiny KV, near the achievable Metal ceiling.

## 4 · Athena-specific wrinkle: the served model is an MoE

The studio serves `gemma-4-26b-a4b-it-8bit` — a 128-expert MoE with ~4B active params/token. This changes the batching economics **against** us relative to a dense model:
- **Batch-1 is cheap** (reads only the active experts, ~4B worth) — great for single-client latency, which is why MoE is a good serving choice today.
- **Batching amortizes weight reads *worse*.** Different tokens in a batch route to different experts, so the *union* of activated experts grows toward all 128 as batch size rises → per-step weight reads climb toward the full 26B instead of staying at 4B. A dense model reads the same constant weight set for any batch size; an MoE's effective weight read grows with batch diversity. So the MoE's batching multiplier is **softer** than a dense model of the same active size, and it needs a **batched, expert-grouped `SwitchGLU` gather-matmul** — an extra substrate-maturity dependency on top of #42.

Practical implication: don't benchmark the batching win on the MoE and generalize it, and don't expect the small-dense-model multipliers (the #42 prototype's 5.5×) to transfer to the 26B MoE. If batching throughput becomes the priority, a **dense** 8-bit checkpoint (the throughput doc's Qwen3.5-27B-8bit at 26 tok/s batch-1, 414 GB/s) may batch more efficiently than the MoE, and is worth an explicit A/B.

## 5 · Physics ceiling vs realized ceiling — the optimistic read

Apple Silicon's **realized** batching today (~4× @16) is far below its **physics** ceiling (~20–30×). That gap is Cause A (kernel maturity), which is improving on a visible track: `vllm-project/vllm-metal` shipped paged varlen Metal attention (v0.2.0); the MLX thread-safety/allocator program landed; mlx-lm's Python continuous batching is production. As the Metal batched-attention path matures, the Mac's near-linear regime should extend from ~batch-4 toward ~batch-16–30, i.e. from ~4× toward ~10–20×.

And the base matters: the Mac's batch-1 decode is **~2× faster than the Spark's** (555 vs 273 GB/s → e.g. ~26 tok/s vs ~12 tok/s for a comparable 27B-class model). So a smaller Mac multiplier off a higher base is more competitive on *aggregate* than 4× vs 120× suggests: ~26 tok/s × ~10× (mature) ≈ 260 aggregate tok/s per model — in the same order of magnitude as the Spark's ~700, on hardware with 2× the bandwidth and a fraction of the power.

## 6 · Can MLX close the gap? — software vs silicon (the M5 inflection)

The gap decomposes into three layers. Only the first is fully in software's hands; the third is silicon — but it is *rising*, and M5 just moved it in exactly the regime that matters for concurrency.

**Layer 1 — MLX / kernels (software; most of the gap you see today; fully recoverable).**
- mlx-swift-lm has no batched decode loop at all (#42 open) — Athena's Swift substrate gets 0× batching today, so the *entire* multiplier is unwritten code, not a hardware limit (Python mlx-lm already ships it).
- The Metal batched-attention / paged-KV / MoE-`SwitchGLU` kernels are young: the in-house gemv→steel_gemm 1.5× cliff at M≥2; vllm-metal's paged attention is months old against CUDA PagedAttention's years.
- Recoverable range: ~4× → toward the ~15–20× that current silicon physically allows. This is the biggest lever and it is 100% software.

**Layer 2 — Metal the API (mostly *not* the limiter).**
- Metal is a capable low-level API and, as of Metal 4, exposes the tensor hardware directly (Tensor APIs / Metal Performance Primitives). "Metal can't do what CUDA does" was really "the kernels weren't written yet" plus "there was no matmul hardware to target" — not an API ceiling.
- The one genuine Metal-vs-CUDA capability gap relevant here is MPS-style multi-tenant *partitioning* (isolation) — which matters for cross-tenant overlap (ADR 038 bans it anyway), not for batching throughput. So "CUDA vs Metal" is ~90% ecosystem maturity and ~10% a feature we don't need.

**Layer 3 — Silicon (the real floor — but it is rising, and M5 is the inflection).**
- The batching ceiling is set by the roofline **ridge** (compute ÷ bandwidth), which is silicon. On M4 and earlier there are **no tensor cores** — matmul runs on general GPU ALUs (a reverse-engineering study finds Metal's matmul primitives lower onto the simdgroup_matrix / FP32-ALU path, beating plain simdgroup_matrix by only 1.05–1.21×). That pins the M4 ridge ~60 and the batching ceiling ~20–30×, far below GB10's ~100–450×.
- **M5 changes the hardware in exactly this regime.** M5 adds GPU **Neural Accelerators** — dedicated matrix units (tensor cores) — programmable via Metal 4, with >4× peak GPU compute vs M4 and a measured 3.65× MLX *prefill* speedup (Qwen3-8B, 20k prompt: 158→579 tok/s). More compute per byte **raises the ridge**, which **raises the batching ceiling**.
- The crucial subtlety: Apple notes **batch-1 decode stays bandwidth-bound even on M5** — tensor cores don't help a single token (a pure bandwidth problem). But **batching is precisely the operation that turns decode compute-bound**, so M5's tensor cores lift the *batched* ceiling specifically — the exact regime that matters for running many sub-agents. The hardware Apple just shipped targets the concurrency case, not the single-stream case.
- MLX support for the Neural Accelerators is **preliminary today, full support expected later in 2026** — landing now, not speculative.

**Bottom line.** Yes — MLX can be updated to get substantially closer, and *most* of the gap measured is software (no Swift batch loop + immature Metal batch kernels), worth ~4× → ~15–20× on current silicon. A residual gap is silicon (the ridge), and Apple Silicon will likely never match the Spark's raw concurrency *multiplier* — its FP4 tensor throughput against half the Mac's bandwidth gives it 100×+ of idle-compute headroom to reclaim. But Athena does not need to match the multiplier: Mac's ~2× bandwidth gives a higher per-request base (better latency, competitive *aggregate*), and the silicon floor is rising each generation — M5's tensor cores already close much of the *batched*-regime gap. The Athena win is gated almost entirely on software right now (mlx-swift-lm #42, Metal batch kernels, MLX M5 tensor-core support), riding a hardware floor that is climbing on its own.

> **Note — FP4 quantization (MXFP4 / NVFP4) is the same "portable format vs hardware-gated compute" split.** NVIDIA's NVFP4 (E2M1 elements + a 16-element FP8-E4M3 block scale + an FP32 global scale) is more accurate at 4-bit than MXFP4 (32-element blocks, power-of-two E8M0 scale) and is natively tensor-core-accelerated on Blackwell. It is **not CUDA-only**: MLX already lists MXFP4/NVFP4 among its quantization modes (accelerated on its CPU/CUDA backends) and llama.cpp landed NVFP4 cross-platform — the *format/codec* is portable to Metal. What is Blackwell-only is **native FP4 tensor-core matmul** (computing in FP4 at ~2× FP8 throughput); on Apple Silicon you dequantize NVFP4 → bf16/fp16, so you get the **memory + accuracy** benefit but not the compute-throughput benefit until Metal exposes FP4 tensor ops (M5 Neural Accelerators — TBD). For Athena this aligns well with the roofline: decode is bandwidth-bound, so the *portable* half — **accuracy-per-byte at 4-bit** (a good NVFP4-class codec could serve 4-bit at ~8-bit quality, closing the int4-affine quality gap that currently favors 8-bit in `decode-throughput-gap-analysis.md`) — is exactly the useful half. The FP4 *compute* win lands in the batched/prefill regime and rides the same rising-silicon-floor curve as batching.

## 7 · Does this change ADR 038? No — it sharpens it.

The decision stands: **serialize execution; batch inside one gated span; substrate- and evidence-gated; cross-tenant overlap forbidden.** This supplement adds:

1. **The trigger's payoff is now quantified and honest.** When gate metrics show routine ≥2-deep contention and the substrate can batch, the realistic win on current Metal kernels is **~3–4× aggregate** (climbing toward ~10–20× as kernels mature), **not** the ~30–120× a CUDA data point implies. Plan the milestone against 3–4×, treat more as upside.
2. **The #42 watch item is now concrete.** It is no longer "wait for upstream to design batching" — a working Swift prototype branch exists (62→344 tok/s, batch 1→32). The port target is real; track whether it merges, and evaluate porting it behind the ADR 028 substrate-first rule if it stalls.
3. **New blocker for Athena specifically: MoE-batched `SwitchGLU`.** Add to ADR 038's blocker list — the served model is an MoE, and batched expert-grouped gather-matmul is a distinct substrate capability from the dense batched decode loop. Consider a dense 8-bit checkpoint as the first batching target if/when this lands.
4. **Reaffirmed first step, unchanged: InferenceGate observability.** Everything above is still gated on having wait-time/queue-depth data. No batching milestone should precede the gate instrument — without it there is no evidence base for the trigger, and no baseline to measure a batching win against.

The presenter's conclusion, corrected for the record: it is true that **mlx-swift-lm doesn't yet ship the continuous batching CUDA has** — but Apple Silicon *also* has a lower roofline ceiling, so closing the software gap gets us to ~30×-headroom hardware, not ~450×. Both facts belong in the plan; optimizing for a CUDA-shaped 120× target on Metal would be chasing a ceiling the hardware doesn't have.

## Sources

External (verified 2026-07-04): [DGX Spark hardware (NVIDIA docs)](https://docs.nvidia.com/dgx/dgx-spark/hardware.html) · [Dendro Logic DGX Spark concurrency benchmark](https://dendro-logic.com/engineering/nvidia-dgx-spark-concurrency-benchmark/) · [vLLM DGX Spark blog](https://vllm-project.github.io/2026/06/01/vllm-dgx-spark.html) · [NVIDIA DGX Spark performance blog](https://developer.nvidia.com/blog/how-nvidia-dgx-sparks-performance-enables-intensive-ai-tasks/) · [mlx-swift-lm #42](https://github.com/ml-explore/mlx-swift-lm/issues/42) · [mlx-lm HTTP server (continuous batching)](https://deepwiki.com/ml-explore/mlx-lm/3.3-http-server) · [vllm-mlx paper, arXiv 2601.19139](https://arxiv.org/abs/2601.19139) · roofline: [arXiv 2605.30571](https://arxiv.org/abs/2605.30571), [arXiv 2503.08311](https://arxiv.org/html/2503.08311v2) · M5 tensor cores: [Apple ML Research — Exploring LLMs with MLX and the M5 Neural Accelerators](https://machinelearning.apple.com/research/exploring-llms-mlx-m5), [Apple M5 newsroom](https://www.apple.com/newsroom/2025/10/apple-unleashes-m5-the-next-big-leap-in-ai-performance-for-apple-silicon/) · Metal matmul path: [Rigel — reverse-engineering the Metal 4.1 tensor path on M4 Max (arXiv 2606.12765)](https://arxiv.org/pdf/2606.12765) · [M5 Max vs DGX Spark benchmark (Skorppio)](https://skorppio.com/blog/apple-m5-max-vs-nvidia-ai-deep-dive) · FP4 quant: [NVIDIA — Introducing NVFP4](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/), [MLX quantization modes (DeepWiki)](https://deepwiki.com/ml-explore/mlx/7-quantization), [Awni Hannun on MXFP4/NVFP4 in MLX](https://x.com/awnihannun/status/1961500133990043967), [NVFP4 in llama.cpp](https://insiderllm.com/guides/fp4-inference-llamacpp-nvfp4-mxfp4/) · operator video: `youtu.be/Ze5XLooTt6g`.
In-house: `docs/parallel-inference-audit-2026-07-04.md`, `docs/decisions/038-serialize-execution-batch-within-the-span.md`, `docs/decode-throughput-gap-analysis.md` (measured M5 Max: 555 GB/s, gemv M=1 420 GB/s vs steel_gemm M≥2 278 GB/s, 8-bit 27B 26 tok/s).
