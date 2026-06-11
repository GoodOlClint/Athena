# Decode-throughput gap analysis — Qwen3.5-27B on M5 Max

Performance characterization of the Athena LLM decode pipeline, measured on
the host below at appVersion 0.10.86 (+ env-gated perf-trace instrumentation,
see end). Every ceiling here comes from **measured** bandwidth and **measured**
per-forward bytes, not theoretical peak or labels.

## TL;DR

- The structured guide, per-token GPU→CPU sync, the 248k-row argmax/softmax,
  and `guidedArgmax` are **all negligible** (0.8% of decode time combined).
  Suspects #1, #2, #4 from the brief are ruled out.
- **4-bit is essentially at its practical ceiling.** ~31–32 tok/s decode; the
  backbone forward is 92% of decode time and runs at ~282 GB/s effective. The
  remaining headroom is in MLX's quantized-matmul kernels (small attention
  projections run at 30–90 GB/s), which is upstream territory.
- **8-bit is the sweet spot at 29 GB:** ~26 tok/s, 414 GB/s effective — the
  *most* bandwidth-efficient of the three.
- **The unquantized 54.6 GB model has a real, sharp pathology:** ~2.8 tok/s,
  forward runs at **79 GB/s** — 5× slower than its honest ceiling and a 3–5×
  efficiency *cliff* vs the quantized models. It is **not** memory pressure,
  size, bf16, mmap, the memoryLimit, the wired limit, swap, or the MLX
  version — all reproduced fast in isolation. Root cause is in the substrate's
  actual bf16 forward execution and needs a Metal GPU capture to pin down.
- KV cache and context length are **not** a decode bottleneck (hybrid arch:
  only 16 of 64 layers carry growing KV). For long context, **prefill**
  dominates wall time (12.4 s for 8 k tokens) and is the real lever there.

## Phase 0 — verified premises (ground truth, not assumptions)

### Host
`Apple M5 Max, MacBook Pro (Mac17,6)`, 18 CPU cores (12 P + 6 E), **40-core
GPU**, **128 GB** unified memory. macOS 26 / Metal 4.

### Memory bandwidth — MEASURED, not assumed
| Source | GB/s |
|---|---|
| Apple published peak (M5 Max, 40-core GPU) | **614** (theoretical) |
| MLX large streaming read (2 GB) — measured | **557** (91% of peak) |
| MLX large decode-like GEMV (2.5 GB fp16) — measured | **555** |
| `mbw` single-thread CPU memcpy | ~67 (copy) — *single CPU core, not the GPU path* |

**Achievable GPU bandwidth ≈ 555 GB/s** (90–91% of the 614 peak — better than
the 70–85% rule of thumb). This is the denominator for every ceiling below.
Note the small-array tax: an 8192² GEMV (0.13 GB) hit only 259 GB/s — small
ops are dispatch-bound, not bandwidth-bound. This matters for decode.

### Model architecture (from config.json, all 3 checkpoints share it)
Qwen3.5-27B is a **hybrid**: 64 layers = **16 full-attention** (GQA, 4 KV heads,
head_dim 256) + **48 linear-attention** (GatedDeltaNet/Mamba, fixed recurrent
state). hidden 5120, intermediate 17408, vocab 248320, **untied** lm_head,
`full_attention_interval=4`.

### Weight bytes — MEASURED from the safetensors headers
| Checkpoint | File | embed (not read/fwd) | **read per decode forward** |
|---|---|---|---|
| 4-bit (g64) | 15.447 GB | 0.715 | layers 13.702 + lm_head 0.715 = **14.42 GB** |
| 8-bit | 29.08 GB | ~1.27 | layers ~26.5 + lm_head ~1.27 = **~27.8 GB** |
| unquant bf16 | 54.64 GB | 2.54 | layers ~49 + lm_head 2.54 = **~51.5 GB** |

The input embedding table is **gathered one row per token** → not part of the
per-forward read. lm_head (full 248320 vocab) **is** read every forward.

### Loaded footprint — MEASURED (`/healthz` + heartbeat)
4-bit resident 15.60 GB, 8-bit 29.25 GB, unquant 51–54.8 GB. **Zero bloat** —
resident tracks file size. No growth during decode.

### KV-cache traffic @ 8500 ctx
Only the 16 full-attn layers grow: 8500 × 16 × 2(K+V) × 4 heads × 256 × 2 B
= **0.557 GB/forward**. The 48 linear-attn layers hold a fixed ~0.3 GB
recurrent state. KV is **secondary** to the ~14–51 GB weight read, confirmed
live: decode rate is essentially context-insensitive (below).

### Tokens per forward
Greedy = 1. MTP speculative yields **1 + accept_rate** tokens per backbone
forward — measured accept **0.83–0.90** → **~1.84–1.9 tok/iter** (matches
`tokens ÷ iters` exactly in every run).

## Realistic ceilings vs measured (decode phase only)

| Config | per-fwd bytes | naive ceiling @555 GB/s | **measured decode** | effective GB/s |
|---|---|---|---|---|
| 4-bit greedy | 14.42 GB | 38.5 tok/s | ~23.8 (brief) | — |
| 4-bit + spec | ~15.0 GB → ×1.84 | ~68 tok/s | **31–32** | 282 |
| 8-bit + spec | ~27.8 GB → ×1.9 | ~38 tok/s | **26.4** | **414** |
| unquant + spec | ~51.5 GB → ×1.9 | ~10.7 tok/s | **2.8** | **79** |

The naive "@555 GB/s" ceiling is an over-estimate because (a) quantized
matmuls carry dequant overhead, (b) the model has many small matrices that
don't saturate bandwidth, and (c) ~30–60% of each forward is
**precision-independent architecture compute** (GDN scans, attention,
layernorms) that isn't a weight read at all. The honest read: **4-bit and
8-bit sit in a reasonable regime; the unquant 79 GB/s is the outlier.**

## Stage attribution (env-gated perf trace, 4-bit, 163 iters)

| Stage | % of decode |
|---|---|
| **backbone forward** | **92.0%** |
| MTP draft forward | 6.5% |
| verify + draft sampling (argmax + `.item()` sync) | **0.4%** |
| bonus/next-draft pick | 0.4% |
| KV trim + Mamba rollback | 0.0% |

Identical shape at 8-bit (94%/4.8%/0.3%) and unquant (95.2%/4.6%/0.0%).

**This overturns the brief's leading suspects.** The per-token GPU→CPU sync
(#1), the LM-head/softmax/argmax over 248320 rows (#2), and `guidedArgmax`'s
mask alloc + 1 MB transfer (#4) together cost **<1%**. Decode time *is* the
backbone forward. Do not spend effort on a fused masked-argmax or async-sample
overlap — the win is sub-1%.

### Why 4-bit "only" gets 282 GB/s effective (model-shape microbench)
Per-layer 4-bit `quantized_matmul`, measured:

| matrix | bytes | effective GB/s |
|---|---|---|
| q_proj 5120→6144 | 17.7 MB | 65–89 |
| kv_proj 5120→1024 | 5.9 MB | 30–34 |
| o_proj 6144→5120 | 17.7 MB | 88–93 |
| gate+up 5120→34816 | 100 MB | 265–277 |
| down 17408→5120 | 50 MB | 196–199 |
| lm_head 5120→248320 | 715 MB | 474–486 |

The big matmuls (MLP, lm_head) approach the bandwidth limit; the **small
attention projections are dispatch- and dequant-bound at 30–90 GB/s** and drag
the aggregate to ~330 GB/s (the real fused forward does ~282). This is an MLX
quantized-kernel-efficiency property, not an Athena-loop bug.

### Speculative is working, but the tax is real
- spec 55.8 ms/iter → 1.84 tok → 30.3 ms/tok = 32.3 tok/s.
- Realized speedup over greedy ≈ **1.36×** (32.3 / 23.8) vs ideal token-yield
  1.84×. The gap is the speculation tax: the 2-position verify forward costs
  ~9 ms more than a 1-position greedy forward, plus the MTP draft forward
  (3.65 ms/iter, which includes a full lm_head re-projection ≈ 0.715 GB).
  Still a net win.

## Long context (8078-token prompt)
- **prefill 12.4 s** (651 tok/s, compute-bound) — dominates wall time.
- decode 250 tok / 8.04 s = **31.1 tok/s** — *identical* to short-context 32.3.
  Per-iter only rose 56→59 ms despite 8 k KV. **KV is not the bottleneck**
  (suspect #3 ruled out). The "12–18 tok/s" numbers in the wild are
  `completion_tokens ÷ total_time`, polluted by prefill — always read the
  heartbeat `phase=decode tokens_per_sec`.

## The unquantized cliff (suspect #8) — characterized and bounded

Measured: **2.8 tok/s, 656 ms/backbone-forward, 79 GB/s effective.** A 3–5×
*cliff* vs the quantized models, not smooth scaling (8-bit at 29 GB is the most
efficient point at 414 GB/s).

**Systematically ruled out** — each reproduced at full speed in isolation
(standalone MLX 0.31.2, matching Athena's pinned mlx-swift 0.31.3):
- **Memory pressure / eviction / swap** — heartbeat during decode: mlx_active
  53 GB, mlx_cache 6 GB (stable, not collapsing), resident 51 GB, against a
  96 GiB memoryLimit and 107 GiB recommended working set. RSS flat, swap flat.
  Tons of headroom.
- **Working-set size** — a 52 GB bf16 independent-matmul sweep: 512 GB/s.
- **Sequential dependency** — a 52 GB bf16 sequential chain (1000 layers,
  *more* dispatch overhead than the real 64): 262 GB/s.
- **bf16 intrinsic / the real weights** — loading the actual 54.6 GB file and
  doing GEMVs over its real tensors: 505 GB/s (mmap **and** forced device copy
  both 505–510, so mmap is not it).
- **memoryLimit** — sweep at 55/60/96 GB limits: all 513–535 GB/s.
- **wired limit** — nobody sets one; forcing a tiny one in isolation changes
  nothing (262 GB/s).
- **MLX version** — pinned mlx-swift 0.31.3 ≈ test harness 0.31.2.

### Localized: the non-quantized bf16 matmul kernel (follow-up investigation)
Cheap-first follow-up narrowed it further. Speculation is innocent — **greedy
(non-spec) unquant also craters** at 1.9 tok/s (98 GB/s, 1-position forward).
The MLX core is innocent — mlx-swift 0.31.3 embeds MLX C++ core **0.31.1** (one
patch behind the test harness's 0.31.2, *same* version scheme), and pinning
Python to **0.31.1** still reproduces fast (262–510 GB/s). Isolation is fully
exhausted: 3D/batched input, real MLP shapes, square/rectangular, sequential —
all 312–544 GB/s. The model is **dense** (no MoE; `num_experts` absent), so the
gather-matmul path isn't involved.

An env-gated in-forward profiler (`ATHENA_FWD_PROFILE=1`, splits each decoder
layer into token-mixing vs MLP) gives the decisive split:

| block | 4-bit (quantized) per-block | unquant (bf16) per-block |
|---|---|---|
| MLP | 0.66 ms / 0.134 GB = **204 GB/s** | 6.2 ms / 0.534 GB = **86 GB/s** |
| GDN | 0.63 ms | **4.6 ms (7.3×)** |
| attn | — | ~2.6 ms |
| layer split | MLP 53% / GDN 38% / attn 9% | MLP 60% / GDN 34% / attn 6% |

(Both runs equally perturbed by the profiler's per-block `eval()`; read the
ratios.) **The GDN block is the tell:** its recurrence math is
precision-independent, so unquant-GDN being 7.3× slower can *only* come from
its bf16 `Linear` in/out projections. The common factor across MLP and GDN is
the **non-quantized bf16 matmul** — it runs ~2.4× slower *per byte* than the
fused `quantized_matmul`, and ~5–6× slower than the *same* bf16 matmul in
isolation. So the cliff is an **MLX kernel-efficiency property of the
non-quantized matmul path** in the real autoregressive forward (small batch,
many sequential dependent matmuls fed by computed intermediates) that the
fused quantized path handles well and that standalone matmuls don't expose.
It is **not** an Athena bug, a loading bug, or a memory issue.

### Controlled confirmation: 9B-unquant vs 27B-4bit (path, not size)
A matched-bytes experiment settles "is it the kernel path or just the weight
size?" `Qwen/Qwen3.5-9B` (same `qwen3_5` hybrid arch — 32 layers, 24 linear +
8 full, MTP head) converted to **unquant MLX** (17.5 GB). Its per-forward read
(~15.5 GB: layers ~13.5 + lm_head 2.0) is **nearly identical to the 27B-4bit's
14.42 GB** — so bytes are matched and only the kernel path differs.

| Model | params | bytes/forward | decode (spec) | backbone eff. | path |
|---|---|---|---|---|---|
| 27B-4bit | 27B | 14.42 GB | **~31 tok/s** | 282 GB/s | quantized qmv/qmm |
| 9B-unquant | 9B | ~15.5 GB | **8.4 tok/s** | **75 GB/s** | non-quant gemv/steel |

The 9B-unquant decodes **~3.8× slower than the 27B-4bit at matched
per-forward bytes — with one-third the parameters.** A pure size/bytes
explanation predicts ~parity; the 3.8× inversion can only be the kernel path.
Decisively, the 9B working set is **18.5 GB resident** — far below the 52 GB
bf16 set that ran at 262–512 GB/s in isolation — yet it still craters to
75 GB/s. **This rules out working-set size as a contributor entirely: the cliff
is 100% the non-quantized matmul kernel path, 0% size.** (Same fwd-profile
shape as the 27B-unquant: MLP 57% / GDN 36% / attn 6%.)

### Metal capture — kernel-level confirmation + fork-fix assessment
A real capture (`ATHENA_METAL_CAPTURE=<path>` + `MTL_CAPTURE_ENABLED=1`, grabs
one warm decode forward) was taken and the dispatched kernels mined from the
`.gputrace` bundle. The M=2 (speculative) forward dispatches:
- `steel_gemm_fused_nax_nt_bfloat16_bfloat16_bm64_bn128_bk256` — the bf16
  MLP/projection matmuls, on a **64-row output tile** fed an M=2 input.
- `steel_gemm_splitk_nt_bfloat16_float32` / `…_float32_float32` — the GDN block.
- **No `gemv` kernel anywhere.**

Reading MLX's `matmul.cpp` dispatch explains it: the efficient `gemv` kernel is
selected only when `std::min(M, N) == 1`. **Greedy decode is M=1 → gemv;
speculative decode feeds `[prev, draft]` = M=2 → falls through to
`steel_matmul`.** Isolated at the decode MLP shape (weights read once either
way):

| M | kernel | GB/s |
|---|---|---|
| 1 | gemv | **420** |
| 2 | steel_gemm | 278 |
| 3, 8 | steel_gemm | ~277 |
| 1/2/4 quantized | qmv/qmm | 246–259 |

So it's a **flat ~1.5× bandwidth-efficiency gap between `gemv` and
`steel_gemm`** for these tall-skinny reads — not tile-waste that scales with M
(M=2..8 are identical). Note the naive "loop `gemv` M times" fix is *wrong*: it
re-reads the weights M times (2× gemv = 1688 µs > steel M=2 = 1281 µs).

**Fork-fix assessment (the question that prompted the capture):**
- **Verified, moderate-effort, bounded:** the non-quantized path has no
  decode-tuned small-M kernel — a `gemv`-class kernel that handles small M
  (2–8) in a single weight pass would recover ~1.5× on the bf16 matmuls in the
  **speculative** path. This is a legitimate MLX-core/kernel change (a fork of
  mlx-swift's vendored `Cmlx/mlx` Metal kernels + `matmul.cpp` routing), not an
  Athena change. Greedy (M=1) already uses `gemv`, so it gets no benefit.
- **The larger 2.4× per-byte gap** (in-forward bf16 MLP 86 GB/s vs quantized
  204 GB/s, from `ATHENA_FWD_PROFILE`) is because **MLX's quantized kernels
  (`qmv`/`qmm`) are heavily decode-tuned and the non-quantized steel path is
  not**. Closing it fully is substantial upstream kernel work.
- **Net:** a fork fix could plausibly lift unquant *speculative* decode by
  ~1.5× (≈2.8 → ~4 tok/s) — still ~6× slower than 8-bit. **Not worth it unless
  bf16 serving is mandatory.** The practical answer is unchanged: **quantize
  (8-bit).** The fix is more interesting as an upstream MLX contribution
  (decode-tuned non-quantized small-M matmul) than as an Athena perf lever.

## Plan to close the gap

Ordered by value ÷ effort. Numbers are measured on this box.

### P0 — Operational (no code; biggest immediate win)
1. **Don't serve the unquantized model.** 8-bit gives ~26 tok/s at near-identical
   quality for 3.6× less memory and **~9× the throughput** (2.8 → 26). For
   the consuming application's latency budget this alone closes the practical gap. Document
   8-bit as the recommended 27B checkpoint; treat unquant as a debugging
   artifact until the cliff is fixed.

### P1 — Long-context wall time (helps the real 8 k-context use case most)
2. **Prefill tuning.** Prefill (651 tok/s, compute-bound) is 60% of an 8 k-token
   request's wall time — far more than decode. Experiment with the M48.4 chunk
   size (currently 512): larger chunks improve GPU utilization on the batched
   matmuls. One-line change in `SpeculativeGeneration`/`GuidedGreedy`; A/B the
   prefill seconds at 512 vs 1024 vs 2048. Expected: meaningful prefill
   speedup, no decode change. *Measure before shipping.*

### P2 — The unquant root cause (localized; further work optional)
3. Root cause is localized to **MLX's non-quantized bf16 matmul kernel** being
   ~2.4× less bandwidth-efficient than `quantized_matmul` in the real forward
   (see the unquant section). Speculation, MLX version, loading, memory, and
   model size are all ruled out. Remaining work is **optional** and only
   warranted if bf16 serving is mandatory:
   - **Metal GPU capture** of one unquant forward to name the slow bf16 kernel
     variant + occupancy, then **file an upstream MLX issue**. (0.31.3 is the
     latest mlx-swift tag — there is nothing to "bump" to today.)
   - Or sidestep entirely by quantizing — which is P0.

### P3 — Speculative efficiency (modest, 4-bit/8-bit)
4. **Drop the MTP draft's lm_head re-projection.** The MTP forward (6.5% of
   decode) re-projects the full 248320-vocab lm_head (~0.715 GB) just to argmax
   one draft token. Under greedy/guided we only need the top-1 — a cheaper
   top-k or a reduced-vocab projection for the draft would trim most of that
   6.5%. Bit-identical contract is preserved (the draft only proposes; the
   backbone verify still decides). Net: a few % decode gain.

### Explicitly NOT worth doing (measured <1%)
- Fused in-place masked-argmax / `guidedArgmax` rework (suspect #4): 0.4%.
- Async overlap of sampling with the next forward (suspect #1): the
  `.item()` sync is 0.4%; autoregressive data-dependency makes overlap
  near-impossible anyway.
- KV-cache / long-context attention optimization for *decode* (suspect #3):
  KV is ~0.5 GB, +3 ms/iter at 8 k. Negligible.

### Ceiling reality check
4-bit at 31–32 tok/s and 8-bit at 26 tok/s are within ~2× of their honest
ceilings once architecture compute and quantized-kernel efficiency are
accounted for. The large multiplicative wins are **operational** (use 8-bit)
and **prefill** (long context), not in the decode sampling/sync loop.

## Instrumentation added (uncommitted, env-gated)
Two env-gated perf seams, zero cost when unset. Recommend keeping both
(generalize the M53 `speculative summary` diagnostic alongside them).
- `Sources/AthenaLLM/SpeculativeGeneration.swift` — `ATHENA_PERF_TRACE=1` emits
  a per-request **stage** breakdown (`perf trace: ... backbone=… mtp=…
  verifySample=…`) by forcing `eval()` at stage boundaries so MLX's lazy graph
  doesn't fold the forward cost into the sampling sync.
- `Sources/AthenaModels/AthenaQwen35.swift` — `ATHENA_FWD_PROFILE=1` (`ForwardProfile`)
  emits a per-request **decoder-block** breakdown (`fwd profile: gdn=… attn=…
  mlp=…`) by forcing `eval()` at the token-mix / MLP boundary inside each
  layer. Perturbs absolute timing (serializes the layer graph); read ratios.
- `Sources/AthenaLLM/SpeculativeGeneration.swift` — `ATHENA_METAL_CAPTURE=<path>`
  (plus `MTL_CAPTURE_ENABLED=1`) captures the 8th warm decode forward into a
  `.gputrace` via the mlx-c `mlx_metal_start_capture`/`stop_capture` symbols
  (declared with `@_silgen_name`). Note the trace includes all bound weight
  buffers (~50 GB for the unquant model) — delete it after inspection. Mine
  dispatched kernel names with `strings` over the bundle's small (<200 KB)
  metadata files; full timing needs the Xcode GPU debugger.
