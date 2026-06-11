# DRAFT — issue for ml-explore/mlx (kernel lives in core, vendored by mlx-swift)

> Note: the routing + kernels are in **ml-explore/mlx** (`mlx/backend/metal/matmul.cpp`,
> Metal `gemv`/`steel_gemm`). mlx-swift just vendors that core, so this should
> be filed upstream on **ml-explore/mlx**. Reproduced via mlx-swift's pinned
> core (0.31.1) and standalone Python MLX 0.31.1 — identical behavior.
>
> Not yet posted — review/edit before filing.

---

**Title:** [Performance] Non-quantized matmul has no small-M kernel: M≥2 falls from `gemv` to `steel_gemm` (~1.5× slower), hurting batched/speculative decode

**Labels:** performance

### Summary

For a tall-skinny "decode" weight shape, the non-quantized (`float16`/`bfloat16`)
matmul is ~1.5× slower per byte at **M=2** than at **M=1**, and stays flat-slow
through M=32. The cause is dispatch: `gemv` is selected only when
`std::min(M, N) == 1`; any M≥2 routes to `steel_gemm`, which is tuned for large
M and underutilizes bandwidth on a 2–8-row input (weights are read once either
way, so this is a kernel-efficiency gap, **not** extra weight traffic).

This disproportionately penalizes **batched / speculative decode**, where the
verify step naturally has M = (1 + n_draft) ≥ 2. The quantized path does *not*
have this M=1→M=2 cliff (its `qmv`/`qmm` small-M path was tuned in #3120), so
non-quantized models pay a decode penalty quantized models don't.

### Environment

- Apple M5 Max (40-core GPU), 128 GB, macOS 26 / Metal 4
- MLX 0.31.1 (also reproduced via mlx-swift 0.31.3, which vendors core 0.31.1)

### Minimal repro

```python
import mlx.core as mx, time
H, N = 5120, 34816  # tall-skinny decode MLP (gate+up) shape
W = mx.random.normal((N, H)).astype(mx.float16); mx.eval(W)
wb = N*H*2/1e9
def bench(fn, it=200, wu=30):
    for _ in range(wu): mx.eval(fn())
    mx.synchronize(); t0=time.perf_counter()
    for _ in range(it): mx.eval(fn())
    mx.synchronize(); return (time.perf_counter()-t0)/it
for M in (1,2,3,4,8,16,32):
    x = mx.random.normal((M,H)).astype(mx.float16); mx.eval(x)
    dt = bench(lambda: x@W.T)
    print(f"M={M:<3d} {dt*1e6:8.1f} us  {wb/dt:7.1f} GB/s")
```

Output (M5 Max, MLX 0.31.1):

```
M=1      849.4 us    419.7 GB/s   <- gemv
M=2     1238.7 us    287.8 GB/s   <- steel_gemm   (1.46x slower than M=1)
M=3     1249.7 us    285.3 GB/s
M=4     1254.3 us    284.2 GB/s
M=8     1242.9 us    286.8 GB/s
M=16    1255.2 us    284.0 GB/s
M=32    1220.8 us    292.0 GB/s
```

The drop is a flat step at M=1→M=2 (not a gradual tile-fill effect: M=2 and
M=32 cost the same), consistent with a kernel-class switch, not tile waste.

For contrast, `quantized_matmul` (4-bit, same shape) has no such cliff — M=1
and M=2 are equal (~279 GB/s):

```
quantized_matmul 4-bit:  M=1 279.4 GB/s   M=2 279.1 GB/s   M=3 237.2   M=4 210.0   M=8 132.8
```

### Root cause (dispatch)

In `mlx/backend/metal/matmul.cpp`, the `Matmul` GPU path routes to `gemv` only
for `min(M, N) == 1`:

```cpp
// Route to gemv if needed
if (std::min(M, N) == 1) {
    return gemv(...);
}
// otherwise:
return steel_matmul(...);
```

A Metal GPU capture of a real M=2 forward confirms the dispatched kernels are
`steel_gemm_fused_nax_nt_bfloat16_bfloat16_bm64_bn128_bk256` and
`steel_gemm_splitk_…` — i.e. a 64-row output tile fed a 2-row input — with no
`gemv` present.

Note: naively looping `gemv` per row is *not* a fix — it re-reads the full
weight matrix M times (measured: 2× gemv = 1688 µs > steel M=2 = 1281 µs). A
real fix needs a small-M kernel that streams the weights **once** while
producing M output rows (a batched/small-M `gemv`, or a `steel_gemm` tile
specialized for tiny M), analogous to the quantized `qmm_t_splitk` added in
#3120.

### Real-world impact (motivation)

On a hybrid GatedDeltaNet + MTP model (Qwen3.5), at temperature-0 MTP
speculative decoding the verify forward is M=2. Measured end-to-end on the
M5 Max, at **matched per-forward weight bytes**:

| model | params | weights read / forward | decode (spec, tok/s) |
|---|---|---|---|
| 27B, 4-bit quantized | 27B | 14.4 GB | **~31** |
| 9B, unquantized bf16 | 9B  | 15.5 GB | **8.4** |

The unquantized 9B decodes ~3.8× slower than the 4-bit 27B *despite reading the
same bytes per forward and having one-third the parameters* — i.e. the
slowdown tracks the matmul **kernel path**, not model/weight size. The M=2
routing cliff above is one isolated, reproducible component; the remainder
appears to be the broader lack of decode-tuned non-quantized small-M kernels
(the quantized path benefits from `qmv` + #3120's split-K, the non-quantized
path does not).

### Ask

Add a small-M (≈2–8) kernel for the non-quantized path so batched/speculative
decode doesn't fall off the `gemv` cliff — mirroring the small-M attention the
quantized kernels already received.

### Related

- #3553 — `qmv` non-linear cost step at M=3 (the *quantized* analogue; shows
  M=1/2 are already well-handled there).
- #3120 — split-K for *quantized* small-M `qmm` (the tuning the non-quantized
  path is missing).
- #3196 — general bf16 GEMM vs PyTorch at large square shapes (different regime).
