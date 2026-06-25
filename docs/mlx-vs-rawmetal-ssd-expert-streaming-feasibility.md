# MLX-layer vs. raw-Metal: can the downstream client's SSD expert streaming be brought into MLX? — research note

**Status:** Research only. No decision, no chosen architecture, no implementation, no code.
This note is the citation-level backing for the high-level disposition reached in
`the downstream client-ssd-streaming-and-kv-snapshot-research.md` §3a.3 ("SSD expert streaming — CONFIRMED
larger lift, the downstream client is not MLX"). Where that note asserted the conclusion, this one reads the
actual source on both sides and shows the work, so a forthcoming ADR rests on ground truth.

**The one question.** Can Apple-Silicon MoE **SSD expert streaming** — keep non-routed weights
resident, stream the long tail of routed experts from disk on demand behind an in-RAM LRU
expert cache — be implemented **at the MLX layer** (additive, upstream-able, Athena reaches it
through its substrate), or does it intrinsically require **raw Metal** (a substrate replacement)?
the downstream client proves the *hardware* can do it. This note decides what it costs to get the *capability* into
MLX, and how big/upstream-able that change is.

**Method discipline.** Fact-in-code (`file:line` + commit/version) is separated from inference
(mine). Assumptions and gaps are marked. No code is proposed.

**Sources read directly:**
- **the downstream client** — `github.com/antirez/the downstream client` @ commit `80ebbc396aee40eedc1d829222f3362d10fa4c6c`
  (2026-06-17). C + Objective-C Metal; files `the downstream client.c`, `ds4_metal.m`, `ds4_ssd.c`, `the downstream client.h`,
  `ds4_gpu.h`, `ds4_cli.c`, `metal/moe.metal`, `ds4_streaming_hotlist.inc`.
- **MLX** — `ml-explore/mlx-swift` **v0.31.3** rev `61b9e011e09a62b489f6bd647958f1555bdf2896`
  (the pin in `Athena/Package.resolved`), bundled C++ core under
  `Source/Cmlx/mlx/mlx`, Swift bindings under `Source/MLX`, C shim under `Source/Cmlx/mlx-c`.
- **Model layer** — `mlx-swift-lm` `Libraries/MLXLMCommon/SwitchLayers.swift` (SwitchGLU).
- **Athena consumers** — `AthenaCore/InferenceModule.swift`, plus the §2.1 facts already cited
  in the companion note (`AthenaQwen35.swift`, `MLXLLMModule.swift`, `MemoryGovernor.swift`).

---

## TL;DR verdict (full reasoning in §C.3)

the downstream client's streaming is **a host-side residency manager wrapped around ordinary quant-matvec Metal
kernels** — the kernel never faults; C/Obj-C host code guarantees every selected expert is
resident (via `pread` into Metal buffers) *before* it dispatches. The kernels carry **no disk
logic**. That is the encouraging half: a port needs a *residency manager*, not a novel kernel.

But MLX v0.31.3 lacks **every** primitive that manager stands on:
1. no mmap / re-faultable file-backed array (loads `pread` the whole tensor into an owned buffer,
   one-shot, `Load::eval_gpu` is unimplemented);
2. no per-expert addressability — `gatherQMM`/`gatherMM` take the **entire** stacked
   `[E, …]` expert tensor as one resident input and index it at kernel time;
3. no per-buffer residency control — only one **process-global** wired-byte budget;
4. no paging/spill/prefetch notion anywhere in-tree.

**Verdict: (ii) reachable only by forking/extending mlx core (new C++ allocator/IO machinery +
a partially-resident gather path), NOT as a small additive contribution, and NOT intrinsically
raw-Metal.** the downstream client's *approach* (host residency manager + plain kernels) *is* expressible in MLX's
architecture in principle, so it is not case (iii). But getting there requires substantial,
mostly-non-upstream-able core changes — much closer to (ii) than (i). The governor reframing
(§3.4 of the companion note) remains the durable prize and is orthogonal to the mechanism.

---

## A. the downstream client — HOW it streams (fact-in-code, the downstream client @ 80ebbc39)

### A.1 Resident vs. streamed; cache structure, slots, LRU, mlock, hotlist

**Resident (non-routed), measured for budgeting** by `weights_streaming_non_routed_bytes` →
`weights_model_map_decode_static_spans(...)` (`the downstream client.c:4403-4416`). The per-layer static set
(`model_map_span_vec_include_layer_decode_static`, `the downstream client.c:4229-4264`) is explicitly: attention
(`attn_q_a/b`, `attn_kv`, norms, sinks, `attn_output_a/b`, compressors), the indexer tensors, the
router (`ffn_gate_inp`, `ffn_exp_probs_b`, `ffn_gate_tid2eid`), and the **shared expert**
(`ffn_*_shexp`). Comment at `the downstream client.c:4266-4274`: "The static set excludes routed expert tensors
because the streaming expert cache serves them."

**Streamed (routed):** `ffn_gate_exps/up_exps/down_exps` are added to the resident set **only when
experts are non-uniform** (`if (!weights_streaming_layer_experts_uniform(...))`,
`the downstream client.c:4275-4286`); in the uniform streaming case they are excluded and served by the cache.

**Expert-cache data structure:** `struct ds4_gpu_stream_expert_cache_entry` at
`ds4_metal.m:392-413` — per entry: `gate_buffer/up_buffer/down_buffer` (Metal buffers),
`gate_abs_offset/up_abs_offset/down_abs_offset` (on-disk byte offsets), `gate_expert_bytes/
down_expert_bytes`, `last_used`, `use_count`, `inflight_seq`, `slab_slot`, `valid`, `slab_backed`.

**Slot model:** a 2-D table `g_stream_expert_cache[MAX_LAYER=61][MAX_EXPERT=384]`
(`ds4_metal.m:424-425`; constants `:369-378`). Backing is **slabs** of fixed-size slots —
`g_stream_expert_cache_slabs[256]`, `g_stream_expert_cache_slab_slot_bytes`, free-slot stack
`g_stream_expert_cache_free_slots[]` (`:444-453`); an entry's `slab_slot` indexes the pool when
`slab_backed` (`install_loaded`, `:10277-10288`).

**LRU / eviction = hotness-then-LRU.** `ds4_gpu_stream_expert_cache_take_reusable`
(`ds4_metal.m:9740-9836`), invoked when `entry_count >= budget` or `force_reuse`, scans all
`[layer][expert]` and evicts the victim with **lowest route-hotness**
(`g_stream_expert_cache_route_hotness`), tie-broken by **oldest `last_used`** (`:9794-9802`). It
skips in-flight entries (`:9783-9786`) and currently-selected/protected experts (`:9787-9793`),
waiting on in-flight if no victim is free (`:9806-9815`). On a hit: `last_used = ++clock;
use_count++` (`:10341-10343`); clock is `g_stream_expert_cache_clock` (`:203`, bumped `:10271`).

**mlock policy = degrade, don't die.** Cache buffers are `mlock`'d in
`ds4_gpu_stream_expert_alloc_buffer` (`:8229-8253`) and per slab-slot (`:8381-8406`). On mlock
**failure it does NOT abort**: it counts `mlock_failures`/`mlock_fail_bytes`, warns once
(`ds4_gpu_stream_expert_cache_warn_mlock_failure`, `:8108-8160`), and **caps the cache budget to
the number of slots it could actually lock** (`g_stream_expert_cache_mlock_budget_cap`, set
`:8200-8202`, applied in `configured_budget` `:7543-7545`). (The companion note's "refuses to
install pageable entries" is the *spirit*; the *code* caps the budget rather than refusing
outright.) Note: the `mlock` in `ds4_ssd.c:108-173` is the unrelated `--simulate-used-memory`
test knob, not the expert cache.

**Profiled-hotlist warm-start.** `ds4_streaming_hotlist.inc` is a compiled static array of
`{layer, expert}` `uint16_t` pairs sorted by hit count (`:1-3`; included at `the downstream client.c:829`), in two
variants (`_pro`/`_flash`) chosen by model in `metal_graph_streaming_expert_hotlist_load_default`
(`the downstream client.c:3961-3970`). The list feeds `ds4_gpu_stream_expert_cache_seed_experts`
(`ds4_metal.m:11423`), which records route-hotness (`note_route_hotness`, `:11495-11498`) and
**synchronously preads** the popular experts into the cache at startup. Disabled by
`--ssd-streaming-cold` and `DS4_METAL_DISABLE_STREAMING_EXPERT_HOTLIST` (`the downstream client.c:13834-13839`);
preload count capped (auto default 4096) by `metal_graph_streaming_expert_preload_count`
(`the downstream client.c:13991-14018`). `ds4_gpu.h:91-92` notes the resident cache is "intentionally kept warm
across sessions."

### A.2 Cache-miss dispatch: addressing, blocking vs. prefetch, quant layout

**On-disk addressing = offset arithmetic, NOT mmap.** An expert is
`tensor_base_offset + expert_id * per_expert_stride`. See `begin_selected_load`
(`ds4_metal.m:10706-10716`): `gate_rel = expert_id * gate_expert_bytes; gate_abs_offsets[i] =
gate_offset + gate_rel; …`. Per-expert stride is `tensor_bytes / DS4_N_EXPERT`
(`the downstream client.c:14053-14054`).

**Load = `pread` into a Metal buffer.** The synchronous miss handler
`ds4_gpu_stream_expert_cache_get_protected` (`ds4_metal.m:10305+`): (1) `F_RDADVISE` readahead
hint (`readahead_range`, `:10348-10350`); (2) allocate/evict a slot (`prepare_load_buffers`,
`:10367`); (3) `pread` gate/up/down directly into `[buf contents]+inner` (`:10385-10411`); (4)
`didModifyRange` to sync to GPU (`:10414-10416`); (5) `install_loaded` (`:10426`). Underlying
syscall: `pread(g_model_fd, dst+pos, want, offset+pos)` in `ds4_gpu_stream_expert_pread_into`
(`:7800`); `g_model_fd` is the open GGUF file. **No `O_DIRECT`/`F_NOCACHE`** — only `F_RDADVISE`
(`:7677`); it relies on the OS page cache + readahead.

**Blocking vs. prefetch — overlap-then-block, not true double-buffer.**
- Prefetch is *started* at routing time: `ds4_gpu_stream_expert_cache_begin_selected_load`
  (`:10618`) is called right after the CPU router picks experts, before the matvec
  (`the downstream client.c:14061-14066`), recording a pending load (`g_stream_expert_pending_load`, `:7697-7725`).
- Reads are parallelized across a pread thread pool (up to ~18 workers; default 9, env
  `DS4_METAL_STREAMING_EXPERT_PREAD_THREADS`; `:7846-8042`, `:7754-7767`).
- But the GPU dispatch **blocks on residency**: before encoding the decode matvec the host calls
  `ds4_gpu_end_commands()` to flush in-flight GPU work (`:24978`), then `prepare_selected_batch`
  (`:24982`) finishes the pending load / synchronously preads any still-missing experts, builds
  the address buffers, and only then dispatches (`:24976-25029`). **Inference (mine):** this is
  "prefetch overlapped with router compute, then block if not ready" — not an async double-buffer
  where the kernel runs on stale data.

**Quant layout = consumed raw, dequantized in-kernel.** The `pread` copies quant blocks verbatim;
no dequant on load. Block structs `block_q2_K`, `block_q4_K`, `block_iq2_xxs` at
`metal/moe.metal:94-111`. The the downstream client default pairs **IQ2_XXS** gate/up with the fused swiglu kernel
`kernel_mul_mv_id_iq2_xxs_pair_swiglu_f32` (`metal/moe.metal:1022`) and **Q2_K** down (header
`the downstream client.c:339-341` lists "Q2_K routed down experts … IQ2_XXS routed gate/up experts"; Q4_K is the
high-memory variant). In-kernel dequant: `dequantize_q2_K` (`moe.metal:234`),
`dequantize_iq2_xxs` (`moe.metal:278`).

### A.3 Where the streaming logic lives (the decisive question)

**Entirely in C/Obj-C HOST code. The Metal kernel has zero disk/fault/streaming awareness.**

- `metal/moe.metal` contains no `pread`/`mmap`/`open`/`fault`/`madvise`/`F_NOCACHE` — only
  `expert_slot` loop vars and `n_slots`/`expert_bytes` struct fields.
- The address-table matvec `kernel_mul_mv_addr_iq2_xxs_pair_swiglu_masked_f32`
  (`metal/moe.metal:1347-1439`) takes `device const uint64_t *gate_addrs/up_addrs`; if an address
  is **0 (missing) it simply `return`s** (`:1376-1380`) — it does not fault or load. It
  `reinterpret_cast`s the host-provided 64-bit pointer to a device buffer and runs the dot product
  (`:1385-1412`).
- `kernel_stream_expert_cache_validate` (`metal/moe.metal:1441-1476`) only **reports** a
  miss/invalid mask back to host in a status buffer; it does not resolve misses.
- Host guarantees residency before dispatch: `..._get_protected` (miss → pread → install) and
  `prepare_selected_batch` (pre-dispatch at `:24982`, after the `end_commands` flush at `:24978`).

**Inference (mine), and the most important finding for the port:** because residency is a
*host-side* concern and the kernels are ordinary quant-matvec/gather kernels reading device
buffers (or raw device addresses), **a port needs a residency manager around existing kernels, not
a new fault-handling kernel.** the downstream client's two small kernel concessions (tolerate a NULL/0 expert
address by returning; report a miss mask) are the *only* kernel-level streaming awareness, and
both are trivial — the heavy machinery is all host C.

### A.4 Budget math

"80% of recommended working set − non-routed" lives in `ds4_ssd_auto_cache_plan`
(`ds4_ssd.c:80-106`): `model_target_bytes = recommended_bytes * 4 / 5` (`:89-91`);
`cache_bytes = model_target_bytes − non_routed_bytes` (`:92-94`);
`cache_experts = cache_bytes / per_expert_bytes` (floor 1, capped at `max_model_experts`,
`:96-104`). Inputs gathered in `ds4_engine_configure_streaming_auto_cache`
(`the downstream client.c:25380-25456`): `recommended = ds4_gpu_recommended_working_set_size()` (`:25396`) →
`[g_device recommendedMaxWorkingSetSize]` (`ds4_metal.m:2970-2974`);
`non_routed_bytes = weights_streaming_non_routed_bytes(...)` (`:25405`);
`per_expert_bytes = ds4_streaming_routed_expert_bytes(...)` (`:25412`). The `--ssd-streaming-cache-experts`
knob is parsed in `ds4_cli.c:1486-1492` (integer → expert count; `NGB` suffix → byte budget) and
pushed via `ds4_gpu_set_streaming_expert_cache_budget` (`the downstream client.c:25731`; setter `ds4_metal.m:2951`).

---

## B. MLX — what it CAN express today (fact-in-code, MLX v0.31.3)

### B.1 Weight materialization & mmap

**No mmap in the loader path.** A grep for `mmap|memory_map|MmapReader|load_lazily` across the
C++ core hits only CUDA `dlopen(RTLD_LAZY)` — nothing in `io/`. The file reader is
`ParallelFileReader`, an `int fd_` read via `pread` in 32 MB chunks on a 4-thread pool
(`Source/Cmlx/mlx/mlx/io/load.h:59-107`; read impls `io/load.cpp:347-393`).

**Loading is graph-lazy, then fully copied on eval.** `load_safetensors` parses only the JSON
header and builds, per tensor, an `array(...)` whose primitive is a `Load` holding reader + byte
offset (`Source/Cmlx/mlx/mlx/io/safetensors.cpp:150-158`; same for `.npy` at
`io/load.cpp:318-322`). The byte copy happens on eval in `Load::eval_cpu`:
`out.set_data(allocator::malloc(out.nbytes()))` then `reader->read(out_ptr, …, offset)`
(`backend/common/load.cpp:30-55`) — **the whole tensor is `malloc`'d into an owned buffer and the
full byte range `pread` into it.** `Load::eval_gpu` is **unimplemented** —
`backend/metal/primitives.cpp:155-156` throws `"[Load::eval_gpu] Not implemented."` (loads run on
the `.cpu` stream; the Swift `loadArrays`/`loadArray` default `stream: .cpu`,
`Source/MLX/IO.swift:97,125`).

**No re-faultable file-backed array.** Array storage is a `std::shared_ptr<Data>` holding an
`allocator::Buffer` (`array.h:484-486`, `set_data` `:439-448`). `Load` is a one-shot primitive;
once eval'd the array no longer references the file offset, and **no API drops a materialized
buffer and re-faults it from disk** (mine: a "reload" means re-issuing `mlx_load_safetensors` and
re-eval). The Swift `loadArrays(data:)` in-memory variant copies the whole file into a Swift
`Data` first (`IO.swift:199-261,308-322`) — strictly worse for residency.

### B.2 SwitchGLU / MoE gather path

**SwitchGLU requires the full stacked `[numExperts, …]` tensor resident.** In
`SwitchLayers.swift`: `SwitchLinear.weight` is one array of shape `[numExperts, outputDims,
inputDims]` (`:166-170`); non-quant path calls `MLX.gatherMM(x, weightT, rhsIndices: indices, …)`
with the whole `weight` (`:198-199`); quant path calls `MLX.gatherQuantizedMM(x, self.weight,
scales:…, biases:…, rhsIndices: indices, …)` with the whole packed tensor (`:244-255`).
`SwitchGLU.callAsFunction` (`:62-86`) routes through these — experts are picked by **`indices`
into the resident stacked tensor**, never by passing a subset (mine: confirmed reading the file;
`indices` flows straight into `upProj/gateProj/downProj` at `:74-79`).

**The primitive takes the whole expert tensor as one input.** `GatherQMM::eval_gpu` derives expert
count from that single tensor: `int E = w.size() / w.shape(-1) / w.shape(-2);`
(`backend/metal/quantized.cpp:1355-1376`), with `rhs_indices` selecting the slice. Op signatures
confirm `w` is one array, not a list: `gather_qmm(x, w, scales, biases, lhs_indices, rhs_indices,
…)` (`ops.h:1440-1452`), `gather_mm(a, b, lhs_indices, rhs_indices, …)` (`ops.h:1494-1500`).
Primitives `GatherMM` (`primitives.h:524-548`) / `GatherQMM` (`:1676-1710`) carry only
sort/group/bits flags — **no per-expert addressability; the expert dimension lives inside one
input array indexed at kernel time.** Backing kernels: `affine_gather_qmm_*_nax`
(`backend/metal/kernels/quantized_nax.h:1340,1408`), FP variants `fp_gather_qmm_*_nax`
(`kernels/fp_quantized_nax.h:681,746,814`).

**Inference (mine):** to feed only routed experts you would have to pre-gather/slice `weight` into
a smaller contiguous tensor yourself (extra op + copy) and pass that — nothing in the MoE path
does it, and that copy defeats the point (it materializes from the already-resident big tensor).

### B.3 Memory / residency APIs

**Swift surface (`Source/MLX/Memory.swift`, `WiredMemory.swift`):** `Memory.cacheLimit`
(`Memory.swift:251-276`), `memoryLimit` (`:290-305`), `clearCache()` (`:356-360`), and the
`activeMemory/cacheMemory/peakMemory/snapshot()` counters (`:175-225`) are **all global, pool-wide
— none is per-buffer.** `withWiredLimit` sync overload is deprecated/no-op (`:315-324`); async
delegates to `WiredMemoryManager` (`:340-350`). The wired machinery (`WiredMemoryManager` actor,
`WiredMemoryTicket`, `WiredSumPolicy`/`WiredMaxPolicy`, `WiredMemory.swift:340,245,152,166`) is an
**admission/ticket layer over one global scalar** — it ultimately calls
`mlx_set_wired_limit(&previous, size_t(limit))` (`WiredMemory.swift:23-31`).

**C/C++ underpinning:** `mlx_set_wired_limit` → `set_wired_limit` (`mlx-c/mlx/c/memory.cpp:83-86`)
→ C++ `set_wired_limit` (rejects > recommended working set; `backend/metal/allocator.cpp:254-262`,
decl `memory.h:66-78`). There **is** an `MTL::ResidencySet` (`backend/metal/resident.cpp:7-26`,
gated on GPUFamilyMetal3 + macOS 15), and the allocator owns exactly one (`allocator.h:62`,
`wired_limit_` `:70`). But `set_wired_limit` just `residency_set_.resize(wired_limit_)`
(`allocator.cpp:97-102`); every malloc'd non-heap buffer is **automatically** inserted
(`allocator.cpp:161,214`) and wired only if it fits under `capacity_`, else parked in an
`unwired_set_` (`resident.cpp:28-38`); `resize` adds/removes in "whatever MTL returns" order, **not
caller-selectable** (`resident.cpp:52-91`). `Device::set_residency_set` is set once
(`device.cpp:801-813`).

**Net:** the only public granularity is a **single global wired-byte budget**. No
`setPurgeableState`, no per-array `makeResident`/pin/evict is exposed at Swift or C++ surface.

### B.4 Any paging / streaming / faultable notion

**Absent.** Grep across both checkouts for
`spill|paged|page_in|page_out|faultable|swap.?out|offload|evict|streaming weight|prefetch`
returns no weight-streaming hits (the lone `spill` is bit-packing in
`TurboQuant/TurboQuantSupport.swift:309-339`, unrelated). What exists is buffer **recycling** —
the `BufferCache` pool reuses *freed* intermediates and can release them under pressure
(`Memory.swift:6-60` docs; `allocator.h:60`; `allocator.cpp:134-137,169-173`) — not fault-in/
spill-out of live weights. **Gap, needs a web check:** whether upstream `ml-explore/mlx` *post*-0.31.3
or its issues/PRs discuss streamed/paged weights is not determinable from a pinned offline
checkout; in-tree at v0.31.3 it is absent.

---

## C. Gap analysis — what MLX would NEED

### C.1 The missing primitives (precise enumeration)

the downstream client's host residency manager (§A.3) stands on four legs. MLX has none of them (§B):

| # | Missing primitive | the downstream client has it as | MLX today |
|---|---|---|---|
| (a) | **mmap/faultable array backing** — a weight array whose bytes can be loaded on demand and dropped/reloaded without rebuilding the graph | `pread` offset-addressed loads into Metal buffers (`ds4_metal.m:7800,10385-10411`) | `Load` is one-shot, full copy, `eval_gpu` unimplemented; no re-fault (B.1) |
| (b) | **per-expert residency manager (LRU + slabs + mlock)** | `g_stream_expert_cache[...]`, take_reusable LRU, slab pool, mlock-cap (`ds4_metal.m:392-453,9740-9836,8200-8202`) | only one **global** wired-byte budget; no per-buffer pin/evict (B.3) |
| (c) | **partially-resident gather** — a gather matvec that tolerates an expert table where only the routed/resident slice is present (addressed by slot/pointer, NULL ⇒ skip) | `kernel_mul_mv_addr_..._masked` reads `uint64_t` addrs, returns on 0 (`moe.metal:1347-1439`) | `gatherQMM`/`gatherMM` need the **whole** `[E,…]` tensor resident, indexed by `rhs_indices` (B.2) |
| (d) | **prefetch / async load to hide SSD latency** | begin_selected_load + pread thread pool + block-before-dispatch (`ds4_metal.m:10618,7846-8042,24978-25029`) | absent; no prefetch/spill (B.4) |

### C.2 For each: which layer, upstream-able?, size/risk

- **(a) Faultable/mmap array backing — mlx core C++ (`io/` + allocator + a new primitive).**
  Needs an mmap-backed (or re-faultable `pread`) buffer that the allocator can drop and the graph
  can re-source, plus a working `Load::eval_gpu` or an equivalent GPU-visible mapped buffer.
  *Upstream-able?* **Partially** — a generic "mmap-backed load" could interest upstream, but a
  *re-faultable, evict-and-reload* buffer cuts against MLX's value-semantics array model (an
  array's data is an owned `shared_ptr<Data>`, B.1) and is likely **fork-first**. *Size/risk:*
  **Large / high.** Touches the allocator, the buffer lifecycle, and the unified-memory model.
- **(b) Per-expert residency manager (LRU + slabs + mlock).** Two homes are possible: a **mlx core**
  allocator extension exposing per-buffer pin/evict, or — *if* (a)+(c) existed — a **model-layer /
  Swift-side** manager owning the LRU/hotlist/mlock policy above MLX (this is the governor-shaped,
  MLX-free-testable part, and the slice Athena would naturally own, cf. companion §3.3B).
  *Upstream-able?* The policy layer is Athena's; the **per-buffer residency hook in the allocator**
  is the upstream-able-but-invasive piece (MLX deliberately exposes only a global wired budget,
  B.3). *Size/risk:* **Medium** for the policy layer (well-understood, the downstream client is a blueprint),
  **Medium-High** for the allocator hook.
- **(c) Partially-resident gather kernel + primitive.** Needs a `GatherQMM`/`GatherMM` variant
  whose expert input is an **address/slot table** (à la the downstream client's masked addr kernel) rather than a
  single dense `[E,…]` tensor, plus the Metal kernel to back it. *Upstream-able?* **Unlikely as-is**
  — it changes the gather op's input contract; upstream would want a strong general use case.
  *Size/risk:* **Medium** kernel work (the downstream client's `moe.metal` masked-addr kernels show it is small and
  the disk-awareness stays in host code, §A.3) **+ Medium** primitive/op plumbing in core.
- **(d) Prefetch / async load.** Sits above (a): a Swift/host scheduler that issues loads at
  routing time and synchronizes before dispatch. *Upstream-able?* The scheduling is Athena/model-
  layer; it depends entirely on (a) existing. *Size/risk:* **Medium**, but **fully gated on (a).**

**Convert-time dependency (mine):** even with (a)–(d), the current weights are stored as a
pre-stacked `[num_experts,…]` tensor (companion §2.1; SwitchLinear `:166-170`). Per-expert fault
wants **per-expert-addressable on-disk layout** (offset table, like the downstream client's GGUF stride math
`the downstream client.c:14053-14054`). That is an `athena convert`/format change — additive but real.

### C.3 The verdict (C.3 (i)/(ii)/(iii))

**Verdict: (ii) — reachable only by forking/extending mlx core (new C++ IO + allocator
machinery and a partially-resident gather path), with a thin additive model-layer/Athena policy
cap on top. It is NOT (i) a small additive MLX contribution, and it is NOT (iii) intrinsically
raw-Metal.**

Reasoning:
- **Not (iii).** the downstream client disproves the "must be raw Metal" fear *at the architecture level*: its
  streaming is a **host-side residency manager around ordinary quant-matvec kernels** (§A.3); the
  kernels carry no disk logic and need only tolerate a missing-expert address. MLX's model —
  host/Swift code orchestrating Metal kernels over unified-memory buffers — is the *same shape*.
  Nothing about expert streaming intrinsically requires abandoning MLX's array/graph engine; the
  faulting lives in host code either way. So this is expressible within MLX's architecture.
- **Not (i).** But MLX v0.31.3 lacks **all four** primitives (§C.1), and the two hardest — (a)
  re-faultable buffer backing and (c) a partially-resident gather contract — cut against MLX's
  core design choices (owned value-semantic array data; global-only residency; whole-tensor gather
  input). These are not "add a function" contributions; they are **core surgery** with low upstream
  acceptance odds. A PR that makes arrays evict-and-re-fault is a change to MLX's memory model, not
  a feature flag.
- **Therefore (ii).** The realistic path is a **fork/extension of mlx core** (a)+(c) [+ allocator
  hook for (b)] **plus an additive model-layer/Athena policy cap** (b-policy)+(d). The Athena-owned
  slice (LRU/hotlist/mlock policy, governor accounting) is genuinely additive and the downstream client-blueprinted;
  the blocking slice is core-fork. This is squarely "fork mlx core / extend kernels," i.e. (ii) —
  heavier than an upstream contribution, lighter than a from-scratch raw-Metal engine.

**Sizing summary (what an attempt entails):**
- *Upstream MLX contribution path (additive):* realistically only (b)-policy and (d)-scheduling are
  cleanly additive, and both are **gated on (a) + (c)** which are not. So a purely-additive MLX
  contribution **cannot deliver the capability** on its own. **Verdict-relevant:** the additive-only
  door is effectively closed without core changes.
- *mlx-core fork path:* (a) faultable backing [Large/High], (c) gather contract + masked kernel
  [Medium+Medium], (b) allocator per-buffer hook [Medium-High], then (b)-policy + (d) above it
  [Medium, MLX-free-testable], plus an `athena convert` per-expert layout change [Medium].
- *raw-Metal alternative (the downstream client's actual route):* a from-scratch engine + custom kernels + GGUF
  loader — **substrate replacement**, antithetical to Athena's pristine-substrate/additive-port
  discipline. Strictly larger than the fork path and discards MLX entirely. Not recommended; listed
  only as the lower bound on "what definitely works on this hardware."

This **confirms and sharpens** companion-note §3a.3: "larger lift" is precisely *core-fork-sized*,
and the reason is enumerable (the four missing primitives, two of which fight MLX's design), not
hand-waved.

---

## D. Consumer impact (pointer only — detailed design is a later ADR)

If MLX gained streaming, Athena's seam would have to stop modeling a slot as **one scalar**. Today
`InferenceModule` exposes only `residentBytes` and `memoryEstimate() -> Int`
(`AthenaCore/InferenceModule.swift:22,26`), and admission is `max(committed, reserved)` vs one
budget number (companion §2.1, `MemoryGovernor.swift:749-774`). To express "X bytes pinned-resident
+ Y bytes streamable," the protocol would need a **partial-residency footprint** — minimally a
`(pinnedResidentBytes, streamableBytes, cacheBudgetBytes)` triple (or a small struct) replacing the
single scalar — and the governor admission math would arbitrate **how deeply** each tenant is
resident (the elastic "speed-dial," companion §3.4) rather than binary fit/no-fit. This is the
already-parked open question (companion §5.6 / §3a.3): *cheap to add the shape now, expensive to
retrofit later.* No design here — flagged for the ADR.

---

## E. Open questions (each tagged with what it needs)

1. **Upstream MLX roadmap:** does `ml-explore/mlx` *post*-0.31.3 or any open issue/PR add
   mmap/faultable backing, per-buffer residency, or partially-resident gather? — *Needs:* a GitHub
   issues/PR + release-notes review (offline checkout can't answer; B.4). **Gates the whole §C.3
   sizing** — if upstream is already moving on (a)/(c), the fork shrinks toward additive.
2. **Re-faultable buffer vs. MLX value semantics:** can a buffer be dropped and re-sourced without
   breaking MLX's owned-`shared_ptr<Data>` array model (B.1)? — *Needs:* a core-MLX spike + a
   maintainer question; this is the make-or-break feasibility item for primitive (a).
3. **Masked/address gather contract:** would upstream accept a gather variant taking an
   address/slot table instead of a dense `[E,…]` input, or is it fork-only? — *Needs:* an upstream
   maintainer question; determines whether (c) is a contribution or a fork.
4. **Convert-time per-expert layout:** cost/shape of an `athena convert` change to write
   per-expert-addressable weights (offset table) vs. today's pre-stacked tensor. — *Needs:* a
   convert-path spike (cf. ADR 012/016 quant-rule work).
5. **True async double-buffer vs. the downstream client's overlap-then-block:** the downstream client blocks before dispatch (§A.2);
   is decode-latency on Athena's targets acceptable under block-on-miss, or is real
   double-buffering needed (and does that change the kernel contract)? — *Needs:* a latency model
   once (a)–(c) exist; do not pre-optimize.
6. **`InferenceModule` partial-residency shape (parked):** adopt the
   `(pinned, streamable, cacheBudget)` footprint now to keep the governor speed-dial expressible,
   even before the mechanism? — *Needs:* a protocol-design decision in the ADR (companion §5.6;
   cheap now, expensive to retrofit).

---

## F. Assumptions & scope made explicit

- the downstream client facts read at commit `80ebbc39`; MLX facts at v0.31.3 rev `61b9e011`; both assumed stable as
  of this note. Athena/model-layer facts cited `file:line`.
- I scoped the downstream client to the **Metal** path (`ds4_metal.m` / `metal/moe.metal`); the CUDA/ROCm streaming
  paths (`ds4_gpu.h:106-123`, `ds4_cuda.cu`) differ and were not analyzed — irrelevant to the
  Apple-Silicon/MLX question.
- I did not line-by-line every one of the downstream client's ~30 `kernel_mul_mv_slots6_*` variants
  (`moe.metal:2017-3950`); I confirmed the residency-in-host pattern on the representative
  gather/addr/validate/swiglu kernels and a full grep sweep (gap noted in §A.3 research).
- "Upstream-able?" judgments are **inference** about likely maintainer acceptance, not facts;
  open question E.1/E.3 convert them to evidence.
- This note amends no ADR. Any pursuit goes through the brownfield change gate (CLAUDE.md) and must
  reconcile with ADR 011 (governor thesis) and the companion note's §3a resolutions.
- **No implementation, no code changes** were made or proposed.
