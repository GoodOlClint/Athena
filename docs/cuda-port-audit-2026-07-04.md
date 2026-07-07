# Athena on CUDA unified-memory platforms — portability audit (2026-07-04)

**Question:** what would it take to run Athena on non-Apple unified-memory hardware — the GB10-class family (DGX Spark, ASUS Ascent GX10, Dell/HP/Lenovo/MSI variants), Jetson Thor/Orin, and GH200/GB200-class coherent-memory systems?
**Method:** two verified investigation passes — (1) an exhaustive Athena-side sweep of Apple platform assumptions at HEAD (`dad8ea1`), every finding file:line-cited; (2) MLX-stack CUDA readiness from local upstream clones (ground truth) cross-checked against web status. Raw reports preserved; sources at bottom.

## Verdict

**Feasible as a port, not a rewrite — and closer than expected.** The MLX stack is no longer Metal-only-plus-a-science-project: the C++ core ships **official CUDA wheels** (`pip install mlx[cuda13]`, demoed by the MLX lead on DGX Spark), and **mlx-swift 0.31.5+ builds on Linux with CUDA on by default, upstream, with a real-GPU CI job**. Crucially for the ADR 011 thesis, the CUDA allocator implements the **identical memory API the governor is built on** (`activeMemory`/`cacheMemory`/`cacheLimit`/`memoryLimit`/`clearCache` — all real on `CudaAllocator`), and GB10-class/Jetson hardware is genuinely one coherent pool, so "one Metal budget" translates to "one CUDA budget" without bending the thesis. MLX has **no other GPU backend** (verified: Metal + CUDA + CPU in-tree; zero ROCm/Vulkan/SYCL) — CUDA unified-memory is the only non-Apple target worth planning for, confirming the operator's read.

**The single unproven link:** `mlx-swift-lm` has never been built or run on Linux (no CI, one `canImport(Metal)` guard, nobody has decoded a token with it off-macOS). **The single biggest Athena-side cost:** the AVFoundation media-decode path (every audio/video byte entering the daemon) — the only true product-code rewrite. Everything else is lifecycle/logging/build/probe swaps with known Linux equivalents.

**Recommended first step: a 1–2 day spike on any GB10 box** — CMake-build mlx-swift + mlx-swift-lm on arm64/CUDA, load a quantized Gemma4, decode. That collapses most of the remaining uncertainty for the price of a weekend.

## Platform-family fit (the governor thesis, per platform)

| Platform | Memory shape | Thesis fit |
|---|---|---|
| **GB10-class** (Spark, GX10, Dell/HP/Lenovo/MSI — same superchip: 20-core Arm v9.2, 128 GB LPDDR5x, NVLink-C2C, sm_121, CUDA 13, DGX OS/Ubuntu 24.04 arm64) | One physical coherent pool; `concurrentManagedAccess` holds; MLX uses `cudaMallocManaged` so arrays are CPU-visible without a migration cliff; `cudaMemGetInfo` total ≈ the machine | **Best analog to Apple UMA — the thesis transfers.** Caveat from NVIDIA's own porting guide: plain `cudaMalloc` memory is NOT CPU-coherent on GB10; MLX's managed allocator sidesteps this. Note the class is a *capacity* box (~273 GB/s), not a speed box — expect Mac-Studio-class bandwidth, not HBM. |
| **Jetson Orin/Thor** | Integrated, physically unified, managed memory supported | Same story as GB10 for MLX/governor purposes. |
| **GH200/GB200** | **Two physical pools** (CPU LPDDR5x + GPU HBM) made coherent via NVLink-C2C/ATS; `cudaMemGetInfo` = HBM only | Works, but "one budget = the machine" does **not** transfer 1:1 — the governor's denominator becomes the GPU pool, and CPU-side coherent memory is a second tier the current design has no concept for. Out of scope for a first port; note for the ADR. |
| Discrete-GPU CUDA | No coherence; managed memory = page migration | Runs, but the UMA assumptions degrade to migration traffic. Not a target. |

## MLX-stack status (evidence in the research appendix)

| Component | Status |
|---|---|
| MLX core CUDA backend | Shipping/official; 60+ files; quantized matmul incl. **gather_qmm (the Gemma4 MoE op)**, SDPA (cuDNN path head_dim ≤128 → Gemma-family head_dim 256 falls to the correct-but-slower `sdpa_vector`; different perf profile than Metal steel-attention), rope/rms_norm/copy for KV ops, lazy eval/streams/JIT via nvrtc |
| MLX memory API on CUDA | **Full parity**: active/peak/cache/set_cache_limit/set_memory_limit/clear_cache on `CudaAllocator`; default budget 95% of `cudaMemGetInfo`; NVML for system-wide truth |
| mlx-c | Builds Linux+CUDA (vendored in mlx-swift CI); exposes `mlx_cuda_is_available` + full memory API |
| mlx-swift | **Linux+CUDA upstream since 0.31.5** (PR #413, 2026-06-18): CUDA default-on for Linux SPM builds, CMake path CI-tested on a real GPU (T4/x86_64). Athena's substrate pins 0.31.4 — one minor bump away. Metal-only Swift surface excluded on Linux; `MLX.Memory.*` cross-backend. **Gap: no arm64(sbsa)/sm_121 CI** — the GB10 build wiring is the untested permutation (core kernels proven on Spark via the Python wheels). |
| mlx-swift-lm | **Unproven on Linux** — the critical-path gap. Plausibly small patches (one `canImport(Metal)` guard; in-repo Foundation-only tokenizer; the two `MLXFast.metalKernel` users — SSM, Bitnet — are fallback-guarded and NOT on Athena's Gemma4/Qwen paths). Until it compiles and decodes on arm64+CUDA, the Swift route is unproven end-to-end. |
| Build-path caveat | The SPM CUDA build **excludes the `quantized/qmm/` kernel dir** (`Package.swift:106–115`); the CMake build compiles everything and is the CI-tested path. For a quantized-inference daemon: **assume CMake**, not bare `swift build`, until the SPM exclusions close. |
| Version pins | Core CMake blocks CUDA 13.1 (pin 12.9/13.0); "some ops not yet implemented" caveat from the maintainers stands. |

## Athena-side inventory (everything that isn't MLX)

**Baseline:** `Package.swift` declares `platforms: [.macOS(.v14)]`; porting discipline exists only in `clients/` (fully gated, already Linux/Windows) and 4 daemon files. ~40 daemon-side Apple touchpoints have zero conditional compilation — **but nearly all sit behind the ADR 008/009 MLX-free seams, which is the port's biggest asset**: the 796-test unit tier should pass on Linux nearly unmodified and becomes the free regression harness.

### Blockers (rewrite/replace)

| # | What | Where | Linux answer |
|---|---|---|---|
| B1 | **AVFoundation media decode** — all `/v1/audio/*` + `/v1/video/*` ingest (`AVAudioFile`/`AVAudioConverter`, the ADR 025 S5 in-memory `AVAssetReader` + `athena-mem://` resource-loader, video demux) | `AudioDecode.swift`, `InMemoryAsset.swift`, `VideoAudioTrack.swift` (~547 lines, clean PCM-out contract) | **libav FFI** (in-process, codec-complete, preserves the no-temp-file property via custom avio I/O) over an ffmpeg subprocess (adds a runtime dep + per-request process) or symphonia-in-rust-shim — note **ADR 025 explicitly rejected the symphonia decoder**; its "no cross-platform driver" premise is now false, so the ADR needs formal supersession either way, but its codec/parity/CVE-tax argument still favors libav. Only true product-code rewrite in the port (~1–2 wk). |
| B2 | **CoreImage vision decode** (`CIImage` in `ChatTurn.swift` + `MLXLLMModule.swift`; substrate's `UserInput.Image.ciImage` case is itself CoreImage-typed) | 2 call sites + a substrate seam | Portable image decoder (stb/libjpeg/libpng or rust-shim) + a **substrate change**: `UserInput.Image` needs a raw-pixel-buffer case. |
| B3 | **Unified-logging pipeline** — sink (`OSUnifiedLogHandler` → `os.Logger`) AND query (`/api/logs` + `athena logs` shell `/usr/bin/log`) | `AthenaLogging.swift` + 2 direct `os.Logger` users + 2 subprocess drivers in `+Admin.swift`/`Logs.swift` | journald: `sd_journal_send` LogHandler preserving the subsystem/category/req/principal fields; query re-shells to `journalctl -o json` (same ndjson-parse shape). swift-log means the sink swap is one file. |
| B4 | **launchd lifecycle** (install/start/stop/restart; ADR 037's sudoless restart = KeepAlive relaunch) | `LaunchdPlist.swift`, `Install/Uninstall/DaemonLifecycle/ConfigCmd` | systemd unit generation — ADR 037's static-plist slice made this *easy* (plist now carries only label/user/exec/`ATHENA_CONFIG`, 1:1 to `[Service]` keys); `KeepAlive` → `Restart=always`, so the exit(0)-restart design ports cleanly. POSIX chown/O_NOFOLLOW hardening ports as-is. |
| B5 | **ADR 024 T1 hardening** (Hardened Runtime, codesign, entitlements, task-port denial) | `deploy/build.sh`, `verify-hardening.sh`, `athena.entitlements` | No codesign analog — re-derive the posture: `prctl(PR_SET_DUMPABLE,0)`, Yama `ptrace_scope`, systemd sandboxing (`NoNewPrivileges`, `ProtectProc`, `MemoryDenyWriteExecute`). Cheap to build; the ADR 024 security *claims* must be re-argued for Linux. |
| B6 | **Build pipeline** (hard xcodebuild requirement, metallib bundling, artifact layout) | `deploy/build.sh`, `test.sh` | Inverts on Linux: plain `swift build`/CMake suffices, no metallib; scripts rewritten; `InstallPlan`'s libexec/metallib adjacency design becomes vestigial (delete, not port). |

### Port with semantic re-verification (small code, high proof burden — this is the thesis)

- **P1 · `phys_footprint` probe (ADR 023 G2):** `ProcessMemory.swift` uses mach `task_info(TASK_VM_INFO)`; feeds admission, `/healthz`, per-model deltas. Linux swap = `/proc/self/smaps_rollup` + NVML — but the **semantic** ("committed = footprint − reclaimable cache" counting GPU unified-memory the way phys_footprint does) must be re-proven empirically on GB10: do `cudaMallocManaged` pages show in RSS? If not, admission lies — the exact bug ADR 023 fixed. `governor_admission_mode=estimate` is the bring-up safety net; MLX's NVML free/total is the system-wide truth lever.
- **P2 · Metal-fault needles (ADR 030 P2):** latch/degrade machinery is MLX-free and ports byte-unchanged; the needle set (`metal::malloc`, "maximum allowed buffer size") needs CUDA members (`cudaErrorMemoryAllocation`, "out of memory"). Trivial — but if missed, the daemon silently reverts to abort-on-OOM.
- **P3 · Prefill ceiling (ADR 030 P1):** derived from Metal `maxBufferSize`; CUDA has no per-buffer max — feed device budget into the same unit-pinned formula (one line at the binding site).
- **P4 · URLSession off-Darwin:** ~10 daemon files use URLSession with no `FoundationNetworking` gating (only `clients/` is gated); mechanical — but the swift-huggingface fork exists because Darwin URLSession download-progress was broken, and corelibs URLSession is a different implementation: **pull progress must be re-proven on Linux**.
- **P5–P7:** `import Darwin` → `Glibc` alternation (11 files, pattern exists); `ConfigJSONEmit` CF-based float detection (the v0.10.226 mlx-lm compat fix) needs a corelibs-safe path + re-run tests; `ProcessHardening` non-Darwin branches are stubs — fill with `setrlimit`, `prctl(PR_SET_DUMPABLE,0)`, `explicit_bzero` (the current plain-`memset` fallback is optimizer-removable — the exact bug `memset_s` avoids).

### Contained swaps

SQLCipher crypto backend `CC → OPENSSL` (amalgamation is portable); PBKDF2 (CommonCrypto → ~20-line HMAC loop over swift-crypto, pin against current vectors); Keychain → keyfile/env (client already designed for it); power assertion → systemd-inhibit/nothing (verify the box doesn't idle-suspend — the M60 lesson); GPU telemetry AppleSiliconMetrics → NVML behind the same best-effort probe shape; WeSpeaker vDSP FFT → MLXFFT/pocketfft (re-pin numerics); egress-proxy session config (parse logic portable, injection differs); Doctor's FileVault/plist/log findings → LUKS/systemd/journal equivalents; e2e `say` → espeak-ng or committed WAV fixtures.

### Already portable (for free)

Hummingbird/NIO/NIOSSL/swift-log/swift-argument-parser (Linux-supported upstream); **swift-crypto was deliberately chosen over CryptoKit** "so AthenaCore stays buildable off Darwin" — T3 idle-KV + ADR 027 snapshot crypto ride unchanged; rust-shim fully portable; governor budget default (`physicalMemory * 0.75`) maps directly onto a 128 GB coherent pool; the whole `clients/` package already ships Linux/Windows.

## Critical path and sequencing

1. **Spike (1–2 days, needs a GB10 box):** CMake-build mlx-swift @ ≥0.31.5 + mlx-swift-lm on arm64/CUDA → load quantized Gemma4 → decode a token → check `MLX.Memory` numbers move. Kill/derisk criteria: arm64 Cmlx build wiring, mlx-swift-lm compile, gather_qmm on sm_121, tokenizer file paths.
   - **No purchase needed to run this spike — a GB10/DGX Spark is cloud-rentable at ~$0.60–0.65/hr** (Enverge, an NVIDIA-forum P2P listing, Server Room, Primcast — verified 2026-07-04), so the full arm64/`sm_121` cross-compilation spike costs **~$6–15**. Sequence it in two stages to spend even less: **Stage 0** — prove the `mlx-swift` → `mlx-swift-lm` → Athena stack *builds + decodes on Linux+CUDA at all* (the biggest unknown — mlx-swift-lm has never been built on Linux) on a cheap **x86+CUDA** box (RunPod RTX 4090 ~$0.34/hr); **Stage 1** — the GB10-specific bits (arm64/`sm_121`, `gather_qmm` on Blackwell, and the unified-memory governor-truthfulness probe, which **only** a coherent-pool GB10 can validate — discrete RunPod GPUs page-migrate) on a **rented DGX Spark**. Only buy a box (cheapest new: ASUS Ascent GX10 1TB ~$3,997) after Stage 1 passes. Runbook: session scratchpad `athena-cuda-spike-runpod.md`.
2. **Governor truthfulness bring-up (P1–P3)** — small diffs, run behind `governor_admission_mode=estimate` until the smaps/NVML probe is validated on-device.
3. **Daemonization + logging (B3, B4, B6, B5)** — wide but shallow; systemd + journald + a Linux build script; re-derive the T1 posture in an ADR 024 amendment.
4. **Media decode (B1, B2)** — the one real rewrite; libav behind the existing PCM seam + ADR 025 supersession; substrate `UserInput.Image` seam for vision. Can ship *after* an LLM/embeddings-only Linux daemon (the audio/video tenants are cleanly separable modules — a first port can simply not register them).
5. **Mechanical sweep + tests (P4–P7, swaps, e2e)** — the ADR 008/009 unit tier is the regression harness.

**Shape of the effort:** LLM/embeddings/governor core = a credible porting milestone (the thesis survives); full parity incl. audio/video = that plus the decode rewrite. The fallback (swap-engine: llama.cpp/vLLM/TensorRT-LLM behind Athena's module protocols) runs sooner but forfeits the governor — none of those engines exposes MLX-shaped one-pool accounting, so Athena would degrade to an API façade over someone else's allocator: per ADR 011, exactly the tier not worth building. Bridge if the spike fails; not the plan.

## Sources

Local ground truth: `~/Source/mlx/mlx` @ `b410f6c` (`backend/cuda/` — `allocator.cpp`, `quantized/quantized.cpp`, `scaled_dot_product_attention.cpp`, `device_info.cpp`), `~/Source/mlx/mlx-swift` @ `2b33d85` (Package.swift Linux/CUDA branch, PR #413 `e23ae6b`, GPU CI workflow), `~/Source/mlx/mlx-swift-lm` @ `5f17df4`, `~/Source/mlx/research/mlx-mem-research/FINDINGS.md`, Athena Sources @ `dad8ea1` (all file:line cites above).
Web: [MLX lead demoing MLX on DGX Spark](https://x.com/awnihannun/status/2011315884552585239) · [mlx-swift #258 (Linux/Windows CUDA request)](https://github.com/ml-explore/mlx-swift/issues/258) · [MLX-on-CUDA discussion #2422](https://github.com/ml-explore/mlx/discussions/2422) · [MLX install docs](https://ml-explore.github.io/mlx/build/html/install.html) · [DGX Spark porting guide (coherence caveats)](https://docs.nvidia.com/dgx/dgx-spark-porting-guide/porting/cuda.html) · [NVIDIA DGX Spark announcement](https://nvidianews.nvidia.com/news/nvidia-announces-dgx-spark-and-dgx-station-personal-ai-computers) · [ASUS Ascent GX10](https://www.asus.com/networking-iot-servers/desktop-ai-supercomputer/ultra-small-ai-supercomputers/asus-ascent-gx10/) · [ServeTheHome GX10 review](https://www.servethehome.com/asus-ascent-gx10-review-a-new-nvidia-gb10-solution/) · [Tom's Hardware DGX Spark review](https://www.tomshardware.com/pc-components/gpus/nvidia-dgx-spark-review/2) · [Simon Willison on DGX Spark](https://simonw.substack.com/p/nvidia-dgx-spark-great-hardware-early) · [EXO: Spark + Mac Studio](https://blog.exolabs.net/nvidia-dgx-spark/)
