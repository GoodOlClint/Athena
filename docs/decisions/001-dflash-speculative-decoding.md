# 001 — DFlash speculative decoding for non-MTP targets (Gemma4-first)

**Status:** Accepted (pending approval — not yet ratified; no code written)
**Date:** 2026-06-11
**Milestone:** M63

## Context

Athena's only lossless decode accelerator today is MTP speculative decoding
(M2/M40/M47), which only engages on MTP-variant checkpoints (vendored
Qwen3.5/3.6). The standard autoregressive models Athena also serves — Gemma4,
Qwen3 — have **no** speculative path and decode one token per backbone forward.
Gemma4 in particular is a first-class validated arch (`gemma4_unified` text
shipped v0.10.107) with no acceleration.

DFlash (arXiv:2602.06036) is a lossless block-diffusion speculative-drafting
technique: a small drafter conditioned on captured target hidden states proposes
a *block* of tokens in parallel; the autoregressive target verifies the block in
one forward and accepts the longest correct prefix. Output is identical to the
AR target's greedy decode — same lossless category as MTP. A clean MLX-Python
reference exists: **bstnxbt/dflash-mlx** (Apache-2.0).

### Feasibility spike findings (2026-06-11)

Read-only spike items completed; both resolved in favor. (The live on-hardware
benchmark is item 3, folded into M63.0 below.)

1. **Port distance — confirmed against real reference code**, not inferred from
   the arch doc:
   - `engine/target_gemma4.py` advertises `supports_recurrent_rollback=False`,
     `supports_kv_trim=True`. Rollback for the Gemma4 path is purely
     `_trim_recent_cache` — length truncation of the sliding `RotatingKVCache`
     and the full-attention `KVCacheSimple`. **The GatedDeltaNet tape/replay
     kernels (`gated_delta_tape`, `tape_replay`, DDTree) never touch the Gemma4
     path** — they exist only for the Qwen3.5/3.6 GDN hybrids
     (`engine/target_qwen_gdn.py`). Gemma4 verify uses stock SDPA (no custom
     `verify_qmm`).
   - The Gemma4 attention hook uses the exact `use_k_eq_v` value-source that
     Athena already fixed in the vendored `Gemma4Text.swift`
     (`project_gemma4-unified-support`): values come from raw `k_proj`, never
     RoPE'd.
   - The drafter (`model.py`, ~470 lines, `DFlashDraftModel`) is a standard
     Qwen3-style transformer (`nn.Linear` / `RMSNorm` / RoPE / SDPA) plus a
     context-fusion `fc` (`Linear(len(target_layers)*hidden → hidden)`) and a
     block of "noise" tokens drafted in one pass. **Every `mx.fast.*` kernel has
     a pure-`mx` fallback right beside it** (`model.py:630-654` — the
     `mx.fast.dflash_cross_attention` path falls back to
     `scaled_dot_product_attention`). The Swift port can ship the pure-`mx` path
     first and defer Metal kernels (M20/M2 "correctness-first" precedent).
   - The canonical decode cycle is documented (reference `docs/architecture.md`
     §"DFlash Cycle"): prefill+capture → stage target argmax → draft block →
     verify block in one target forward → accept-longest-prefix → commit
     `1 + acceptance_len` → KV-trim the rejected tail → restage.

2. **Drafter-weight license gate — clear.** Drafters
   `z-lab/gemma-4-31B-it-DFlash` and `z-lab/gemma-4-26B-A4B-it-DFlash` are tagged
   **apache-2.0** (~0.4B params each). As Gemma derivatives they are additionally
   governed by Google's **Gemma Terms of Use** — the same terms Athena already
   accepts by serving Gemma targets at all. **Athena never redistributes
   weights**: the operator pulls the drafter from Hugging Face at runtime exactly
   like any other model weight (the passive-oracle HF-fetch carve-out). The only
   thing Athena's repo vendors is the Apache-2.0 *code* we port → a NOTICE /
   attribution file. No redistribution blocker.

3. **Drafter availability constraint.** Drafters exist only for
   `gemma-4-31b-it-4bit` and `gemma-4-26b-a4b-it-4bit`. Athena's on-disk Gemma4
   is **12B** — no 12B drafter exists. First-supported targets therefore require
   pulling a new 4-bit target+drafter pair. Decision (operator, 2026-06-11):
   support **both** pairs, **31B dense first**, 26B-A4B MoE as a follow-on slice.

### M63.0 live benchmark results (2026-06-11) — **GO**

Stood up dflash-mlx 0.1.10 (Python 3.11, MLX 0.31.2) on the Mac Studio
(Mac17,6, 128 GB), `mlx-community/gemma-4-31b-it-4bit` +
`z-lab/gemma-4-31B-it-DFlash` (draft `w4`, block 16, adaptive verify),
256-token generations. Thermal pressure read "unknown" (possibly throttled), so
these are **conservative**:

| prompt | baseline | DFlash | speedup | acceptance |
|---|---|---|---|---|
| smoke (math reasoning) | 28.3 t/s | 65.4 t/s | 2.31× | 0.867 |
| chat (free prose) | 28.1 t/s | 53.7 t/s | 1.91× | 0.672 |
| code (Swift) | 28.2 t/s | 71.5 t/s | 2.54× | 0.789 |
| **json-extract (schema)** | 27.6 t/s | 84.5 t/s | **3.06×** | 0.809 |
| **json-list (array)** | 27.7 t/s | 107.5 t/s | **3.88×** | 0.852 |

**Structured output did NOT collapse acceptance — it was the *best* case** (the
feared failure mode is the reverse of reality: predictable JSON tokens draft
easily). Note this is *prompt-induced* JSON with no grammar mask; Athena's M47
Guide-mask-the-draft should hold acceptance at least this high under an enforced
schema, since the mask narrows draft and verify to the same allowed set. Worst
case (free prose) is still 1.91× with zero quality regression (lossless). DFlash
peak memory 18.6 GB vs 17.8 GB baseline (draft + captured features ≈ +0.8 GB);
fits the M5 governor budget. The idle installed daemon coexisted with the Python
run — no stop needed. **Go/no-go gate: GO.** Proceed to M63.1.

## Decision

Add a **DFlash lossless speculative-draft engine** to Athena, Gemma4-first,
default-OFF, vendored-from bstnxbt/dflash-mlx with Apache-2.0 attribution.

Key design choices:

1. **New engine, not an extension of the MTP loop.** `SpeculativeGeneration`
   (the MTP loop) is tightly bound to `AthenaQwen35Model` and a single-token
   draft with GDN/Mamba rollback. DFlash uses a *separate* draft model, an
   *attention-only* target (KV-trim rollback, no recurrent state), and a *block*
   draft. It lives in a new `Sources/AthenaLLM/DFlashGeneration.swift` that
   mirrors the MTP loop's *scaffolding* (GuidedDecoder/M47, prefix-cache hooks,
   heartbeat `DecodeProgress` counters, EOS-without-leak, M60.5 cancellation) but
   implements the DFlash cycle.

2. **Lossless ⇒ structured output is preserved.** Every emitted token equals the
   target's (Guide-masked) argmax, so JSON-schema / tool constrained decoding
   still works — apply the Guide at the target verify step. Tight-schema draft
   self-rejection is exactly the problem **M47** already solved: Guide-mask the
   draft block with `decoder.pick` (non-state-advancing) so it is drawn from the
   schema-allowed set. Carry the reference's "fall back to target-only AR when
   the speculative surplus goes negative" auto-fallback as the safety valve.

3. **temp==0 is lossless = every emitted token is the target's argmax under
   the verify forward.** DFlash verify accepts a draft token iff it equals the
   target argmax and otherwise emits the target argmax, so the committed
   sequence is exactly the target's *block-forward* greedy sequence.

   **AMENDED 2026-06-11 (M63.3 finding):** the original constraint "bit-identical
   to non-speculative (single-token) greedy" is **unattainable for any block
   speculative method on Gemma4**, and this is an intrinsic MLX kernel property,
   not an engine defect. The MLX SDPA dispatch
   (`backend/metal/scaled_dot_product_attention.cpp` `use_fallback`) selects a
   different kernel by query length — `sdpa_vector` for `qL=1` (the
   non-speculative single-token path), `sdpa_full`/pure-`mx` fallback for `qL≥2`
   (any block verify) — with different floating-point rounding. At genuine
   near-ties (~1/64 generated tokens on a representative prompt) the two kernels
   pick different argmax tokens. A diagnostic (`testKernelBlockSizeSensitivity`)
   confirms B=2/4/8/16 all agree with each other and differ from B=1 at the same
   single position; the `n≥2` block-forward greedy is self-consistent. (This is
   also why MTP's 2-token verify stayed bit-identical on Qwen — its smaller GQA
   keeps `qL=2` inside the vector kernel — but does not on Gemma4-31B, whose
   larger GQA pushes `qL≥2` out of it.) The only way to make the non-speculative
   path agree would be to route it through an `n≥2` forward, which would shift
   the validated base-decode numerics globally — rejected.

   **Revised contract:** DFlash is *lossless* in the speculative-decoding sense
   (it provably reproduces the self-consistent `n≥2` block-forward greedy) and
   matches single-token greedy except at SDPA-kernel near-ties. The **gate**
   (`DFlashEngineParityTests.testDFlashBitIdenticalToGreedy`) asserts DFlash
   matches single-token greedy up to the first divergence, and that the
   divergence is a *verified* kernel tie (the `n≥2` block forward of the shared
   prefix predicts exactly what DFlash committed, and differs from the
   single-token choice) — so the engine introduces **no error of its own**. An
   acceptance-rate observer (reuse `SpeculativeStats`) ships in the same slice.

4. **Gemma4 hidden-capture = a tracked additive substrate delta.** The engine
   needs the target's per-layer hidden states at selected layers during prefill
   and verify. Swift cannot monkeypatch the substrate forward (as the Python
   reference does). The substrate Gemma4 is already a tracked path-dep fork
   (`gemma4_unified` alias + sanitize + the k_eq_v fix). M63 adds an
   **additive-only** `callReturningHidden(capturing:)` to the substrate Gemma4
   text model, leaving the existing `callAsFunction` byte-unchanged (the M2.2b
   `logitsAndHidden` precedent). This **overrides the pristine-substrate rule**
   for the capture hooks specifically — the same scoped override M20 took for the
   TurboQuant codec — because no clean external seam exists (the layer loop and
   `RotatingKVCache` interplay are internal). Recorded and reviewable; existing
   Gemma4 inference paths are structurally unchanged.

5. **Config: 5-touchpoint, default-OFF, TOML.** A `dflash_enabled` flat key
   (default `false`) plus a target→draft mapping — a built-in registry of the two
   known pairs, overridable via TOML and an `ATHENA_DFLASH` env override
   (precedence env > TOML > built-in), mirroring the `kv_compression` /
   `prompt_cache` patterns. No JSON config (user preference).

6. **Surface: reuse the per-request `speculative` flag; no new route.** The
   runtime picks the engine from the loaded model — MTP-capable target → MTP
   loop; Gemma4 target with a draft attached → DFlash loop — exactly the M40
   "runtime picks the loop, not a new toggle" lesson. Route/field changes (the
   `speculative` description, draft-attachment surfacing on `/v1/models` or
   `/healthz`) go in `OpenAPISpec.swift` with the bidirectional drift-guard. The
   `{"error":{...}}` envelope is unchanged.

7. **Passive oracle preserved.** Drafter weights are HF fetches; no new outbound
   surface. No webhooks, no telemetry.

## Rejected alternatives

- **Native CUDA re-port / control-plane-over-vLLM** (the "two backend
  embodiments" sketch) — out of scope for M63, which is specifically the Apple
  Silicon in-process accelerator. Tracked separately in
  `project_diffusiongemma-support-investigation`.
- **Extend `SpeculativeGeneration`/MTP to cover Gemma4** — rejected: the draft
  source, target arch, rollback mechanics, and block-vs-single-token shape all
  differ; overloading one loop would entangle the bit-identical MTP contract with
  a different engine. A parallel engine with shared scaffolding is cleaner.
- **Full Athena-side vendor of the Gemma4 forward** (Whisper/Sortformer style)
  to keep the substrate pristine — rejected for M63: it duplicates a large,
  already-vendored-and-fixed forward and risks numeric divergence from the
  validated path. The additive substrate-delta capture hook is smaller and lower
  risk (M2.2b precedent).
- **Qwen3 / Qwen-GDN targets first** — rejected as first slice: GDN targets need
  the tape/replay rollback kernels DFlash's hard path, which Gemma4-first
  specifically dodges. Separate later milestone.
- **DiffusionGemma as the primary fast path** — rejected: parallel out-of-order
  token fill breaks constrained decoding, which is exactly what the consuming application's
  schemas need; DFlash preserves the M3/M49/M53 structured-output investment.

## Consequences

- Athena gains a second resident model (the drafter) under the M5 memory
  governor; draft weights + draft KV must be registered with the governor and the
  M41/M42 model-lifecycle/allowlist machinery.
- The substrate path-dep fork grows by one additive capture method on Gemma4
  (tracked; existing paths byte-unchanged). Athena reproducibility continues to
  depend on the substrate clone state.
- A NOTICE/attribution file for the Apache-2.0 bstnxbt/dflash-mlx source enters
  the repo.
- CLAUDE.md "Canonical pipelines"/"Architecture" gains a reference to this ADR
  once ratified.

## Implementation plan (sliced — conventions: appVersion bump IN each slice
commit in BOTH `Sources/athena/Athena.swift` and
`clients/Sources/athena/Athena.swift`; direct-to-main commit + annotated semantic
tag per slice; xcodebuild not swift build for MLX; e2e gate; `graphify update .`
after code changes; error envelope unchanged; passive-oracle preserved)

- **M63.0 — Feasibility benchmark (Python only, no Athena code, no version
  bump).** Stand up dflash-mlx in a venv; pull `gemma-4-31b-it-4bit` +
  `z-lab/gemma-4-31B-it-DFlash`; measure tok/s + acceptance on representative
  Athena/the consuming application prompts **including a structured-output prompt** (acceptance
  may collapse there). Record a numbers table. **Go/no-go gate**: if structured
  acceptance is below a useful threshold even with Guide-masked drafts, re-scope
  before any Swift. Coordinate Metal budget with the running daemon.
- **M63.1 — Vendored draft model + NOTICE (additive-only).** Port
  `DFlashDraftModel` into `Sources/AthenaModels/DFlash/` (pure-`mx` SDPA path;
  no Metal kernel yet); load + strict-bind z-lab drafter weights; Apache-2.0
  NOTICE. Additive-only ⇒ existing paths byte-unchanged (structural proof, M20.1/
  M21.1). Draft-forward parity test vs reference.
- **M63.2 — Gemma4 hidden-capture seam (tracked additive substrate delta).** Add
  `callReturningHidden(capturing:)` to substrate Gemma4 (existing forward
  byte-unchanged); Athena-side Gemma4 target adapter capturing selected-layer
  hiddens during prefill+verify + KV-trim rollback over the sliding/full caches.
- **M63.3 — DFlash decode engine + dispatch (default OFF).** New
  `Sources/AthenaLLM/DFlashGeneration.swift` implementing the 8-step cycle with
  Guide-aware block drafts (M47); new `runSpeculative` dispatch branch (Gemma4 +
  draft attached); config 5-touchpoint (`dflash_enabled` + target→draft registry/
  override + `ATHENA_DFLASH`). **Bit-identical-greedy A/B gate + acceptance-rate
  observer = slice deliverables.** Governor + lifecycle registration of the
  drafter. e2e gate; graphify update.
- **M63.4 — Structured output + observability + OpenAPI.** Validate Guide masking
  end-to-end through DFlash on a structured prompt (M47 draft-self-rejection fix +
  target-only auto-fallback safety valve); reuse the `speculative` flag (runtime
  picks engine); update `OpenAPISpec.swift` + drift-guard; surface draft
  attachment; usage/`cached_tokens` parity. Error envelope unchanged.
- **M63.5 — 26B-A4B MoE pair (follow-on). BLOCKED — out of DFlash scope.**
  `mlx-community/gemma-4-26b-a4b-it-4bit` is a 128-expert MoE
  (`num_experts=128`, 30 layers), and the substrate `Gemma4Text.swift` has **no
  expert-routing support** (only the dense `Gemma4MLP` + PLE gating) — the MoE
  target cannot load at all. Unblocking M63.5 requires a substrate Gemma4 MoE
  port, a Gemma4-architecture workstream independent of DFlash. All the
  DFlash-side plumbing is already in place: the `DFlashRegistry` pair (M63.3b),
  the capture seam (arch-general — `callReturningHidden` works on any
  `Gemma4TextModel`; MoE only changes the MLP), and the dispatch. When substrate
  MoE support lands, M63.5 is just: pull the pair + run the existing
  parity/bit-identical gates.

### Status (2026-06-11)

M63.1–M63.4 SHIPPED (v0.10.108–112): the **31B dense Gemma4 path is complete**
— lossless DFlash speculative decoding, default-off, dispatched through the
request path, with stop-token parity, validated end-to-end (bit-identical to the
block-forward greedy; matches single-token greedy except at the documented SDPA
kernel ties). M63.5 (MoE) is blocked on substrate Gemma4 MoE support as above —
the unblocking workstream is **ADR 002 (M64)**, which adds the additive Gemma4 MoE
port; M63.5 ships as M64.4.

### Deferred / tracked (NOT silent descopes)

- **DFlash-accelerated structured output** (M63.4 deferral) — Guide-masked block
  draft/verify (the M47 generalization over a block) + target-only auto-fallback.
  Structured requests stay correct on the substrate guided path; DFlash does not
  yet accelerate constrained decoding. The verify side needs sequential
  pick+commit over the block with the Guide advancing per committed token; the
  draft side needs a Guide peek-advance/rollback shape the current Guide lacks.

- `mx.fast.dflash_cross_attention` + verify Metal kernels — pure-`mx` fallback
  ships first; kernels are a perf follow-up (M20 #909 / M2 "Release tuning"
  precedent).
- DFlash prefix-snapshot L1/L2 reuse — Athena's M59 prompt-prefix cache already
  covers cross-request reuse on the MTP path; DFlash prefix-cache integration is a
  later slice.
- Adaptive verify policy / DDTree / copyspec / sparse prefill — optional
  optimizations.
- Qwen3-pure-attention and Qwen-GDN targets — separate milestone (GDN needs the
  tape/replay port Gemma4-first dodges).
- temp>0 sampling for DFlash — mirror M40; deferred (DFlash verify is argmax;
  block sampling-speculative is a separate formulation).
