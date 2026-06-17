# 012 — Vision-aware `athena convert`: load via the VLM path, keep the vision tower full-precision

**Status:** Accepted — **Implemented & validated**. M72.1 (vision-aware convert: load + quantize + config) shipped v0.10.161. M72.2 (real-model validation) DONE 2026-06-17: `athena convert google/gemma-4-26B-A4B-it --q-bits 4` → loads + accurately describes a test image; the base `google/gemma-4-26B-A4B` converts + loads with the vision tower intact (chat blocked only by its missing chat template — use `-it`).
**Date:** 2026-06-17
**Milestone:** M72 — goal: `athena convert google/gemma-4-26B-A4B` produces a model that loads and serves **vision**.

## Context

M71 (ADR 010) made the daemon **serve** vision: a checkpoint with a top-level
`vision_config` loads through the substrate `VLMModelFactory` (`MLXVLM.Gemma4`, image
tower). Validated 2026-06-17 on `mlx-community/gemma-4-e2b-it-4bit` and — for the MoE
target — on `mlx-community/gemma-4-26b-a4b-it-4bit` (15.89 GB; loaded in 1.3 s; correctly
described a test image). So the **serve** side handles MoE + vision end-to-end.

The operator's goal is to convert the **original Google weights** themselves:
`athena convert google/gemma-4-26B-A4B`. Current state (verified 2026-06-17):

- `google/gemma-4-26B-A4B` is `model_type: gemma4`, `vision_config` + `audio_config`
  present, with the **full `vision_tower`** (355 keys, `patch_embedder.input_proj`) — the
  substrate-supported VLM arch (NOT the omni `gemma4_unified_audio` minimal
  `vision_embedder` arch the substrate does not implement; see ADR 010 / RUNBOOK J).
- **`athena convert` is text-only** (`Sources/AthenaLLM/ModelConvert.swift`):
  1. It loads via the generic `loadModelContainer(from:)` (line 75). The factory registry
     tries the **LLM** factory first and it wins for `gemma4` → `MLXLLM.Gemma4Model`, whose
     `sanitize()` **strips `vision_tower` (and `audio_tower`)**. The vision weights never
     reach the resident model, so they are never re-saved.
  2. `quantize(model:groupSize:bits:)` (line 104) quantizes **every** Quantizable layer
     blanket — it would quantize the vision tower too.
  3. `writeConfig` (line 143) round-trips the whole source config (so `vision_config` would
     survive) and writes a flat global `quantization` block.
- The substrate loader is **`.scales`-driven** (`MLXLMCommon/Load.swift:41-51`): on load it
  quantizes a layer **iff** `weights["\(path).scales"]` exists in the saved file, taking the
  bits from the per-layer config (else the global). So whatever convert saves *with* scales
  is quantized on reload; whatever it saves *without* scales stays full-precision.
- The reference recipe — what a correct converted model looks like — is
  `mlx-community/gemma-4-26b-a4b-it-4bit`: the **language model is quantized** (global 4-bit
  + per-layer **8-bit** on the MoE `mlp.{gate,up,down}_proj` + `router.proj`, mode `affine`),
  and the **vision tower is NOT quantized** (full-precision `.linear.weight`, no `.scales`,
  zero quant-config entries).

## Decision

Make `athena convert` **vision-aware**, mirroring the serve path (ADR 010) and the
reference recipe. Four sub-decisions:

1. **Load the model for conversion through the VLM path** when the source checkpoint is a
   vision checkpoint (`ModelConfigInfo.hasVisionConfig`, the same detector M71.2 uses):
   `VLMModelFactory.shared.loadContainer(from:using:)` instead of the generic
   `loadModelContainer`. This keeps the vision tower in the resident model so it is
   re-saved. (Audio is still dropped — `MLXVLM.Gemma4.sanitize` strips `audio_tower`; the
   converted model is text + vision, matching the served capability.)

2. **Quantize everything except the encoder towers.** Replace the blanket
   `quantize(model:groupSize:bits:)` with the **closure form**
   `quantize(model:) { path, module in … }` (the same shape the loader uses). The exact
   scheme is pinned to the reference (`gemma-4-26b-a4b-it-4bit` quant config + `.scales`
   inventory, verified 2026-06-17):
   - `nil` for the **encoder towers** — any `vision_tower.*` / `audio_tower.*` path →
     **full-precision, no `.scales`** (the `.scales`-driven loader then leaves them
     unquantized). NOTE: only the ENCODER towers are skipped — the multimodal *projections*
     `embed_vision` / `embed_audio` **are** quantized (4-bit) in the reference.
   - **(groupSize: 64, bits: 8)** for the per-layer **dense** `…layers.N.mlp.{gate,up,down}_proj`
     and `…layers.N.router.proj`.
   - the global **(groupSize, bits)** (default 4-bit) for everything else quantizable — the MoE
     experts `experts.switch_glu.{gate,up,down}_proj`, `self_attn.{q,k,v,o}_proj`,
     `embed_tokens`, and `embed_vision.embedding_projection`.
   (Encoder-tower skip is checked FIRST, so a tower's own `mlp.down_proj` is full-precision,
   not 8-bit.)

3. **Emit a matching quantization config** from `writeConfig`: a single **quant-rule
   function** is the source of truth, used BOTH by the quantize closure (2) and to generate
   the config's per-layer `quantization` block (global 4-bit + the MoE 8-bit overrides),
   so the reloaded model rebuilds exactly the layers convert quantized. `vision_config` /
   `audio_config` are preserved by the existing whole-object round-trip; the vision tower
   gets **no** quant-config entry (consistent with no `.scales`).

4. **No new substrate code.** This is an Athena-side change to `ModelConvert` (+ the shared
   quant-rule). `MLXVLM.Gemma4` already serves the converted model (proven on the reference).

## Consequences

- `athena convert google/gemma-4-26B-A4B` (and other gemma-4 VLM checkpoints) produces a
  model whose vision tower is full-precision and language model is mixed 4/8-bit — load- and
  serve-compatible with the M71.2 VLM path. Closes the operator's goal.
- The vision tower stays full-precision: a few hundred MB larger than a fully-quantized
  model, but it matches the reference and avoids the special `Gemma4ClippableLinear` quant
  scheme (out of scope; could be a later size-optimization).
- **Base vs instruct:** `google/gemma-4-26B-A4B` (base) ships **no chat template** and is a
  multimodal base → chat/text output degenerates (ADR-002-era caveat). Convert mechanics are
  identical; for usable chat use the `-it` variant. The plan validates vision description on
  a `-it` checkpoint.
- Convert memory: a 26B bf16 source (~52 GB, mmap) quantizing to ~16 GB output — the
  existing incremental-materialize + capped cache (ModelConvert lines 110-126) must hold for
  the VLM model too (vision tower adds a full-precision slice). A validation risk, not a
  design change.
- The quant-rule is gemma-4-MoE-shaped (router/mlp paths). A non-MoE or non-gemma vision
  checkpoint converts with vision-skipped + global-bits language quant (no 8-bit overrides) —
  still correct, just not mixed-precision.
- **Generality / audio (operator intent).** The rule is framed as *"quantize everything except
  the encoder towers"* — the skip set lists the vision **and** audio ENCODER prefixes
  (`vision_tower`, `audio_tower`) now, even though audio is not served (the `embed_vision` /
  `embed_audio` projections are quantized, matching the reference). So convert already produces
  the right artifact for an audio-bearing checkpoint (audio encoder preserved full-precision,
  its projection + language quantized) the moment the substrate gains an audio encoder —
  **no convert rework**. M72 does NOT and cannot
  *validate* audio: the substrate `sanitize` strips `audio_tower` on load (no Swift audio
  encoder yet — ADR 010's deferred audio-in-chat), so an audio tower has nothing to load into.
  Audio-in-chat is the blocker, the convert pipeline is not; convert covers audio for free once
  that port lands. This costs no extra effort now — listing the audio prefixes in the skip set
  is the same work as listing vision.

## Alternatives rejected

- **Flat global quant of the whole model (incl. vision)** — rejected: blanket-quantizing the
  vision tower diverges from the reference and risks the clippable-quant mismatch; the
  operator chose to match the reference (vision full-precision, mixed-quant language).
- **Convert via the text LLM factory + re-attach vision** — rejected: the LLM factory strips
  the tower; re-attaching is a parallel re-implementation of the VLM load the substrate
  already provides.
- **Port the omni `gemma4_unified_audio` vision arch** — out of scope: the 26B-A4B target is
  the supported full-`vision_tower` family; omni support is a separate milestone (RUNBOOK J).

## Amendment — 2026-06-17: the 8/4 mixed scheme is **Gemma-4-specific**; other arches quantize uniformly

**Problem found (post-ship):** `Gemma4QuantRule`'s per-layer 8-bit override —
`isOverrideLayer` matching `.mlp.{gate,up,down}_proj` (+ `.router.proj`) — was pinned to the
`mlx-community/gemma-4-26b-a4b-it-4bit` reference, where it hits only the *few* dense
`mlp` layers among mostly-MoE `experts.switch_glu.*` (which have no `.mlp.` and stay 4-bit).
That assumption is **Gemma-4-MoE-shaped** and silently misfires on other arches, because the
override keys on a *Gemma-4 naming convention*, not on a real shared-vs-expert distinction:

- **`qwen3_5` (dense 27B):** *every* layer's `mlp.{gate,up,down}_proj` matches → the whole MLP
  stack (≈⅔ of weights) is forced to 8-bit. A "4-bit" convert came out **22.9 GB** instead of
  ~13.5 GB (verified: 192 = 64 layers × 3 overrides, all 8-bit).
- **`qwen3_5_moe` (35B-A3B):** Qwen names its **routed experts** `mlp.switch_mlp.*_proj`
  (contains `.mlp.`) → the override forces the **expert bulk** to 8-bit. The "4-bit" build came
  out **34 GB** ≈ the 8-bit build (35 GB) ≈ **½ of the 65 GB unquant** — i.e. effectively 8-bit;
  a true 4-bit would be ~16 GB (verified: 240 = 40 layers × {shared_expert,switch_mlp}×3, all
  8-bit).

**Decision:** the mixed 8/4 scheme **reproduces a specific published Gemma-4 quant and is not a
general quantizer.** `athena convert` applies the per-layer 8-bit override **only when
`model_type == "gemma4"`**; every other arch (`qwen3_5`, `qwen3_5_moe`, …) is quantized
**uniformly at the global `--q-bits`**. The **encoder-tower full-precision skip
(`vision_tower`/`audio_tower`) stays unconditional** for *all* arches — it is correct for any
vision checkpoint and is what lets a converted Qwen-VL load (the M71.2 `.scales`-driven loader
leaves the tower full-precision). Detection keys on the top-level `model_type`
(`gemma4` / `qwen3_5` / `qwen3_5_moe`), captured in `ModelConfigInfo`.

**Consequences:** Gemma-4 converts are **byte-unchanged**. Re-converting the Qwen models yields
true 4-bit sizes (dense ~13.5 GB, MoE ~16 GB); the existing oversized exports still load/serve
(valid mixed-precision), they just use more of the Metal budget — re-convert to reclaim it. A
per-arch shared-vs-expert *quality* knob (8-bit shared path for Qwen-MoE, like Gemma-4 gets) is
**deliberately deferred** — uniform 4-bit is correct and predictable; revisit only if a Qwen-MoE
shows a measurable quality regression. Plan: `docs/convert-quant-arch-gate-plan.md`.
