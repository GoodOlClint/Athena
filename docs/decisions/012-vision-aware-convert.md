# 012 — Vision-aware `athena convert`: load via the VLM path, keep the vision tower full-precision

**Status:** Proposed — pending operator ratification
**Date:** 2026-06-17
**Milestone:** M72 (proposed number — operator's call) — goal: `athena convert google/gemma-4-26B-A4B` produces a model that loads and serves **vision**.

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

2. **Quantize the language model, NOT the vision tower.** Replace the blanket
   `quantize(model:groupSize:bits:)` with the **closure form**
   `quantize(model:) { path, module in … }` (the same shape the loader uses), returning:
   - `nil` for any `vision_tower.*` / `audio_tower.*` path → **full-precision, no `.scales`**
     (so the `.scales`-driven loader leaves it unquantized — matches the reference);
   - the per-layer **(groupSize: 64, bits: 8)** for MoE `mlp.{gate,up,down}_proj` +
     `router.proj` paths;
   - the global **(groupSize, bits)** (default 4-bit) for every other quantizable
     language-model layer.

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
- **Generality / audio (operator intent).** The rule is framed as *"quantize the language
  model; leave EVERY modality tower full-precision"* — the skip set lists the vision **and
  audio** tower prefixes (`vision_tower`/`embed_vision`, `audio_tower`/`embed_audio`) now, even
  though audio is not served. So convert already produces the right artifact for an
  audio-bearing checkpoint (audio tower preserved full-precision, language quantized) the
  moment the substrate gains an audio encoder — **no convert rework**. M72 does NOT and cannot
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
