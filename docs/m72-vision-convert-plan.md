# M72 — Vision-aware `athena convert` — change plan

**Status:** Proposed change plan — approval gate (do not implement until ratified).
**Goal:** `athena convert google/gemma-4-26B-A4B` produces a model that loads and serves **vision**.
**ADR:** [`docs/decisions/012-vision-aware-convert.md`](decisions/012-vision-aware-convert.md).
**Milestone number proposed** — operator picks the number + tags.

## Why (one line)

M71 made the daemon *serve* vision; M72 makes the daemon *produce* a vision model from raw
Google weights via the in-process convert pipeline.

## Serve baseline — VALIDATED 2026-06-17 (so the gap is convert-only)

- `mlx-community/gemma-4-e2b-it-4bit` (`gemma4_audio`, full `vision_tower`) → image described.
- `mlx-community/gemma-4-26b-a4b-it-4bit` (MoE, 15.89 GB) → loaded 1.3 s, **correctly described
  the test image** ("a solid red square on the left and a solid white area on the right").
- So `MLXVLM.Gemma4` serves **MoE + vision** end-to-end. The only thing missing is a convert
  pipeline that emits such a model.

## Reference recipe (target output shape)

`mlx-community/gemma-4-26b-a4b-it-4bit`:
- **Language model quantized**: global 4-bit (`mode: affine`, `group_size: 64`) + **per-layer
  8-bit** on MoE `language_model.model.layers.N.mlp.{gate,up,down}_proj` + `router.proj`.
- **Vision tower full-precision**: `vision_tower.*.linear.weight`, **no `.scales`**, no quant
  entries. The substrate loader (`.scales`-driven, `Load.swift:41-51`) therefore leaves it
  unquantized automatically.

## Scope

**In:** vision-aware load + quantize + config emission in `ModelConvert`, so a `gemma4`
vision checkpoint (full `vision_tower`) converts to a load/serve-compatible model with the
vision tower full-precision and the (MoE) language model mixed 4/8-bit; validation on a `-it`
checkpoint then on `google/gemma-4-26B-A4B`.

**Out:** the omni `gemma4_unified_audio` vision arch (separate substrate port — RUNBOOK J);
quantizing the vision tower (full-precision by design); audio-in-chat (audio tower dropped by
`sanitize`); non-gemma VLM arches (convert still works with vision-skipped + global-bits
language quant, just no MoE 8-bit overrides).

**General by construction (operator intent: "convert should work for everything, incl. audio").**
The quant-rule skips EVERY modality tower (`vision_tower`/`embed_vision` **and**
`audio_tower`/`embed_audio`) and quantizes the language model — so convert already emits the
correct artifact for an audio-bearing checkpoint once the substrate can serve audio. M72 does
not *target* audio (blocked: `sanitize` strips `audio_tower`, no Swift audio encoder yet — ADR
010); convert covers it for free when that port lands, no rework. Listing the audio prefixes
now is the same effort as listing vision.

## Verified current-state seams (the map)

- `Sources/AthenaLLM/ModelConvert.swift:75` — generic `loadModelContainer` (LLM factory wins →
  strips `vision_tower`). **→ route vision checkpoints via `VLMModelFactory`.**
- `ModelConvert.swift:104` — blanket `quantize(model:groupSize:bits:)`. **→ closure form,
  skip vision/audio, mixed 4/8-bit language.**
- `ModelConvert.swift:143 writeConfig` — round-trips config (preserves `vision_config`),
  writes a flat quant block. **→ emit per-layer quant config from the shared quant-rule.**
- `MLXLMCommon/Load.swift:41-51` — loader quantizes iff `.scales` present, per-layer bits from
  config. (The contract convert must satisfy; no change.)
- `Sources/AthenaLLM/ModelConfigInfo.swift` — `hasVisionConfig` (reuse M71.2's detector).
- `MLXVLM.Gemma4` — serves MoE + vision (validated; no change).

## Slices (each: xcodebuild Release → swift test + e2e phase → annotated tag → graphify)

- **M72.1 — complete vision-aware convert (load + quantize + config). SHIPPED v0.10.161.**
  Load/quant/config are interdependent (vision-aware load without the quant filter would
  blanket-quantize the vision tower), so they land together as one coherent slice, with the
  pure rule as the tested core (no broken intermediate). Adds `Gemma4QuantRule` (path →
  `(groupSize, bits)?`): `nil` for the **encoder** towers `vision_tower`/`audio_tower`; 8-bit
  for the dense `…mlp.{gate,up,down}_proj` + `router.proj`; global (4-bit) else — incl. the MoE
  experts `experts.switch_glu.*`, attn, `embed_tokens`, and `embed_vision.embedding_projection`
  (projections ARE quantized; only encoder towers are skipped). `ModelConvert` loads vision
  checkpoints via `VLMModelFactory` (`hasVisionConfig`), quantizes with the closure-form
  `quantize(model:filter:)` driven by the rule, and `writeConfig` emits `{group_size, bits,
  mode:affine}` + the per-layer 8-bit overrides derived from the modules that actually gained
  `.scales` (via `rule.overrides(forModules:)`), so weights and config agree by construction.
  Unit-tested: `Gemma4QuantRuleTests` (paths → bits, tower-skip, experts≠8-bit, override
  emission).
- **M72.2 — real-model validation. ✅ DONE 2026-06-17.** Converted `google/gemma-4-e2b-it`
  (3.5 GB) and `google/gemma-4-26B-A4B-it` (14.9 GB) via `athena convert … --q-bits 4`; both
  loaded through the VLM path and **accurately described the test image** end-to-end. Verified
  the converted configs/weights match the reference recipe (global 4-bit + dense-`mlp`/`router`
  8-bit; vision_tower full-precision; experts 4-bit; `vision_config` preserved). The base
  `google/gemma-4-26B-A4B` also converts + loads (vision tower intact) but its chat is blocked
  by the base's missing chat template (`missingChatTemplate` — use `-it`). Recorded in RUNBOOK
  scenario K. **No code change** (the convert code shipped in M72.1). Known wart:
  `missingChatTemplate` returns `500 module_load_failed` instead of a clear 400 — minor
  follow-up.

## Test bar

- **CI (stub-tier):** the pure `Gemma4QuantRule` unit tests + a `writeConfig` round-trip test
  (`swift test`); existing convert e2e phase stays green. xcodebuild Release per slice.
- **Real-model (RUNBOOK):** convert → load → image-description on a `-it` checkpoint, then the
  26B base. Convert is heavy (26B bf16 source); run on the Studio, not CI.
- **Regression:** a text-only `gemma4`/qwen convert is byte-unchanged (vision-skip only
  triggers on `hasVisionConfig`; non-vision path identical).

## Risks / open questions

- **Convert memory for 26B**: ~52 GB bf16 source → ~16 GB output + a full-precision vision
  slice; the incremental-materialize + capped cache must hold. Validate on the Studio.
- **`quantize(model:)` closure form** must accept the per-layer `(groupSize, bits)?` return
  (the loader uses exactly this shape — high confidence) and compose with the VLM model's MoE
  `SwitchLinear` / vision `Gemma4ClippableLinear` (skipped) modules.
- **Config key ↔ module path alignment**: the emitted per-layer keys must match the paths the
  loader feeds `perLayerQuantization.quantization(layer:)`. Derive both from the same rule.
- **Base chat quality**: `google/gemma-4-26B-A4B` base has no chat template → poor chat; vision
  description still demonstrable. Validate chat on `-it`.

## Approval gate

This plan + ADR 012 are the approval artifacts. On ratification: confirm the milestone number +
first tag, then implement M72.1. No implementation code until then.
