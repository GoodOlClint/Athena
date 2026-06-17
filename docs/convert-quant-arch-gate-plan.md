# Convert quant: arch-gate the Gemma-4 mixed-precision scheme — change plan

**Status:** Plan — awaiting operator review (do not implement yet)
**Date:** 2026-06-17
**Decision of record:** ADR 012 amendment (2026-06-17) — the 8/4 mixed scheme is
Gemma-4-specific; other arches quantize uniformly at the global bits.
**Milestone:** operator-assigned. appVersion bump in the slice commit.

## Problem (verified, not theory)

`Gemma4QuantRule` forces `mlp.{gate,up,down}_proj` (+ `.router.proj`) to 8-bit. That override
reproduces the `mlx-community/gemma-4-26b-a4b-it-4bit` recipe, where it hits only the handful
of dense `mlp` layers among MoE `experts.switch_glu.*` (which contain no `.mlp.` and stay
4-bit). On other arches the **path pattern** misfires:

| Convert | model_type | What the override catches | Size got | Size should be |
|---|---|---|---|---|
| Gemma-4 26B-A4B | `gemma4` | few dense `mlp` (correct) | ~14.9 GB | ✓ (reference) |
| Qwen3.5-27B (dense) | `qwen3_5` | **all 192** `.mlp.*_proj` | 22.9 GB | ~13.5 GB |
| Qwen3.5-35B-A3B (MoE) | `qwen3_5_moe` | **240** incl. `switch_mlp` experts | 34 GB | ~16 GB |

Evidence: per-module quant configs (all overrides 8-bit, incl. `mlp.switch_mlp.down_proj`);
size ladder vs the 65 GB unquant MoE (4-bit build 34 GB ≈ 8-bit build 35 GB ≈ ½ unquant).

## Decision (fix A)

Apply the 8-bit per-layer override **only for `model_type == "gemma4"`**; quantize every other
arch **uniformly at the global `--q-bits`**. Keep the **encoder-tower skip
(`vision_tower`/`audio_tower` → full-precision) unconditional** for all arches.

Rejected — "gate on MoE": insufficient. The override's *patterns* are Gemma-4-specific, so a
Qwen-MoE would still mis-split (`switch_mlp` experts caught). Arch-gating the whole scheme is
the honest fix.

Deferred — per-arch shared-vs-expert quality knob (8-bit shared path for Qwen-MoE). Uniform
4-bit is correct + predictable; revisit only on a measured Qwen-MoE quality regression.

## Design

- **`Gemma4QuantRule`** gains a `mixedPrecision: Bool` (the 8-bit dense/router override is
  applied only when true). `quantization(forPath:)`:
  1. `isEncoderTower` → `nil` (full-precision) — **unchanged, all arches**.
  2. `mixedPrecision && isOverrideLayer` → `overrideBits` (8).
  3. else → global `bits`.
  Because `overrides(forModules:)` derives from `quantization(forPath:)`, the emitted
  `config.json` stays in lock-step — a uniform convert writes **zero** per-module overrides.
- **Arch detection** lives in `ModelConfigInfo` (already reads config.json): expose the
  top-level `model_type`; `mixedPrecision = (modelType == "gemma4")`. (Keys observed:
  `gemma4`, `qwen3_5`, `qwen3_5_moe`.)
- **`ModelConvert`** sets `mixedPrecision` from `ModelConfigInfo` when constructing the rule
  (both the quantize closure and the config emit use the same instance — no second source).
- Naming: keep `Gemma4QuantRule` for now (minimal blast radius); a rename to a neutral
  `ConvertQuantRule` is noted as optional follow-up, not in this slice.

## Slices (each: commit + annotated tag; appVersion bump in the slice commit; test-pinned)

1. **Rule + detection + wiring** — add `mixedPrecision` to `Gemma4QuantRule`, gate the
   override; expose `modelType` on `ModelConfigInfo`; set the flag in `ModelConvert`.
   **Tests (`swift test`, MLX-free):** with `mixedPrecision=false`, a dense
   `…layers.0.mlp.down_proj` AND a `…mlp.switch_mlp.down_proj` both return the **global** bits
   (not 8); encoder tower still `nil`; `overrides(forModules:)` returns `[]`. With
   `mixedPrecision=true`, existing Gemma-4 expectations unchanged (regression-pin the current
   192/240-style behavior on representative paths).
2. **Docs** — `athena convert` reference / quickstart note: mixed 8/4 is Gemma-4-only; other
   arches uniform at `--q-bits`; encoder towers always full-precision.

## Verification

- `swift test` (the rule is pure) + `xcodebuild Release`.
- **Real-model RUNBOOK (operator, Metal host):** re-convert `Qwen3.5-27B` and
  `Qwen3.5-35B-A3B` at `--q-bits 4`; confirm sizes drop to ~13.5 GB / ~16 GB, the emitted
  `config.json` has **no 8-bit overrides** (global 4-bit only, vision tower absent from quant
  config), and both **load + chat** (dense) / **load + describe an image** (MoE-VL). Re-convert
  Gemma-4 26B-A4B and confirm it is **byte-identical** to the pre-fix output.

## Companion change — convert output naming (`-mlx-Nbit`)

Folded in (operator request): a quantized convert's output name now carries the
`-mlx` family marker like the unquantized one — `<base>-mlx-<N>bit` (was
`<base>-<N>bit`); unquantized stays `<base>-mlx`. Purely a label — quant level
is read from `config.json`, never parsed from the name — so it's cosmetic and
affects only NEW converts (existing store entries keep their names). Touches
`ModelConvert.outName` + the `athena convert --name` help. Ships in the same
convert commit as the quant fix (both touch `ModelConvert.swift`, so a clean
hunk-split isn't worth interactive staging) — called out distinctly in the
commit message.

## Out of scope

No change to the encoder-tower skip, the load path, or the Gemma-4 recipe. No per-arch
quality knob. No rename (optional follow-up). Existing oversized Qwen exports keep working;
re-convert to reclaim memory.
