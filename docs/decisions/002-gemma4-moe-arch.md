# 002 — Gemma4 MoE architecture support (26B-A4B; unblocks DFlash M63.5)

**Status:** Accepted — Implemented (M64.1–M64.4 shipped v0.10.113–116)
**Date:** 2026-06-11
**Milestone:** M64

## Context

Athena serves Gemma4 as a first-class validated arch (`gemma4_unified` text shipped
v0.10.107; dense DFlash speculative decoding shipped v0.10.108–112, ADR 001). The
mixture-of-experts Gemma4 targets — first `mlx-community/gemma-4-26b-a4b-it-4bit`
(128 experts, top-8, `moe_intermediate_size=704`) — **cannot load**: the vendored
substrate `Gemma4Text.swift` predates Gemma4 MoE and implements only the dense
`Gemma4MLP`, so the expert-routing weights have no home. This is the sole blocker
behind ADR 001's deferred **M63.5** (DFlash on the 26B-A4B target): every other
piece of DFlash plumbing — the `DFlashRegistry` pair, the arch-general
hidden-capture seam, the request-path dispatch — is already in place.

Upstream `mlx-lm` `gemma4_text.py` already carries the full MoE implementation
(`Router`, `Experts`, `GeGLU`, and the hybrid dense+sparse `DecoderLayer`). This
ADR is therefore a **line-exact additive port** of that block into the substrate,
reusing the substrate's existing `SwitchGLU`.

### Feasibility spike findings (M64.0, 2026-06-11)

Read-only spike completed against the real reference code, the substrate, and the
target checkpoint's `config.json` + weight map (fetched from HF; weights not yet
pulled). Both named risks resolved **in favor — no load-time blocker**.

1. **Mixed per-layer 4/8-bit quantization — resolved, zero code.** The 26B's
   `quantization_config` is default 4-bit/group-64 with explicit **8-bit**
   overrides for every `language_model.model.layers.N.mlp.{gate,up,down}_proj`
   and `router.proj`. The substrate already honors this generically:
   `BaseConfiguration` parses it into `perLayerQuantization` with a
   `quantization(layer:)` resolver, and `Load.swift` (`loadWeights`) drives
   `quantize(model:)` with a predicate that, for any module whose
   `"{path}.scales"` is present in the weights, returns
   `perLayerQuantization.quantization(layer: path)?.asTuple`. Experts
   (`switch_glu.*`, no override) → default 4-bit; `mlp.*`/`router.proj` → 8-bit;
   `router.scale`/`router.per_expert_scale`/`layer_scalar`/norms (no `.scales`)
   → skipped. `SwitchLinear` is `Quantizable` (`toQuantized` →
   `QuantizedSwitchLinear` / `gatherQuantizedMM`). The dense 31B already loads
   quantized through the identical nested `language_model.model.layers.*` path,
   so module-path matching is proven. **The port writes no quantization code.**

2. **Substrate-vs-upstream Gemma4 divergence — bounded to the MoE block.** The
   26B's scalars (`num_kv_shared_layers=0`, `hidden_size_per_layer_input=0` (no
   PLE), `use_double_wide_mlp=false`, `attention_k_eq_v=true`,
   `num_global_key_value_heads=2`, `global_head_dim=512`, 30 layers, hidden
   2816, 16 q / 8 kv heads, sliding 1024, softcap 30) are identical in
   attention/RoPE/norm shape to the shipped 31B. Verified from the weight map:
   full-attention LM layers carry no `v_proj` (k-eq-v), sliding layers do —
   exactly what the shipped `Gemma4Attention` already implements. So attention,
   MLP, RoPE, the 4-norm sandwich, kv-share, and PLE paths are **byte-unchanged**;
   the only new surface is the MoE block.

3. **Weight layout — pre-split, no fused gate_up, no sanitize-split.** The
   mlx-community checkpoint stores experts already split and per-expert-stacked:
   `layers.N.experts.switch_glu.{gate,up,down}_proj.{weight,scales,biases}`
   (maps directly onto a plain `SwitchGLU`),
   `layers.N.router.{proj.{weight,scales,biases}, scale, per_expert_scale}`, and
   the three extra norms `layers.N.{post_feedforward_layernorm_1,
   post_feedforward_layernorm_2, pre_feedforward_layernorm_2}.weight`. The
   upstream `sanitize` gate_up_proj split and the substrate's pre-staged
   `FusedGateUpSwitchGLU` are **not needed** for this target.

## Decision

Add **Gemma4 MoE support** as a **tracked additive delta** on the substrate
path-dep fork, Gemma4-first (26B-A4B), reusing the substrate `SwitchGLU`.

1. **Line-exact additive port into substrate `Gemma4Text.swift`.** Port the
   upstream `Router`, `Experts`, `GeGLU`, and the hybrid `DecoderLayer` forward.
   Mirror Gemma's router ordering exactly (RMS-norm with `scale·hidden^-0.5` →
   `proj` → top-k via `argPartition` → **softmax over the selected logits** →
   `·per_expert_scale`), which differs from the in-tree Qwen3MoE
   softmax-then-select. Reuse the substrate `SwitchGLU`
   (`MLXLMCommon/SwitchLayers.swift`); do **not** re-implement the expert
   gather-matmul.

2. **Hybrid dense+sparse layer, summed — every layer, gated on
   `enable_moe_block`.** `h = post_ff_norm( post_ff_norm_1(mlp(pre_ff_norm(h))) +
   post_ff_norm_2(experts(pre_ff_norm_2(h), router(h))) ) + residual`. This is
   not a per-layer dense/MoE split; when `enable_moe_block` every layer runs both
   branches.

3. **Additive substrate delta; dense path byte-unchanged.** Config gains
   `enableMoeBlock` (default false), `numExperts`, `topKExperts`,
   `moeIntermediateSize`; the new modules and the three extra norms are
   constructed only when `enableMoeBlock`, and the dense `DecoderLayer` branch is
   untouched. This **overrides the pristine-substrate rule** for Gemma4
   specifically — the same scoped override M20 (TurboQuant) and M63.2 (the
   capture seam) already took — because the layer/cache interplay is internal and
   no clean external seam exists. Proof obligation: a bit-identical greedy A/B on
   a dense Gemma4 (`gemma-4-12B-it-8bit`) showing the dense decode is unchanged.

4. **Mixed quantization is config-driven (no port code).** Per finding 1; the
   port only ensures the module tree produces paths matching the config's
   per-layer keys (it does, via the multimodal `language_model.model.layers.*`
   nesting the 31B already uses).

5. **No new Athena arch registration.** model_type `gemma4` → `Gemma4Model` →
   `Gemma4TextModel` via the existing alias; `gemma4`/`gemma4_text` are already in
   `SupportedModels.validatedSubstrate`. M64 is a substrate port + tests + the
   M63.5 DFlash validation.

6. **Validation = Python-parity + coherence.** A dumped-reference fixture (the
   M63.1 pattern): reference logits + selected hidden for a short token sequence
   from Python `mlx-lm`, asserted bf16-equal against the Swift forward; plus a
   model-on coherence gate (degenerate repetition is the broken-routing
   signature).

7. **DFlash M63.5 falls out for free.** The capture seam is arch-general
   (captures post-layer `h`; the MoE layer keeps the same signature), the drafter
   is qwen3-attention (ported M63.1), and the registry pair + dispatch already
   exist. M64.4 = pull the pair + run the existing bit-identical/acceptance gates;
   no new engine code.

## Rejected alternatives

- **Re-implement the expert gather-matmul** instead of reusing `SwitchGLU` —
  rejected: the substrate already has a quantization-aware `SwitchGLU` /
  `QuantizedSwitchLinear` used by every in-tree MoE arch; duplicating it adds risk
  with no benefit.
- **Athena-side vendor of the Gemma4 MoE forward** (Whisper/Sortformer style) to
  keep the substrate pristine — rejected for the same reason as ADR 001 §Rejected:
  it duplicates a large, already-vendored-and-fixed forward and risks numeric
  divergence from the validated dense path. The gated additive delta is smaller.
- **Port the fused `gate_up_proj` path / use `FusedGateUpSwitchGLU`** — rejected
  for this target: the mlx-community 4-bit checkpoint ships experts pre-split, so
  a plain `SwitchGLU` maps directly and no split is needed.
- **Defer until a non-quantized or differently-packed checkpoint exists** —
  rejected: the 4-bit mixed-quant checkpoint is the shipping target and the
  substrate already supports its quant scheme.

## Consequences

- The substrate path-dep fork grows by one additive MoE block on Gemma4 (tracked;
  dense paths byte-unchanged). Athena reproducibility continues to depend on the
  substrate clone state.
- A second large resident target (the 26B-A4B) and its drafter come under the M5
  governor / M41/M42 lifecycle machinery — same as the 31B DFlash pair.
- ADR 001's M63.5 deferral is unblocked by this work (flipped to shipped in M64.4).
- CLAUDE.md "ADRs" gains a pointer to this ADR once ratified.

## Implementation plan (sliced — conventions: substrate commit + Athena
`appVersion` bump IN the slice commit in BOTH `Sources/athena/Athena.swift` and
`clients/Sources/athena/Athena.swift`; direct-to-main commit + annotated semantic
tag per slice; xcodebuild not swift build for MLX; `graphify update .` after code;
error envelope + passive-oracle unchanged; **dense Gemma4 path byte-unchanged,
gated on `enableMoeBlock`**). Current `appVersion` = 0.10.112.

- **M64.0 — Feasibility spike (read-only, no code, no bump). DONE — GO.** Both
  named risks resolved in favor; port surface pinned (this ADR).
- **M64.1 — Substrate MoE port, gated off (v0.10.113).** Config fields +
  `Router`/`GeGLU`/`Experts` + the three extra norms + hybrid `DecoderLayer`
  forward, all gated on `enableMoeBlock` (default false). **Gate = dense
  byte-unchanged**: bit-identical greedy A/B on `gemma-4-12B-it-8bit` + structural
  review (new paths unreachable when off). Substrate commit; Athena bump/tag.
- **M64.2 — Load + Python-parity gate (v0.10.114).** Pull
  `gemma-4-26b-a4b-it-4bit`; empirically confirm the mixed-quant load (finding 1);
  dumped-reference forward parity (logits + selected hidden over a short token
  sequence) asserted bf16-equal vs Python `mlx-lm`. MoE-numerics correctness gate;
  any port fixes land here. Athena bump/tag.
- **M64.3 — e2e coherence + validated tier (v0.10.115).** Model-on governed
  generation coherence gate; `athena show` capability line; confirm `gemma4`
  validated tier. Athena bump/tag.
- **M64.4 — Unblocked M63.5: DFlash on 26B-A4B (v0.10.116).** Pull
  `z-lab/gemma-4-26B-A4B-it-DFlash`; run the existing
  `DFlashEngineParityTests.testDFlashBitIdenticalToGreedy` + acceptance-observer
  gates on the MoE target; OpenAPI/doc note; flip ADR 001's M63.5 entry to
  shipped. Athena bump/tag.

## Status (2026-06-11) — Implemented

All four slices shipped (v0.10.113–116); the substrate Gemma4 MoE delta lives at
`../mlx-swift-lm @ ce29d00` (additive, dense path byte-unchanged). Gates passed:
M64.1 dense bit-identical A/B on `gemma-4-12B-it-8bit`; M64.2 Python-parity on
`gemma-4-26b-a4b-it-4bit` (next-token argmax matches, cosine 0.99967, all
mismatches verified bf16 near-ties, dense 31B as the methodology control); M64.3
model-on coherence ("Paris") + schema-guided JSON through the Athena path;
M64.4 DFlash on the 26B-A4B pair (lossless, bit-identical 64/64, acceptance
0.64). Both named spike risks held: mixed per-tensor quant loaded with no port
code, and the substrate divergence stayed bounded to the MoE block. ADR 001's
M63.5 is unblocked and shipped.

**Methodology note (carry forward):** Python-vs-Swift logit parity must use an
in-distribution (chat-templated) prompt. Synthetic OOD token ids sit on near-ties
everywhere and disagree even between two correct ports — the gate falsely failed
on the *validated* dense 31B until the prompt was made real.

## References

- ADR 001 — DFlash speculative decoding (M63); this ADR unblocks its M63.5.
- Upstream reference: `mlx-lm` `gemma4_text.py` (`Router`/`Experts`/`GeGLU`/
  hybrid `DecoderLayer`).
- In-tree templates: substrate `SwitchLayers.swift` (`SwitchGLU` /
  `QuantizedSwitchLinear`), `Qwen3MoE.swift` (router top-k idiom),
  `Load.swift` + `BaseConfiguration.swift` (per-layer quantization).
- Precedent for the tracked additive substrate-delta override of the
  pristine-substrate rule: M20 (TurboQuant), M63.2 (the DFlash capture seam).
