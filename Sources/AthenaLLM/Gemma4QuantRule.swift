import Foundation

/// M72 — the single source of truth for how `athena convert` quantizes a
/// gemma-4 (VLM / MoE) checkpoint. The SAME rule drives the convert quantize
/// closure AND the emitted per-layer `quantization` config, so the saved
/// weights and the config can never disagree. Pure (no MLX) → unit-testable
/// under `swift test`.
///
/// Scheme pinned to `mlx-community/gemma-4-26b-a4b-it-4bit` (quant config +
/// `.scales` inventory, verified 2026-06-17):
///
/// - The image/audio **encoder towers** (`vision_tower`, `audio_tower`) stay
///   **full-precision** — no `.scales`, so the substrate's `.scales`-driven
///   loader (`MLXLMCommon/Load.swift`) leaves them unquantized. The multimodal
///   *projections* (`embed_vision` / `embed_audio`) are NOT encoder towers and
///   ARE quantized at the global bits.
/// - The per-layer **dense** `…mlp.{gate,up,down}_proj` and `…router.proj`
///   take an 8-bit override.
/// - Everything else quantizable — the MoE experts
///   (`experts.switch_glu.*`), `self_attn.{q,k,v,o}_proj`, `embed_tokens`,
///   `embed_vision.embedding_projection` — takes the global (e.g. 4-bit)
///   setting.
///
/// The encoder-tower skip is checked FIRST, so a tower's own `mlp.*` sublayers
/// are full-precision, never mistaken for an 8-bit language layer. The audio
/// prefix is listed now (forward-compat) even though audio is not yet served.
public struct Gemma4QuantRule: Sendable, Equatable {
    /// Quantization group size (e.g. 64).
    public let groupSize: Int
    /// Global bits for the language model (e.g. 4).
    public let bits: Int
    /// Per-layer override bits for the dense `mlp` + `router.proj` (e.g. 8).
    public let overrideBits: Int
    /// ADR 012 amendment (2026-06-17) — apply the per-layer 8-bit override
    /// ONLY for `gemma4`. The override pattern (`.mlp.{…}_proj`) is the
    /// Gemma-4 reference recipe, where it hits the *few* dense `mlp` layers
    /// among MoE `experts.switch_glu.*` (no `.mlp.`, stay 4-bit). On a
    /// fully-dense arch (`qwen3_5`) it would catch EVERY `mlp` layer; on a
    /// different MoE naming (`qwen3_5_moe`, experts `mlp.switch_mlp.*`) it
    /// would catch the EXPERT bulk — both inflating a "4-bit" convert toward
    /// 8-bit. `false` ⇒ quantize uniformly at the global `bits` (the
    /// encoder-tower skip still applies, so vision stays full-precision).
    public let mixedPrecision: Bool

    public init(
        groupSize: Int = 64, bits: Int, overrideBits: Int = 8,
        mixedPrecision: Bool = true
    ) {
        self.groupSize = groupSize
        self.bits = bits
        self.overrideBits = overrideBits
        self.mixedPrecision = mixedPrecision
    }

    /// Whether the Gemma-4 mixed 8/4 scheme applies to a checkpoint of this
    /// `model_type`. The scheme reproduces a specific published Gemma-4 quant
    /// and is NOT a general quantizer, so it is gated to `gemma4`; every other
    /// arch quantizes uniformly at the global bits. Single source of the arch
    /// key, shared by `ModelConvert` and the tests.
    public static func appliesMixedPrecision(modelType: String?) -> Bool {
        modelType == "gemma4"
    }

    /// `(groupSize, bits)` for a quantizable layer at `path`; `nil` ⇒ SKIP
    /// (leave full-precision, write no `.scales`).
    public func quantization(forPath path: String)
        -> (groupSize: Int, bits: Int)?
    {
        if Self.isEncoderTower(path) { return nil }
        if mixedPrecision, Self.isOverrideLayer(path) {
            return (groupSize, overrideBits)
        }
        return (groupSize, bits)
    }

    /// True for the image/audio ENCODER towers (full-precision). Matches a
    /// tower's whole subtree, so e.g. `vision_tower.encoder.layers.0.mlp.\
    /// down_proj` is skipped here BEFORE the 8-bit rule can see it.
    public static func isEncoderTower(_ path: String) -> Bool {
        path.contains("vision_tower") || path.contains("audio_tower")
    }

    /// True for the per-layer dense `mlp.{gate,up,down}_proj` and
    /// `router.proj` that take the 8-bit override. Excludes the MoE experts
    /// (`experts.switch_glu.*` contains no `.mlp.` and does not end in
    /// `.router.proj`), which fall through to the global bits.
    public static func isOverrideLayer(_ path: String) -> Bool {
        if path.hasSuffix(".router.proj") { return true }
        if path.contains(".mlp.") {
            return path.hasSuffix(".gate_proj")
                || path.hasSuffix(".up_proj")
                || path.hasSuffix(".down_proj")
        }
        return false
    }

    /// The per-module override entries `athena convert` must write into the
    /// converted `config.json`: every quantized module whose bits differ from
    /// the global `bits` (i.e. the 8-bit dense `mlp` + `router.proj`). Driven
    /// by `quantization(forPath:)`, so the emitted config stays in lock-step
    /// with what convert actually quantized. Encoder-tower modules never
    /// appear — they are full-precision, so they are not in `modules` (no
    /// `.scales`).
    public func overrides(forModules modules: [String])
        -> [(path: String, groupSize: Int, bits: Int)]
    {
        modules.compactMap { path in
            guard let q = quantization(forPath: path), q.bits != bits else {
                return nil
            }
            return (path, q.groupSize, q.bits)
        }
    }
}
