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

    public init(groupSize: Int = 64, bits: Int, overrideBits: Int = 8) {
        self.groupSize = groupSize
        self.bits = bits
        self.overrideBits = overrideBits
    }

    /// `(groupSize, bits)` for a quantizable layer at `path`; `nil` ⇒ SKIP
    /// (leave full-precision, write no `.scales`).
    public func quantization(forPath path: String)
        -> (groupSize: Int, bits: Int)?
    {
        if Self.isEncoderTower(path) { return nil }
        if Self.isOverrideLayer(path) { return (groupSize, overrideBits) }
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
