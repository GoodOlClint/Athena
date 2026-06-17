import XCTest

@testable import AthenaLLM

/// M72 — the convert quant-rule, pinned to the real
/// `mlx-community/gemma-4-26b-a4b-it-4bit` path patterns (the `.scales`
/// inventory + quant-config overrides verified 2026-06-17). Pure, no MLX.
final class Gemma4QuantRuleTests: XCTestCase {
    private let rule = Gemma4QuantRule(groupSize: 64, bits: 4, overrideBits: 8)

    private func q(_ path: String) -> (groupSize: Int, bits: Int)? {
        rule.quantization(forPath: path)
    }

    // Encoder towers stay FULL-PRECISION (skip → nil).
    func testVisionTowerSkipped() {
        XCTAssertNil(q("vision_tower.encoder.layers.0.self_attn.q_proj"))
        // a tower's own mlp must be skipped, NOT taken as an 8-bit language mlp
        XCTAssertNil(q("vision_tower.encoder.layers.0.mlp.down_proj"))
    }

    func testAudioTowerSkippedForwardCompat() {
        XCTAssertNil(q("audio_tower.encoder.layers.0.mlp.up_proj"))
    }

    // Multimodal PROJECTIONS are quantized at the global bits (NOT skipped).
    func testEmbedVisionProjectionQuantizedGlobal() {
        XCTAssertEqual(q("embed_vision.embedding_projection")?.bits, 4)
    }

    // Per-layer dense mlp + router → 8-bit override.
    func testDenseMLPAndRouterAreEightBit() {
        for p in [
            "language_model.model.layers.0.mlp.gate_proj",
            "language_model.model.layers.7.mlp.up_proj",
            "language_model.model.layers.29.mlp.down_proj",
            "language_model.model.layers.3.router.proj",
        ] {
            XCTAssertEqual(q(p)?.bits, 8, "\(p) should be 8-bit")
            XCTAssertEqual(q(p)?.groupSize, 64)
        }
    }

    // MoE experts, attention, embeddings → global 4-bit.
    func testExpertsAttnEmbedAreGlobal() {
        for p in [
            "language_model.model.layers.0.experts.switch_glu.gate_proj",
            "language_model.model.layers.0.experts.switch_glu.down_proj",
            "language_model.model.layers.0.self_attn.q_proj",
            "language_model.model.layers.0.self_attn.o_proj",
            "language_model.model.embed_tokens",
        ] {
            XCTAssertEqual(q(p)?.bits, 4, "\(p) should be global 4-bit")
        }
    }

    // The experts switch_glu must NOT be mistaken for the 8-bit dense mlp.
    func testExpertsNotEightBit() {
        XCTAssertFalse(
            Gemma4QuantRule.isOverrideLayer(
                "language_model.model.layers.0.experts.switch_glu.gate_proj"))
        XCTAssertTrue(
            Gemma4QuantRule.isOverrideLayer(
                "language_model.model.layers.0.mlp.gate_proj"))
    }

    // Config emission: only the 8-bit modules (dense mlp + router) become
    // override entries; global-bit modules and (absent) towers do not.
    func testOverridesForConfigEmission() {
        let modules = [
            "language_model.model.layers.0.mlp.gate_proj",  // 8-bit → override
            "language_model.model.layers.0.mlp.up_proj",  // 8-bit → override
            "language_model.model.layers.0.router.proj",  // 8-bit → override
            "language_model.model.layers.0.experts.switch_glu.gate_proj",  // 4-bit
            "language_model.model.layers.0.self_attn.q_proj",  // 4-bit
            "embed_vision.embedding_projection",  // 4-bit
        ]
        let ov = rule.overrides(forModules: modules)
        XCTAssertEqual(Set(ov.map(\.path)), [
            "language_model.model.layers.0.mlp.gate_proj",
            "language_model.model.layers.0.mlp.up_proj",
            "language_model.model.layers.0.router.proj",
        ])
        XCTAssertTrue(ov.allSatisfy { $0.bits == 8 && $0.groupSize == 64 })
    }

    // MARK: - ADR 012 amendment — arch-gated mixed precision

    /// The 8/4 mixed scheme applies ONLY to gemma4; every other arch (and an
    /// unknown/absent model_type) quantizes uniformly.
    func testAppliesMixedPrecisionGatedToGemma4() {
        XCTAssertTrue(Gemma4QuantRule.appliesMixedPrecision(modelType: "gemma4"))
        for other in ["qwen3_5", "qwen3_5_moe", "gemma4_text", "llama", nil] {
            XCTAssertFalse(
                Gemma4QuantRule.appliesMixedPrecision(modelType: other),
                "\(other ?? "nil") must not get the gemma4 mixed scheme")
        }
    }

    /// With `mixedPrecision: false` (any non-gemma4 arch) there is NO 8-bit
    /// override: a fully-dense Qwen's every `mlp.*_proj` AND a Qwen-MoE's
    /// routed experts (`mlp.switch_mlp.*`) — the two paths that bloated the
    /// "4-bit" converts — all quantize at the GLOBAL bits. Encoder towers stay
    /// skipped (vision full-precision for any arch), and no override entries
    /// are emitted into the config.
    func testUniformQuantForNonGemma4() {
        let uniform = Gemma4QuantRule(
            groupSize: 64, bits: 4, overrideBits: 8, mixedPrecision: false)
        func u(_ p: String) -> (groupSize: Int, bits: Int)? {
            uniform.quantization(forPath: p)
        }
        // Dense Qwen MLP — would be 8-bit under the gemma4 override; now 4-bit.
        XCTAssertEqual(u("language_model.model.layers.0.mlp.down_proj")?.bits, 4)
        // Qwen-MoE routed EXPERTS — the bulk; must be global 4-bit, not 8.
        XCTAssertEqual(
            u("language_model.model.layers.0.mlp.switch_mlp.down_proj")?.bits, 4)
        XCTAssertEqual(
            u("language_model.model.layers.0.mlp.shared_expert.up_proj")?.bits, 4)
        // Attention/embeds → 4-bit; encoder tower → still skipped.
        XCTAssertEqual(u("language_model.model.layers.0.self_attn.q_proj")?.bits, 4)
        XCTAssertNil(u("vision_tower.encoder.layers.0.mlp.down_proj"))
        // No per-module overrides emitted (everything quantizable == global).
        let ov = uniform.overrides(forModules: [
            "language_model.model.layers.0.mlp.down_proj",
            "language_model.model.layers.0.mlp.switch_mlp.gate_proj",
            "language_model.model.layers.0.self_attn.q_proj",
        ])
        XCTAssertTrue(ov.isEmpty, "uniform convert writes no 8-bit overrides")
    }
}
