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
}
