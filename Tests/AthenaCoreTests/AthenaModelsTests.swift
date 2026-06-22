import Foundation
import MLX
import XCTest

@testable import AthenaModels

/// Parity guard for the vendored Qwen3.5: the substrate's config Codable
/// was copied + type-renamed, so this asserts it still decodes a real
/// Qwen3.5 `config.json`. Pure (no MLX).
///
/// L11 (M70.3): this used to SKIP whenever no local mtp model was present, so
/// on a vanilla CI host the parity guard never ran and a Codable drift would
/// ship green. A real (weights-free) `config.json` is now checked in under
/// `Fixtures/`, so the guard always RUNS; a local checkpoint, if present, is
/// still preferred (catches a real-model drift the fixture wouldn't).
final class AthenaModelsConfigTests: XCTestCase {

    /// Prefer a local mtp checkpoint; else the checked-in fixture (a real,
    /// weights-free dense `qwen3_5` config with an MTP head). Source-relative
    /// path — `Fixtures/` is excluded from the bundle, mirroring the other
    /// fixture tests.
    private func qwen35ConfigURL() -> URL {
        let local = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(
                "mlx-models/Qwen3.5-2B-4bit-mtp/config.json")
        if FileManager.default.fileExists(atPath: local.path) { return local }
        return URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .appending(path: "Fixtures/qwen3_5_config.json")
    }

    func testVendoredConfigDecodesRealQwen35() throws {
        let data = try Data(contentsOf: qwen35ConfigURL())

        let cfg = try JSONDecoder().decode(
            AthenaQwen35Configuration.self, from: data)

        XCTAssertEqual(cfg.modelType, "qwen3_5")
        XCTAssertGreaterThan(cfg.textConfig.hiddenLayers, 0)
        XCTAssertGreaterThan(cfg.textConfig.hiddenSize, 0)
        XCTAssertGreaterThan(cfg.textConfig.vocabularySize, 0)
        // qwen3_5 is the hybrid GDN model: full-attention every N layers.
        XCTAssertGreaterThan(cfg.textConfig.fullAttentionInterval, 0)
        // The -mtp checkpoint declares an MTP head (M2.2a opt-in gate).
        XCTAssertGreaterThan(cfg.textConfig.mtpNumHiddenLayers, 0)
    }

    func testInlineConfigMtpDefaultsToZero() throws {
        let json = #"{"model_type":"qwen3_5","hidden_size":2048}"#
        let cfg = try JSONDecoder().decode(
            AthenaQwen35Configuration.self, from: Data(json.utf8))
        XCTAssertEqual(cfg.textConfig.mtpNumHiddenLayers, 0)
    }

    func testInlineTextConfigDefaultsApply() throws {
        // A bare text-config object: absent keys fall back to defaults
        // (verifies the renamed Codable keeps its default-tolerant decode).
        let json = #"{"model_type":"qwen3_5","hidden_size":2048}"#
        let cfg = try JSONDecoder().decode(
            AthenaQwen35Configuration.self, from: Data(json.utf8))
        XCTAssertEqual(cfg.modelType, "qwen3_5")
        XCTAssertEqual(cfg.textConfig.hiddenSize, 2048)
        XCTAssertEqual(cfg.textConfig.fullAttentionInterval, 4)  // default
    }
}

/// M50.4 — regression for the allocator-pool leak class M46.6 caught
/// in the embedder. The TriAttention scorer fires per attention layer
/// during eviction passes on long-context decode (every `divideLength`
/// tokens past `kvBudget`); without the end-of-call clear, per-layer
/// norm MLXArrays accumulate across the eviction batch. Pure-MLX
/// synthetic inputs, no model load — gated only by MLX/Metal.
final class TriAttentionScorerMemoryRegressionTests: XCTestCase {

    func testScoreKeysPoolStaysBoundedAcrossManyCalls() throws {
        guard
            ProcessInfo.processInfo.environment["ATHENA_RUN_MODEL_TESTS"]
                == "1"
        else { throw XCTSkip("set ATHENA_RUN_MODEL_TESTS=1 (needs MLX/Metal)") }

        // Representative shape: [B=1, kvHeads=8, seqLen=1024, headDim=128]
        // — close to a real per-layer KV slice on a mid-sized model.
        // Plain float fill avoids an MLXRandom dependency in the test
        // target (mirrors WeSpeakerTests' approach).
        let n = 1 * 8 * 1024 * 128
        var rng = SystemRandomNumberGenerator()
        var raw = [Float](repeating: 0, count: n)
        for i in 0..<n {
            raw[i] = Float(Int(rng.next() % 1000)) / 1000.0 - 0.5
        }
        let keys = MLXArray(raw, [1, 8, 1024, 128])

        // Warmup so first-call lazy allocations settle.
        _ = TriAttentionScorer.scoreKeys(keys, aggregation: .mean)
        MLX.Memory.clearCache()
        let baseline = MLX.Memory.cacheMemory

        // 32 scorer calls = an eviction sweep over ~32 layers.
        for _ in 0..<32 {
            _ = TriAttentionScorer.scoreKeys(keys, aggregation: .mean)
        }

        let after = MLX.Memory.cacheMemory
        // Without M50.4's clear, the pool scales with the per-layer
        // norm-tensor footprint × 32.
        let ceiling = 64 * 1024 * 1024
        XCTAssertLessThan(
            after - baseline, ceiling,
            "MLX cache pool drifted \(after - baseline) bytes "
            + "above baseline after 32 scoreKeys calls (M50.4 leak)")
    }
}
