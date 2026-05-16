import Foundation
import XCTest

@testable import AthenaModels

/// Parity guard for the vendored Qwen3.5: the substrate's config Codable
/// was copied + type-renamed, so this asserts it still decodes a real
/// Qwen3.5 `config.json`. Pure (no MLX); skips when the model isn't on
/// this machine so CI without the external SSD stays green.
final class AthenaModelsConfigTests: XCTestCase {

    func testVendoredConfigDecodesRealQwen35() throws {
        // The known-good default checkpoint (substrate-vetted, validated
        // coherent). Skips when the external SSD isn't mounted.
        let configURL = URL(
            fileURLWithPath:
                "/Volumes/SB-XTM5/mlx-models/Qwen3.5-2B-4bit/config.json")
        guard FileManager.default.fileExists(atPath: configURL.path) else {
            throw XCTSkip("Qwen3.5 model not present on this machine")
        }
        let data = try Data(contentsOf: configURL)

        let cfg = try JSONDecoder().decode(
            AthenaQwen35Configuration.self, from: data)

        XCTAssertEqual(cfg.modelType, "qwen3_5")
        XCTAssertGreaterThan(cfg.textConfig.hiddenLayers, 0)
        XCTAssertGreaterThan(cfg.textConfig.hiddenSize, 0)
        XCTAssertGreaterThan(cfg.textConfig.vocabularySize, 0)
        // qwen3_5 is the hybrid GDN model: full-attention every N layers.
        XCTAssertGreaterThan(cfg.textConfig.fullAttentionInterval, 0)
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
