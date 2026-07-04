import Foundation
import XCTest

import AthenaCore

/// Usability audit 2026-07-02 §4 — typed listing. Pins the fused-MTP probe
/// (weight-index scan, config-gated) and the ModelSupport wire projection that
/// the TYPE column reads. MLX-free (ADR 008/009); fixtures are written to temp
/// dirs so the probe exercises real config.json / index.json I/O.
final class MTPCheckpointTests: XCTestCase {
    private func makeCheckpoint(
        config: String, indexWeightKeys: [String]?
    ) throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("mtp-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(
            at: dir, withIntermediateDirectories: true)
        try config.write(
            to: dir.appendingPathComponent("config.json"),
            atomically: true, encoding: .utf8)
        if let keys = indexWeightKeys {
            let map = keys.map { "\"\($0)\": \"model.safetensors\"" }
                .joined(separator: ", ")
            try "{ \"weight_map\": { \(map) } }".write(
                to: dir.appendingPathComponent("model.safetensors.index.json"),
                atomically: true, encoding: .utf8)
        }
        return dir
    }

    /// The DoD discriminator: a `-mtp` build and a stock same-family checkpoint
    /// differ ONLY in whether the weight index carries `mtp.*` tensors → only
    /// `fused_mtp` flips. Mirrors the real converts, whose config drops
    /// `mtp_num_hidden_layers` and whose index prefix is `language_model.mtp.*`
    /// — the weight index is the authority, not config.
    func testFusedMTPVsStockDifferOnlyInWeightIndex() throws {
        let cfg = #"{ "model_type": "qwen3_5" }"#  // no mtp_num_hidden_layers
        let fused = try makeCheckpoint(
            config: cfg,
            indexWeightKeys: [
                "language_model.mtp.fc.weight", "lm_head.weight",
            ])
        let stock = try makeCheckpoint(
            config: cfg, indexWeightKeys: ["lm_head.weight"])
        defer {
            try? FileManager.default.removeItem(at: fused)
            try? FileManager.default.removeItem(at: stock)
        }
        XCTAssertTrue(MTPCheckpoint.hasFusedMTP(in: fused))
        XCTAssertFalse(MTPCheckpoint.hasFusedMTP(in: stock))
        // Both classify as a servable LLM — fused MTP is an attribute, not a
        // modality (audit §4).
        XCTAssertEqual(ModelSupport.detect(in: fused).modality, .llm)
        XCTAssertEqual(ModelSupport.detect(in: stock).modality, .llm)
    }

    /// Both the `gemma4_assistant` and the real `gemma4_unified_assistant`
    /// drafter model_types classify as `draft`, never llm/vision (DoD §4).
    func testAssistantVariantsClassifyAsDraft() throws {
        for mt in ["gemma4_assistant", "gemma4_unified_assistant"] {
            let dir = try makeCheckpoint(
                config: "{ \"model_type\": \"\(mt)\" }", indexWeightKeys: nil)
            defer { try? FileManager.default.removeItem(at: dir) }
            let s = ModelSupport.detect(in: dir)
            XCTAssertEqual(s.modality, .mtpDrafter, "\(mt)")
            XCTAssertTrue(s.isDraft, "\(mt)")
            XCTAssertEqual(s.wireModality, "draft", "\(mt)")
        }
    }
}
