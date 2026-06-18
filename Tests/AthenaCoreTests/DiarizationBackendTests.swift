import Foundation
import XCTest

@testable import AthenaCore

/// Pure diarization-backend classification (ADR 018 / M74). No MLX — the
/// detector is MLX-free so it always runs in CI (ADR 009). Pins the decision
/// table that routes the `diarization` slot to the Sortformer engine vs the
/// pyannote segmentation engine, and that the `method` selector can be matched
/// against a resident model's backend.
final class DiarizationBackendTests: XCTestCase {

    func testSortformerType() {
        XCTAssertEqual(
            DiarizationBackend.classify(modelType: "sortformer"),
            .sortformer)
    }

    func testPyannoteCanonicalMirrorType() {
        // aufklarer/Pyannote-Segmentation-MLX ships `model_type:
        // "pyannote-segmentation"`.
        XCTAssertEqual(
            DiarizationBackend.classify(modelType: "pyannote-segmentation"),
            .pyannoteSegmentation)
    }

    func testPyannoteUpstreamAndUnderscoreAliases() {
        for t in ["pyannet", "pyannote_segmentation", "PyanNet"] {
            XCTAssertEqual(
                DiarizationBackend.classify(modelType: t),
                .pyannoteSegmentation, "for \(t)")
        }
    }

    func testCaseInsensitive() {
        XCTAssertEqual(
            DiarizationBackend.classify(modelType: "SORTFORMER"),
            .sortformer)
    }

    func testNilOrEmptyIsUnknown() {
        XCTAssertEqual(DiarizationBackend.classify(modelType: nil), .unknown)
        XCTAssertEqual(DiarizationBackend.classify(modelType: ""), .unknown)
        XCTAssertEqual(DiarizationBackend.classify(modelType: "  "), .unknown)
    }

    func testUnrecognizedTypeIsUnknown() {
        XCTAssertEqual(
            DiarizationBackend.classify(modelType: "whisper"), .unknown)
    }

    // MARK: config.json reader (real temp dirs)

    private func tempDir(config: String?) throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("diar-backend-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: dir, withIntermediateDirectories: true)
        if let config {
            try config.data(using: .utf8)!.write(
                to: dir.appendingPathComponent("config.json"))
        }
        return dir
    }

    func testDetectReadsModelTypeFromConfig() throws {
        let dir = try tempDir(config: #"{"model_type":"pyannote-segmentation"}"#)
        defer { try? FileManager.default.removeItem(at: dir) }
        XCTAssertEqual(DiarizationBackend.detect(in: dir), .pyannoteSegmentation)
    }

    func testDetectSortformerConfig() throws {
        let dir = try tempDir(config: #"{"model_type":"sortformer","num_speakers":4}"#)
        defer { try? FileManager.default.removeItem(at: dir) }
        XCTAssertEqual(DiarizationBackend.detect(in: dir), .sortformer)
    }

    func testDetectMissingConfigIsUnknown() throws {
        let dir = try tempDir(config: nil)
        defer { try? FileManager.default.removeItem(at: dir) }
        XCTAssertEqual(DiarizationBackend.detect(in: dir), .unknown)
    }

    func testDetectMalformedConfigIsUnknown() throws {
        let dir = try tempDir(config: "{not json")
        defer { try? FileManager.default.removeItem(at: dir) }
        XCTAssertEqual(DiarizationBackend.detect(in: dir), .unknown)
    }
}
