import Foundation
import XCTest

@testable import AthenaTranscription

/// Pure (MLX-free) detection of non-Whisper ASR architectures the
/// transcription engine can't load (e.g. Parakeet). Always runs in CI
/// (ADR 009). Pins that real Whisper checkpoints are never falsely rejected.
final class TranscriptionArchTests: XCTestCase {

    func testWhisperIsSupported() {
        XCTAssertFalse(
            TranscriptionArch.isUnsupported(
                modelType: "whisper",
                architectures: ["WhisperForConditionalGeneration"]))
    }

    func testMissingMetadataIsNotRejected() {
        // Unknown/absent metadata is left to the Whisper loader, not pre-judged.
        XCTAssertFalse(
            TranscriptionArch.isUnsupported(modelType: nil, architectures: []))
        XCTAssertFalse(
            TranscriptionArch.isUnsupported(modelType: "", architectures: []))
    }

    func testParakeetRejectedByModelType() {
        XCTAssertTrue(
            TranscriptionArch.isUnsupported(
                modelType: "parakeet", architectures: []))
    }

    func testRNNTAndTDTArchitecturesRejected() {
        XCTAssertTrue(
            TranscriptionArch.isUnsupported(
                modelType: nil, architectures: ["EncDecRNNTBPEModel"]))
        XCTAssertTrue(
            TranscriptionArch.isUnsupported(
                modelType: "tdt", architectures: []))
    }

    func testFastConformerAndNemoRejected() {
        XCTAssertTrue(
            TranscriptionArch.isUnsupported(
                modelType: "fastconformer", architectures: []))
        XCTAssertTrue(
            TranscriptionArch.isUnsupported(
                modelType: nil, architectures: ["EncDecCTCModelBPE", "nemo"]))
    }

    func testReadConfigFromDir() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("ta-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        try #"{"model_type":"parakeet","architectures":["EncDecRNNTBPEModel"]}"#
            .data(using: .utf8)!
            .write(to: dir.appendingPathComponent("config.json"))
        XCTAssertTrue(TranscriptionArch.isUnsupported(in: dir))
    }

    func testWhisperDirNotRejected() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("ta-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        try #"{"model_type":"whisper"}"#.data(using: .utf8)!
            .write(to: dir.appendingPathComponent("config.json"))
        XCTAssertFalse(TranscriptionArch.isUnsupported(in: dir))
    }
}
