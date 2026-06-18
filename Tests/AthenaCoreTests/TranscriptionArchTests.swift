import Foundation
import XCTest

@testable import AthenaCore

/// Pure (MLX-free) transcription-engine routing (ADR 020). Always runs in CI
/// (ADR 009). Pins the positive router: Whisper checkpoints → `.whisper`,
/// Parakeet checkpoints (both config shapes) → `.parakeet`, everything else →
/// `.unsupported`. Replaces the v0.10.170 denylist semantics.
final class TranscriptionArchTests: XCTestCase {

    // MARK: classify — the pure decision

    func testWhisperByModelType() {
        XCTAssertEqual(
            TranscriptionArch.classify(.init(modelType: "whisper")), .whisper)
    }

    func testWhisperByArchitecture() {
        XCTAssertEqual(
            TranscriptionArch.classify(
                .init(architectures: ["WhisperForConditionalGeneration"])),
            .whisper)
    }

    /// transformers-style Parakeet (e.g. nvidia/parakeet-tdt-0.6b-v3):
    /// top-level `model_type` + `architectures`.
    func testParakeetTransformersConfig() {
        XCTAssertEqual(
            TranscriptionArch.classify(
                .init(
                    modelType: "parakeet_tdt",
                    architectures: ["ParakeetForTDT"])),
            .parakeet)
    }

    /// NeMo / MLX-style Parakeet (mlx-community/parakeet-tdt-0.6b-v3 — the
    /// port's load source): NO top-level `model_type`/`architectures`; the
    /// signal is the NeMo `target` + `decoding.model_type: tdt`.
    func testParakeetNeMoConfig() {
        XCTAssertEqual(
            TranscriptionArch.classify(
                .init(
                    target:
                        "nemo.collections.asr.models.rnnt_bpe_models"
                        + ".EncDecRNNTBPEModel",
                    decodingModelType: "tdt")),
            .parakeet)
    }

    /// The NeMo `target` alone names RNN-T/Conformer (not "parakeet"/"tdt"); the
    /// decisive Parakeet signal is `decoding.model_type: tdt`. Without it, a
    /// bare RNN-T config is not claimed by the Parakeet engine.
    func testNeMoTargetWithoutTDTIsUnsupported() {
        XCTAssertEqual(
            TranscriptionArch.classify(
                .init(
                    target:
                        "nemo.collections.asr.models.EncDecCTCModelBPE")),
            .unsupported)
    }

    func testBertIsUnsupported() {
        XCTAssertEqual(
            TranscriptionArch.classify(
                .init(modelType: "bert", architectures: ["BertModel"])),
            .unsupported)
    }

    func testOtherAsrFamiliesUnsupported() {
        for t in ["canary", "wav2vec2", "hubert", "wavlm"] {
            XCTAssertEqual(
                TranscriptionArch.classify(.init(modelType: t)), .unsupported,
                "\(t) should route to .unsupported")
        }
    }

    func testEmptyConfigIsUnsupported() {
        XCTAssertEqual(TranscriptionArch.classify(.init()), .unsupported)
    }

    // MARK: detect — config.json on disk

    private func writeConfig(_ json: String) throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("ta-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: dir, withIntermediateDirectories: true)
        try json.data(using: .utf8)!
            .write(to: dir.appendingPathComponent("config.json"))
        return dir
    }

    func testDetectWhisperDir() throws {
        let dir = try writeConfig(#"{"model_type":"whisper"}"#)
        defer { try? FileManager.default.removeItem(at: dir) }
        XCTAssertEqual(TranscriptionArch.detect(in: dir), .whisper)
    }

    func testDetectParakeetNeMoDir() throws {
        // Mirrors the real mlx-community/parakeet-tdt-0.6b-v3 config shape.
        let dir = try writeConfig(
            #"{"target":"nemo.collections.asr.models.rnnt_bpe_models.EncDecRNNTBPEModel","decoding":{"model_type":"tdt","strategy":"greedy_batch"}}"#
        )
        defer { try? FileManager.default.removeItem(at: dir) }
        XCTAssertEqual(TranscriptionArch.detect(in: dir), .parakeet)
    }

    func testDetectParakeetTransformersDir() throws {
        let dir = try writeConfig(
            #"{"model_type":"parakeet_tdt","architectures":["ParakeetForTDT"]}"#)
        defer { try? FileManager.default.removeItem(at: dir) }
        XCTAssertEqual(TranscriptionArch.detect(in: dir), .parakeet)
    }

    func testDetectMissingConfigIsUnsupported() {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("ta-missing-\(UUID().uuidString)")
        XCTAssertEqual(TranscriptionArch.detect(in: dir), .unsupported)
    }
}
