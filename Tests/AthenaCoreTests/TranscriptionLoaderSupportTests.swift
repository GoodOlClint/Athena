import AthenaCore
import Foundation
import XCTest

@testable import AthenaTranscription

/// Regression pins for the ADR 021 S2 wiring: the transcription loader consults
/// the shared `ModelSupport` predicate and refuses an unloadable packaging with
/// a cause-naming **4xx** before dispatching into an engine loader that would
/// fail deep with an opaque 500. These run in the logic tier — the gate throws
/// from config metadata alone, so no MLX/Metal is touched.
final class TranscriptionLoaderSupportTests: XCTestCase {

    /// A model store with one entry `<name>/config.json` holding `json`.
    private func store(name: String, config json: String) throws -> URL {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("tls-\(UUID().uuidString)")
        let dir = root.appendingPathComponent(name, isDirectory: true)
        try FileManager.default.createDirectory(
            at: dir, withIntermediateDirectories: true)
        try json.data(using: .utf8)!
            .write(to: dir.appendingPathComponent("config.json"))
        return root
    }

    private func loadError(name: String, config: String) async -> Error? {
        do {
            let root = try store(name: name, config: config)
            defer { try? FileManager.default.removeItem(at: root) }
            let mod = MLXTranscriptionModule(
                modelIds: [name], modelStoreRoot: root)
            try await mod.rebind(to: name)
            return nil
        } catch {
            return error
        }
    }

    /// The M76 field incident: a transformers-format `parakeet_tdt` checkpoint
    /// (no `joint.vocabulary`) must be a 400 naming the missing field — NOT the
    /// opaque 500 the deep ParakeetLoader produced.
    func testTransformersParakeetWithoutJointIs400NotDeep500() async throws {
        let err = await loadError(
            name: "parakeet-transformers",
            config:
                #"{"model_type":"parakeet_tdt","architectures":["ParakeetForTDT"],"vocab_size":8192}"#
        )
        guard let e = err as? AthenaError else {
            return XCTFail("expected AthenaError, got \(String(describing: err))")
        }
        XCTAssertEqual(e.code, "unsupported_transcription_arch")
        XCTAssertEqual(e.httpStatus, 400)
        XCTAssertTrue(
            e.message.lowercased().contains("joint.vocabulary"),
            "message must name the missing structural field: \(e.message)")
        // Guidance rule (ADR 021 D5): no hard-coded model id / HF repo. The
        // model name here is bare (no slash), so any slash would be a repo id.
        XCTAssertFalse(
            e.message.contains("/"),
            "error must not embed an org/name repo id: \(e.message)")
        for repo in [
            "nvidia/parakeet-tdt-0.6b-v3", "mlx-community/parakeet-tdt-0.6b-v3",
        ] {
            XCTAssertFalse(e.message.contains(repo), e.message)
        }
    }

    /// A non-large-v3 Whisper vocab is refused pre-load with a 400 (was a
    /// post-load 500), naming the 51866 requirement.
    func testWhisperWrongVocabIs400() async throws {
        let err = await loadError(
            name: "whisper-bad-vocab",
            config: #"{"model_type":"whisper","n_vocab":51865}"#)
        guard let e = err as? AthenaError else {
            return XCTFail("expected AthenaError, got \(String(describing: err))")
        }
        XCTAssertEqual(e.code, "unsupported_transcription_arch")
        XCTAssertEqual(e.httpStatus, 400)
        XCTAssertTrue(e.message.contains("51866"), e.message)
    }

    /// A non-ASR checkpoint in the transcription slot → cause-naming 400
    /// ("neither Whisper nor Parakeet"), not a deep loader 500.
    func testNonAsrModelIs400() async throws {
        let err = await loadError(
            name: "an-llm", config: #"{"model_type":"qwen3_5"}"#)
        guard let e = err as? AthenaError else {
            return XCTFail("expected AthenaError, got \(String(describing: err))")
        }
        XCTAssertEqual(e.code, "unsupported_transcription_arch")
        XCTAssertEqual(e.httpStatus, 400)
    }
}
