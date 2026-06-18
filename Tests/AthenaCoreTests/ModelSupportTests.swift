import Foundation
import XCTest

@testable import AthenaCore

/// The unified model-support predicate (ADR 021 / M77). MLX-free, so it always
/// runs in CI (ADR 008/009). Pins the full cross-modality classification +
/// loadability matrix that the three consumers (loaders / convert / pull
/// preflight) share, plus the binding guidance rule: every `.unsupported`
/// reason/guidance names the structural requirement and hard-codes **no** model
/// id / HF repo.
final class ModelSupportTests: XCTestCase {

    // MARK: helpers

    private func probe(
        modelType: String? = nil,
        architectures: [String] = [],
        target: String? = nil,
        decodingModelType: String? = nil,
        hasVisionConfig: Bool = false,
        stMarkers: Bool = false,
        nVocab: Int? = nil,
        hasJointVocabulary: Bool = false
    ) -> ModelSupport.Probe {
        ModelSupport.Probe(
            info: ModelConfigInfo(
                modelType: modelType, hasVisionConfig: hasVisionConfig),
            transcription: .init(
                modelType: modelType, architectures: architectures,
                target: target, decodingModelType: decodingModelType),
            hasSentenceTransformerMarkers: stMarkers,
            whisperNVocab: nVocab,
            hasJointVocabulary: hasJointVocabulary)
    }

    // MARK: transcription — Whisper

    func testWhisperGoodVocabIsLoadable() {
        let s = ModelSupport.classify(
            probe(modelType: "whisper", nVocab: 51866))
        XCTAssertEqual(s.modality, .transcription(.whisper))
        XCTAssertEqual(s.loadability, .loadable)
    }

    func testWhisperWrongVocabIsUnsupported() {
        let s = ModelSupport.classify(
            probe(modelType: "whisper", nVocab: 51865))
        XCTAssertEqual(s.modality, .transcription(.whisper))
        guard case let .unsupported(reason, guidance) = s.loadability else {
            return XCTFail("expected .unsupported, got \(s.loadability)")
        }
        XCTAssertTrue(reason.contains("51866"), reason)
        XCTAssertTrue(reason.contains("51865"), reason)
        XCTAssertFalse(guidance.isEmpty)
    }

    func testWhisperMissingVocabIsUnsupported() {
        let s = ModelSupport.classify(probe(modelType: "whisper"))
        XCTAssertEqual(s.modality, .transcription(.whisper))
        if case .unsupported = s.loadability {} else {
            XCTFail("missing n_vocab must be .unsupported")
        }
    }

    // MARK: transcription — Parakeet (the M76 incident case)

    /// NeMo-format export (mlx-community/parakeet-tdt-0.6b-v3 shape): `target` +
    /// `decoding.model_type: tdt` + a non-empty `joint.vocabulary` → loadable.
    func testParakeetNeMoWithJointIsLoadable() {
        let s = ModelSupport.classify(
            probe(
                target:
                    "nemo.collections.asr.models.rnnt_bpe_models"
                    + ".EncDecRNNTBPEModel",
                decodingModelType: "tdt", hasJointVocabulary: true))
        XCTAssertEqual(s.modality, .transcription(.parakeet))
        XCTAssertEqual(s.loadability, .loadable)
    }

    /// transformers-format checkpoint (nvidia/parakeet-tdt-0.6b-v3 shape):
    /// `model_type: parakeet_tdt` but NO `joint.vocabulary` → the field
    /// incident. Classifies as Parakeet, refused at the packaging layer.
    func testParakeetTransformersWithoutJointIsUnsupported() {
        let s = ModelSupport.classify(
            probe(
                modelType: "parakeet_tdt",
                architectures: ["ParakeetForTDT"],
                hasJointVocabulary: false))
        XCTAssertEqual(s.modality, .transcription(.parakeet))
        guard case let .unsupported(reason, _) = s.loadability else {
            return XCTFail("expected .unsupported, got \(s.loadability)")
        }
        XCTAssertTrue(
            reason.lowercased().contains("joint.vocabulary"), reason)
    }

    // MARK: diarization

    func testSortformerIsDiarizationLoadable() {
        let s = ModelSupport.classify(probe(modelType: "sortformer"))
        XCTAssertEqual(s.modality, .diarization(.sortformer))
        XCTAssertEqual(s.loadability, .loadable)
    }

    func testPyannoteIsDiarizationLoadable() {
        let s = ModelSupport.classify(
            probe(modelType: "pyannote-segmentation"))
        XCTAssertEqual(s.modality, .diarization(.pyannoteSegmentation))
        XCTAssertEqual(s.loadability, .loadable)
    }

    // MARK: speaker-embedding

    func testWeSpeakerIsSpeakerEmbeddingLoadable() {
        let s = ModelSupport.classify(
            probe(modelType: "wespeaker-resnet34-lm"))
        XCTAssertEqual(s.modality, .speakerEmbedding)
        XCTAssertEqual(s.loadability, .loadable)
    }

    // MARK: generative / vision / embedding (ModelClass delegation)

    func testGenerativeIsBestEffortUnknown() {
        // A named generative type classifies as `.llm`, but arch coverage is
        // inherited from the substrate — loadability is best-effort `.unknown`.
        let s = ModelSupport.classify(probe(modelType: "qwen3_5"))
        XCTAssertEqual(s.modality, .llm)
        XCTAssertEqual(s.loadability, .unknown)
    }

    func testVisionIsBestEffortUnknown() {
        let s = ModelSupport.classify(
            probe(modelType: "gemma4", hasVisionConfig: true))
        XCTAssertEqual(s.modality, .vision)
        XCTAssertEqual(s.loadability, .unknown)
    }

    func testEmbeddingByTypeIsLoadable() {
        let s = ModelSupport.classify(probe(modelType: "bert"))
        XCTAssertEqual(s.modality, .embedding)
        XCTAssertEqual(s.loadability, .loadable)
    }

    func testEmbeddingBySTMarkersIsLoadable() {
        // gemma3_text is also a generative arch; the ST markers make it
        // embedding (mirrors ModelClassTests).
        let s = ModelSupport.classify(
            probe(modelType: "gemma3_text", stMarkers: true))
        XCTAssertEqual(s.modality, .embedding)
        XCTAssertEqual(s.loadability, .loadable)
    }

    func testNoModelTypeIsUnsupported() {
        let s = ModelSupport.classify(probe())
        XCTAssertEqual(s.modality, .unsupported)
        if case .unsupported = s.loadability {} else {
            XCTFail("empty config must be .unsupported")
        }
    }

    // MARK: precedence — a named audio type never mis-files as generative

    func testParakeetTypeBeatsGenerativeClassification() {
        // `parakeet_tdt` is a named model_type ModelClass would call generative;
        // the transcription detector must claim it first.
        let s = ModelSupport.classify(
            probe(modelType: "parakeet_tdt", hasJointVocabulary: true))
        XCTAssertEqual(s.modality, .transcription(.parakeet))
    }

    // MARK: guidance rule (ADR 021 decision 5) — no hard-coded model id / repo

    /// Every `.unsupported` reason+guidance must name the structural
    /// requirement and contain no HF repo id. A repo id is always `org/name`,
    /// so a slash-free string cannot embed one — the strictest possible pin.
    func testNoUnsupportedMessageHardCodesARepoId() {
        let knownRepoIds = [
            "nvidia/parakeet-tdt-0.6b-v3",
            "mlx-community/parakeet-tdt-0.6b-v3",
            "mlx-community/whisper-large-v3-turbo",
            "aufklarer/WeSpeaker-ResNet34-LM-MLX",
        ]
        // Exercise every path that yields `.unsupported`.
        let verdicts: [ModelSupport] = [
            .classify(probe(modelType: "whisper", nVocab: 51865)),
            .classify(probe(modelType: "whisper")),
            .classify(
                probe(modelType: "parakeet_tdt", hasJointVocabulary: false)),
            .classify(probe()),
        ]
        for v in verdicts {
            guard case let .unsupported(reason, guidance) = v.loadability else {
                return XCTFail("expected .unsupported for \(v.modality)")
            }
            for s in [reason, guidance] {
                XCTAssertFalse(
                    s.contains("/"),
                    "guidance must be slash-free (no org/name repo id): \(s)")
                for repo in knownRepoIds {
                    XCTAssertFalse(
                        s.contains(repo), "must not hard-code \(repo): \(s)")
                }
                XCTAssertFalse(s.isEmpty)
            }
        }
    }

    // MARK: detect(in:) — config.json on disk

    private func writeConfig(_ json: String) throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("ms-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: dir, withIntermediateDirectories: true)
        try json.data(using: .utf8)!
            .write(to: dir.appendingPathComponent("config.json"))
        return dir
    }

    func testDetectWhisperDir() throws {
        let dir = try writeConfig(
            #"{"model_type":"whisper","n_vocab":51866}"#)
        defer { try? FileManager.default.removeItem(at: dir) }
        let s = ModelSupport.detect(in: dir)
        XCTAssertEqual(s.modality, .transcription(.whisper))
        XCTAssertEqual(s.loadability, .loadable)
    }

    func testDetectParakeetTransformersDirIsUnsupported() throws {
        // The real nvidia transformers config: parakeet_tdt, no joint.
        let dir = try writeConfig(
            #"{"model_type":"parakeet_tdt","architectures":["ParakeetForTDT"],"vocab_size":8192}"#
        )
        defer { try? FileManager.default.removeItem(at: dir) }
        let s = ModelSupport.detect(in: dir)
        XCTAssertEqual(s.modality, .transcription(.parakeet))
        if case .unsupported = s.loadability {} else {
            XCTFail("transformers Parakeet (no joint.vocabulary) must refuse")
        }
    }

    func testDetectParakeetNeMoDirIsLoadable() throws {
        let dir = try writeConfig(
            #"{"target":"nemo.collections.asr.models.rnnt_bpe_models.EncDecRNNTBPEModel","decoding":{"model_type":"tdt"},"joint":{"vocabulary":["a","b","c"]}}"#
        )
        defer { try? FileManager.default.removeItem(at: dir) }
        let s = ModelSupport.detect(in: dir)
        XCTAssertEqual(s.modality, .transcription(.parakeet))
        XCTAssertEqual(s.loadability, .loadable)
    }

    func testDetectMissingConfigIsUnsupported() {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("ms-missing-\(UUID().uuidString)")
        XCTAssertEqual(ModelSupport.detect(in: dir).modality, .unsupported)
    }
}
