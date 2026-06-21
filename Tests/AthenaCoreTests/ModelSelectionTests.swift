import Foundation
import XCTest

@testable import AthenaCore

/// ADR 026 — store-backed model selection. The MLX-free decision logic
/// (resolution, the ambiguity rule, store-class enumeration, the per-slot
/// modality acceptors) is unit-pinned here (ADR 008/009), exactly like the
/// `ModelSupport`/`ModelClass` detectors it composes.
final class ModelSelectionResolveTests: XCTestCase {
    private let store = ["BAAI/bge-small-en-v1.5", "Qwen/Qwen3-Embedding-4B"]

    // MARK: requested id present

    func testRequestedFullIdResolves() {
        XCTAssertEqual(
            ModelSelection.resolve(
                available: store, configuredDefault: nil,
                requested: "Qwen/Qwen3-Embedding-4B"),
            .resolved("Qwen/Qwen3-Embedding-4B"))
    }

    func testRequestedBareNameResolvesToCanonical() {
        // A bare store-dir basename resolves to the canonical stored id.
        XCTAssertEqual(
            ModelSelection.resolve(
                available: store, configuredDefault: nil,
                requested: "Qwen3-Embedding-4B"),
            .resolved("Qwen/Qwen3-Embedding-4B"))
    }

    func testRequestedCaseInsensitive() {
        XCTAssertEqual(
            ModelSelection.resolve(
                available: store, configuredDefault: nil,
                requested: "qwen3-embedding-4b"),
            .resolved("Qwen/Qwen3-Embedding-4B"))
    }

    func testRequestedMissingIsNotAvailable() {
        XCTAssertEqual(
            ModelSelection.resolve(
                available: store, configuredDefault: nil,
                requested: "nope/other"),
            .notAvailable)
    }

    // MARK: omitted id — the ambiguity rule

    func testOmittedWithConfiguredDefaultUsesIt() {
        XCTAssertEqual(
            ModelSelection.resolve(
                available: store,
                configuredDefault: "Qwen3-Embedding-4B", requested: nil),
            .resolved("Qwen/Qwen3-Embedding-4B"))
        // empty string ⇒ treated as unset.
        XCTAssertEqual(
            ModelSelection.resolve(
                available: ["only"], configuredDefault: "",
                requested: ""),
            .resolved("only"))
    }

    func testOmittedSoleModelUsed() {
        XCTAssertEqual(
            ModelSelection.resolve(
                available: ["only-one"], configuredDefault: nil,
                requested: nil),
            .resolved("only-one"))
    }

    func testOmittedMultipleNoDefaultIsAmbiguous() {
        XCTAssertEqual(
            ModelSelection.resolve(
                available: store, configuredDefault: nil, requested: nil),
            .ambiguous)
    }

    func testOmittedEmptyStoreIsNotAvailable() {
        XCTAssertEqual(
            ModelSelection.resolve(
                available: [], configuredDefault: nil, requested: nil),
            .notAvailable)
        // A configured default that names nothing in the store also can't
        // resolve an empty store.
        XCTAssertEqual(
            ModelSelection.resolve(
                available: [], configuredDefault: "ghost", requested: nil),
            .notAvailable)
    }

    /// A configured default that is no longer in the store falls through to
    /// the count-based rule rather than hard-failing.
    func testStaleDefaultFallsThroughToCount() {
        XCTAssertEqual(
            ModelSelection.resolve(
                available: ["only"], configuredDefault: "removed",
                requested: nil),
            .resolved("only"))
        XCTAssertEqual(
            ModelSelection.resolve(
                available: store, configuredDefault: "removed",
                requested: nil),
            .ambiguous)
    }

    // MARK: displayDefault

    func testDisplayDefault() {
        XCTAssertEqual(
            ModelSelection.displayDefault(
                available: store, configuredDefault: "Qwen3-Embedding-4B"),
            "Qwen/Qwen3-Embedding-4B")
        XCTAssertEqual(
            ModelSelection.displayDefault(
                available: ["only"], configuredDefault: nil), "only")
        // Ambiguous / empty ⇒ "".
        XCTAssertEqual(
            ModelSelection.displayDefault(
                available: store, configuredDefault: nil), "")
        XCTAssertEqual(
            ModelSelection.displayDefault(
                available: [], configuredDefault: nil), "")
    }
}

final class ModelModalityAcceptorTests: XCTestCase {
    func testSlotAcceptors() {
        XCTAssertTrue(ModelModality.llm.isLLMSlot)
        // The LLM slot also serves vision checkpoints (ADR 010/012).
        XCTAssertTrue(ModelModality.vision.isLLMSlot)
        XCTAssertFalse(ModelModality.embedding.isLLMSlot)

        XCTAssertTrue(ModelModality.embedding.isEmbeddingSlot)
        XCTAssertTrue(
            ModelModality.transcription(.whisper).isTranscriptionSlot)
        XCTAssertTrue(
            ModelModality.diarization(.sortformer).isDiarizationSlot)
        XCTAssertTrue(ModelModality.speakerEmbedding.isSpeakerEmbeddingSlot)
        // Cross-slot negatives.
        XCTAssertFalse(ModelModality.llm.isEmbeddingSlot)
        XCTAssertFalse(
            ModelModality.embedding.isTranscriptionSlot)
    }
}

final class StoreModelClassTests: XCTestCase {
    /// Build a temp store with one generative (LLM) dir and one embedder
    /// (bert) dir, plus a non-model dir, and assert the per-modality scan
    /// returns only the matching basenames.
    func testEnumerateByModality() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("athena-store-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        func makeModel(_ name: String, config: String) throws {
            let dir = root.appendingPathComponent(name, isDirectory: true)
            try FileManager.default.createDirectory(
                at: dir, withIntermediateDirectories: true)
            try Data(config.utf8).write(
                to: dir.appendingPathComponent("config.json"))
        }
        try makeModel("llm-a", config: #"{"model_type":"qwen3"}"#)
        try makeModel("emb-a", config: #"{"model_type":"bert"}"#)
        // A dir with no config.json is not a model.
        try FileManager.default.createDirectory(
            at: root.appendingPathComponent("not-a-model"),
            withIntermediateDirectories: true)

        XCTAssertEqual(
            StoreModelClass.ids(storeRoot: root, accept: { $0.isLLMSlot }),
            ["llm-a"])
        XCTAssertEqual(
            StoreModelClass.ids(
                storeRoot: root, accept: { $0.isEmbeddingSlot }),
            ["emb-a"])
        // nil root ⇒ empty.
        XCTAssertEqual(
            StoreModelClass.ids(storeRoot: nil, accept: { _ in true }), [])
    }
}
