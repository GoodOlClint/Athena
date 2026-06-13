import Foundation
import XCTest

@testable import AthenaEmbedding

/// L8 (M70.3) — the embedding stub used to fold only the text, so every model
/// produced the SAME vector and a test could assert only the echoed `model`
/// label, not that the vector actually depends on the model. The stub now
/// folds the served id into the hash, mirroring the real module (where
/// different models genuinely differ). Pin: distinct models ⇒ distinct
/// vectors, stable per (model, text), L2-normalized.
final class StubEmbeddingTests: XCTestCase {

    func testDistinctModelsProduceDistinctVectors() {
        let a = StubEmbeddingModule.stubVector(text: "hello world", model: "model-a")
        let b = StubEmbeddingModule.stubVector(text: "hello world", model: "model-b")
        XCTAssertNotEqual(a, b, "same text, different model ⇒ different vector")
    }

    func testStablePerModelAndText() {
        let a1 = StubEmbeddingModule.stubVector(text: "hi", model: "m")
        let a2 = StubEmbeddingModule.stubVector(text: "hi", model: "m")
        XCTAssertEqual(a1, a2, "deterministic per (model, text)")
        let other = StubEmbeddingModule.stubVector(text: "ho", model: "m")
        XCTAssertNotEqual(a1, other, "different text ⇒ different vector")
    }

    func testNonEmptyVectorIsL2Normalized() {
        let v = StubEmbeddingModule.stubVector(text: "some text", model: "m")
        let norm = sqrt(v.reduce(0) { $0 + $1 * $1 })
        XCTAssertEqual(norm, 1.0, accuracy: 1e-5)
        XCTAssertEqual(v.count, 8)
    }

    /// The embed path actually routes the SERVED id into the vector math, and
    /// reports it — so selecting a different model both echoes a different
    /// label AND returns a different vector (the L8 gap).
    func testEmbedVectorDependsOnSelectedModel() async throws {
        let stub = StubEmbeddingModule(modelIds: ["m1", "m2"])
        let r1 = try await stub.embed(["hi"], model: "m1")
        let r2 = try await stub.embed(["hi"], model: "m2")
        XCTAssertEqual(r1.model, "m1")
        XCTAssertEqual(r2.model, "m2")
        XCTAssertNotEqual(
            r1.vectors[0], r2.vectors[0],
            "the vector must change with the served model, not just the label")
        // And it equals the pure helper for the served id.
        XCTAssertEqual(
            r1.vectors[0],
            StubEmbeddingModule.stubVector(text: "hi", model: "m1"))
    }
}
