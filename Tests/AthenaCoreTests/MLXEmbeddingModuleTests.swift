import AthenaCore
import Foundation
import XCTest

@testable import AthenaEmbedding

/// Stub embeddings — deterministic, model-free, CI-safe.
final class StubEmbeddingModuleTests: XCTestCase {

    func testDeterministicShapeOrderAndNorm() async throws {
        let m = StubEmbeddingModule()
        let a = try await m.embed(["alpha", "beta", "alpha"]).vectors
        XCTAssertEqual(a.count, 3)
        XCTAssertEqual(a[0].count, 8)
        XCTAssertEqual(a[0], a[2], "same input ⇒ same vector")
        XCTAssertNotEqual(a[0], a[1])
        // L2-normalized.
        let norm = sqrt(a[1].reduce(0) { $0 + $1 * $1 })
        XCTAssertEqual(norm, 1, accuracy: 1e-4)
    }

    func testEmptyInputEmptyOutput() async throws {
        let out = try await StubEmbeddingModule().embed([]).vectors
        XCTAssertTrue(out.isEmpty)
    }

    // M39: per-request model selection over the configured set.

    func testAbsentModelServesDefaultTruthfully() async throws {
        let m = StubEmbeddingModule(modelIds: ["m-small", "m-large"])
        let batch = try await m.embed(["hi"], model: nil)
        XCTAssertEqual(batch.model, "m-small", "nil ⇒ first declared")
    }

    func testSelectsAllowedModelAndEchoesTruthfully() async throws {
        let m = StubEmbeddingModule(modelIds: ["m-small", "m-large"])
        let batch = try await m.embed(["hi"], model: "m-large")
        XCTAssertEqual(batch.model, "m-large", "served id is echoed")
    }

    func testUnknownModelThrowsModelNotAvailable() async throws {
        let m = StubEmbeddingModule(modelIds: ["m-small"])
        do {
            _ = try await m.embed(["hi"], model: "nope/not-loaded")
            XCTFail("expected modelNotAvailable")
        } catch let e as AthenaError {
            XCTAssertEqual(e.code, "model_not_available")
            XCTAssertEqual(e.httpStatus, 400)
        }
    }
}

/// MLX module — construction/accounting only (loading downloads a model,
/// so the real path is gated below).
final class MLXEmbeddingModuleEstimateTests: XCTestCase {

    func testDefaultEstimateAndId() async {
        let m = MLXEmbeddingModule()
        XCTAssertEqual(m.id, .textEmbedding)
        let est = await m.memoryEstimate()
        XCTAssertEqual(est, 512 * 1024 * 1024)
        let resident = await m.residentBytes
        XCTAssertEqual(resident, 0, "unloaded ⇒ 0 resident")
    }

    func testCustomEstimate() async {
        let est = await MLXEmbeddingModule(estimatedBytes: 99)
            .memoryEstimate()
        XCTAssertEqual(est, 99)
    }
}

/// Real end-to-end embedding through the governor. Gated: loading a model
/// downloads weights, so it runs only with ATHENA_RUN_MODEL_TESTS=1.
final class MLXEmbeddingIntegrationTests: XCTestCase {

    func testGovernedEmbeddingProducesVectors() async throws {
        guard
            ProcessInfo.processInfo.environment["ATHENA_RUN_MODEL_TESTS"]
                == "1"
        else { throw XCTSkip("set ATHENA_RUN_MODEL_TESTS=1 (heavy)") }

        let m = MLXEmbeddingModule()
        let gov = MemoryGovernor(totalBudgetBytes: Int(8) << 30)
        await gov.register(m, evictable: true)
        try await gov.ensureLoaded(.textEmbedding)

        let v = try await m.embed(["hello world", "a different sentence"])
            .vectors
        XCTAssertEqual(v.count, 2)
        XCTAssertEqual(v[0].count, 384, "bge-small is 384-dim")
        XCTAssertEqual(
            sqrt(v[0].reduce(0) { $0 + $1 * $1 }), 1, accuracy: 1e-3)
        XCTAssertNotEqual(v[0], v[1])
    }
}
