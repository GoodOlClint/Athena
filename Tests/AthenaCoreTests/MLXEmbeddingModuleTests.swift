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

    func testAbsentModelServesConfiguredDefault() async throws {
        let m = StubEmbeddingModule(
            modelIds: ["m-small", "m-large"], configuredDefault: "m-small")
        let batch = try await m.embed(["hi"], model: nil)
        XCTAssertEqual(batch.model, "m-small", "nil ⇒ configured default")
    }

    /// ADR 026: omit `model` with >1 selectable and NO configured default is
    /// ambiguous — a 400 `ambiguous_model`, never a silent pick.
    func testAbsentModelAmbiguousWithoutDefault() async throws {
        let m = StubEmbeddingModule(modelIds: ["m-small", "m-large"])
        do {
            _ = try await m.embed(["hi"], model: nil)
            XCTFail("expected ambiguous_model")
        } catch let e as AthenaError {
            XCTAssertEqual(e.code, "ambiguous_model")
            XCTAssertEqual(e.httpStatus, 400)
        }
    }

    /// ADR 026: omit `model` with exactly ONE selectable model uses it (no
    /// configured default needed).
    func testAbsentModelSoleModelUsed() async throws {
        let m = StubEmbeddingModule(modelIds: ["only-one"])
        let batch = try await m.embed(["hi"], model: nil)
        XCTAssertEqual(batch.model, "only-one")
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

    /// NI2: the stub now resolves a request's `model` by store-dir identity
    /// (bare basename OR full HF id), exactly like the real MLX module, so
    /// stub-validated tests exercise the same id-acceptance as production.
    /// Previously the bare name 400'd under --engine stub but succeeded
    /// under --engine mlx.
    func testBareStoreNameResolvesForParityWithMLX() async throws {
        let m = StubEmbeddingModule(
            modelIds: ["BAAI/bge-small-en-v1.5", "Qwen/Qwen3-Embedding-4B"])
        // Naming the model by its bare store-dir basename now resolves to
        // the configured full HF id (was modelNotAvailable pre-NI2).
        let batch = try await m.embed(["hi"], model: "Qwen3-Embedding-4B")
        XCTAssertEqual(batch.model, "Qwen/Qwen3-Embedding-4B")
        let r = await m.residentModelId()
        XCTAssertEqual(r, "Qwen/Qwen3-Embedding-4B")
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

/// NI6 — the greedy length-bucketing packer, extracted from the embedding
/// forward so its pack/reassembly invariant is CI-testable without a model.
/// (The gated integration test only ever fed 2 short inputs that fit one
/// bucket, so multi-bucket / budget-split / order-preservation were never
/// exercised.) Pure index/length bookkeeping — no MLX.
final class EmbeddingBucketingTests: XCTestCase {

    private func buckets(
        _ lens: [Int], budget: Int = 32_768, maxItems: Int = 64
    ) -> [[Int]] {
        MLXEmbeddingModule.lengthBuckets(
            tokenLengths: lens, tokenBudget: budget,
            maxItemsPerBucket: maxItems)
    }

    func testEveryIndexAppearsExactlyOnce() {
        let lens = [400, 2500, 30, 30, 1200, 7, 900, 50, 50, 4000]
        let bs = buckets(lens)
        let all = bs.flatMap { $0 }.sorted()
        XCTAssertEqual(
            all, Array(0 ..< lens.count),
            "every original index packed exactly once (reassembly safety)")
    }

    func testNoBucketExceedsItemCap() {
        let lens = Array(repeating: 10, count: 200)
        let bs = buckets(lens, maxItems: 64)
        XCTAssertEqual(bs.flatMap { $0 }.count, 200)
        for b in bs {
            XCTAssertLessThanOrEqual(b.count, 64, "per-bucket item cap honored")
        }
        XCTAssertGreaterThanOrEqual(bs.count, 4, "200 / 64 ⇒ ≥4 buckets")
    }

    func testTokenBudgetSplitsLongItems() {
        // 9 items of length 300, budget 1000: 3×300=900≤1000 but 4×300=1200>1000.
        let bs = buckets(Array(repeating: 300, count: 9), budget: 1000)
        for b in bs {
            // all items are length 300, so the bucket's max length is 300.
            XCTAssertLessThanOrEqual(
                b.count * 300, 1000, "bucket stays within the token budget")
        }
        XCTAssertEqual(bs.count, 3, "9 items, 3 per bucket")
    }

    func testSingletonOversizedItemGetsOwnBucket() {
        // One item alone exceeds the budget → still admitted in its own bucket
        // (it must be embedded; NI4 rejects truly-oversized inputs upstream).
        let bs = buckets([50, 5000, 50], budget: 1000)
        let big = bs.first { $0.contains(1) }
        XCTAssertEqual(big, [1], "oversized item is its own singleton bucket")
    }

    func testPackedAscendingByLength() {
        // Huge budget ⇒ one bucket; indices ordered by ascending token length.
        let bs = buckets([100, 5, 50], budget: 1_000_000)
        XCTAssertEqual(bs.count, 1)
        XCTAssertEqual(bs[0], [1, 2, 0], "sorted by ascending length")
    }

    func testEmptyInputYieldsNoBuckets() {
        XCTAssertTrue(buckets([]).isEmpty)
    }
}
