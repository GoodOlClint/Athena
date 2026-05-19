import AthenaCore
import AthenaModels
import Foundation
import MLX
import XCTest

@testable import AthenaLLM

/// M21.4 — model-on quality sanity for the TriAttention norm-only KV
/// eviction path, plus a cache-level eviction + populated-array
/// round-trip (the MLXArray round-trip M21.3 deferred here — same
/// M20.3 → M20.4 split). Heavy / needs the MLX runtime, so gated
/// behind ATHENA_RUN_MODEL_TESTS=1 (CI never runs it).
final class TriAttentionE2ETests: XCTestCase {

    /// MLX-runtime gate only (no model needed).
    private func skipUnlessMLX() throws {
        guard
            ProcessInfo.processInfo.environment["ATHENA_RUN_MODEL_TESTS"]
                == "1"
        else {
            throw XCTSkip("set ATHENA_RUN_MODEL_TESTS=1 to run (heavy)")
        }
    }

    private func skipUnlessModel() throws -> URL {
        try skipUnlessMLX()
        let env = ProcessInfo.processInfo.environment
        let modelURL = ModelStore().resolve(env["ATHENA_TEST_MODEL"])
        guard
            FileManager.default.fileExists(
                atPath: modelURL.appendingPathComponent("config.json").path)
        else {
            throw XCTSkip("model not present at \(modelURL.path)")
        }
        return modelURL
    }

    private func generate(
        model: URL, kv: KVCompression
    ) async throws -> String {
        let llm = MLXLLMModule(
            modelDirectory: model,
            parameters: .init(
                maxTokens: 48, temperature: 0, kvCompression: kv))
        let gov = MemoryGovernor(totalBudgetBytes: Int(64) << 30)
        await gov.register(llm, evictable: false)
        try await gov.ensureLoaded(.llm)
        var out = ""
        for await chunk in llm.generate(
            prompt: "List three primary colors, comma separated.")
        {
            out += chunk
        }
        return out.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func assertCoherent(_ text: String, _ label: String) {
        XCTAssertFalse(text.isEmpty, "\(label): empty output")
        XCTAssertGreaterThan(
            text.count, 3, "\(label): implausibly short output")
        let distinct = Set(text.replacingOccurrences(of: " ", with: ""))
        XCTAssertGreaterThan(
            distinct.count, 2,
            "\(label): degenerate output (\(distinct.count) distinct "
                + "chars): \(text)")
    }

    func testTriAttentionProducesCoherentOutput() async throws {
        let model = try skipUnlessModel()
        let out = try await generate(model: model, kv: .triattention)
        assertCoherent(out, "triattention")
    }

    /// Sanity A/B: the uncompressed path and the TriAttention path both
    /// produce coherent text. Not required identical — eviction changes
    /// which tokens remain in the cache by design; only that enabling it
    /// does not collapse generation (here the budget is not even hit, so
    /// the evicting cache must be a correct KVCacheSimple drop-in).
    func testUncompressedVsTriAttentionBothCoherent() async throws {
        let model = try skipUnlessModel()
        let plain = try await generate(model: model, kv: .none)
        assertCoherent(plain, "none")
        let tri = try await generate(model: model, kv: .triattention)
        assertCoherent(tri, "triattention")
    }

    /// Cache-level: drive enough decode steps to exceed a tiny budget and
    /// assert eviction (a) fires and bounds the cache, (b) never drops
    /// the pinned prefill, and (c) the populated keys/values survive a
    /// state/metaState `fromState` round-trip with shape preserved.
    func testEvictionBoundsCacheAndPopulatedRoundTrip() throws {
        try skipUnlessMLX()
        let prefill = 10
        let cfg = TriAttentionConfig(
            kvBudget: 16, divideLength: 4, scoreAggregation: .mean,
            prefillPin: true)
        let cache = TriAttentionKVCache(config: cfg)
        func z(_ n: Int) -> MLXArray {
            MLXArray.zeros([1, 2, n, 8], dtype: .float32)
        }

        _ = cache.update(keys: z(prefill), values: z(prefill))
        XCTAssertEqual(cache.offset, prefill)
        for _ in 0 ..< 40 {
            _ = cache.update(keys: z(1), values: z(1))
        }

        // Eviction fired (without it offset would be prefill + 40 = 50)
        // and the cache is bounded; the pinned prefill is never dropped.
        XCTAssertLessThan(
            cache.offset, prefill + 40,
            "eviction did not fire — cache grew unbounded")
        XCTAssertLessThanOrEqual(
            cache.offset, prefill + cfg.kvBudget + cfg.divideLength,
            "cache exceeded the eviction bound")
        XCTAssertGreaterThanOrEqual(
            cache.offset, prefill, "prefill was evicted")
        XCTAssertEqual(Int(cache.metaState[5]), prefill)

        let state = cache.state
        XCTAssertEqual(state.count, 2)
        XCTAssertEqual(state[0].dim(2), cache.offset)

        let restored = try TriAttentionKVCache.fromState(
            state: state, metaState: cache.metaState)
        XCTAssertEqual(restored.offset, cache.offset)
        XCTAssertEqual(restored.metaState, cache.metaState)
        XCTAssertEqual(restored.state.count, 2)
        XCTAssertEqual(restored.state[0].shape, state[0].shape)
        XCTAssertEqual(restored.state[1].shape, state[1].shape)
    }
}
