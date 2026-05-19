import XCTest

import AthenaModels

/// CI-safe coverage for `TriAttentionKVCache`'s self-contained
/// prompt-cache round-trip — the `metaState`/`fromState` codec that
/// reconstructs the eviction policy + bookkeeping. Pure metadata (empty
/// `state`), so no MLX runtime is needed; the full populated-array
/// round-trip is exercised by the M21.4 model-on e2e (mirrors the
/// M20.3 → M20.4 split — there is no CI-safe surface for MLXArray IO).
final class TriAttentionCacheTests: XCTestCase {

    func testMetaStateRoundTripsConfigAndBookkeeping() throws {
        let meta = ["4096", "64", "max", "false", "0", "0", "false", "0"]
        let cache = try TriAttentionKVCache.fromState(
            state: [], metaState: meta)
        XCTAssertEqual(cache.config.kvBudget, 4096)
        XCTAssertEqual(cache.config.divideLength, 64)
        XCTAssertEqual(cache.config.scoreAggregation, .max)
        XCTAssertFalse(cache.config.prefillPin)
        XCTAssertEqual(cache.offset, 0)
        // Round-trip identity: the getter reproduces the input exactly.
        XCTAssertEqual(cache.metaState, meta)
    }

    func testMetaStatePreservesPrefillAndStepCounters() throws {
        let meta = ["2048", "128", "mean", "true", "300", "256", "true", "44"]
        let cache = try TriAttentionKVCache.fromState(
            state: [], metaState: meta)
        XCTAssertTrue(cache.config.prefillPin)
        XCTAssertEqual(cache.config.scoreAggregation, .mean)
        XCTAssertEqual(cache.offset, 300)
        XCTAssertEqual(cache.metaState, meta)
    }

    func testFromStateRejectsTruncatedMetaState() {
        XCTAssertThrowsError(
            try TriAttentionKVCache.fromState(
                state: [], metaState: ["2048", "128", "mean"]))
    }
}
