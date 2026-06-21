import Foundation
import XCTest

@testable import AthenaLLM

/// NC4 (M70.3) — `PrefixKVCache` carries two load-bearing invariants with zero
/// prior coverage: SECURITY (`scopeKey` + the `e.scope == scope` filter must
/// never serve one principal's/model's KV to another) and MEMORY
/// (count/byte/idle-TTL eviction with refcount protecting in-flight entries).
/// The KV-tensor clone/restore stays MLX-gated, but the scope keys, the prefix
/// math, and the eviction policy are pure metadata logic and are pinned here.
final class PrefixKVCacheTests: XCTestCase {

    // MARK: - scopeKey isolation (SECURITY)

    /// In the security-default `.principal` mode the key is (model, principal)
    /// only — the request's `cacheKey` is IGNORED, so a crafted cacheKey can
    /// never breach principal isolation, and different principals / models
    /// never share a key.
    func testScopeKeyPrincipalModeIsolates() {
        let c = PrefixKVCache(maxEntries: 4, scope: .principal)
        let alice = c.scopeKey(model: "m", principal: "alice", cacheKey: nil)
        let aliceAgain = c.scopeKey(
            model: "m", principal: "alice", cacheKey: "ignored")
        let bob = c.scopeKey(model: "m", principal: "bob", cacheKey: nil)
        let alice2 = c.scopeKey(model: "m2", principal: "alice", cacheKey: nil)
        XCTAssertEqual(alice, aliceAgain, "cacheKey must be ignored in .principal")
        XCTAssertNotEqual(alice, bob, "different principals must not share KV")
        XCTAssertNotEqual(alice, alice2, "different models must not share KV")
        // anon principal is its own bucket, distinct from a named one.
        let anon = c.scopeKey(model: "m", principal: nil, cacheKey: nil)
        XCTAssertNotEqual(anon, alice)
    }

    /// `.cacheKey` mode keys by the OpenAI `prompt_cache_key`, falling back to
    /// the principal when omitted; the model id is always part of the key.
    func testScopeKeyCacheKeyMode() {
        let c = PrefixKVCache(maxEntries: 4, scope: .cacheKey)
        let k1 = c.scopeKey(model: "m", principal: "alice", cacheKey: "shared")
        let k2 = c.scopeKey(model: "m", principal: "bob", cacheKey: "shared")
        XCTAssertEqual(k1, k2, "same cache_key shares regardless of principal")
        let fallback = c.scopeKey(model: "m", principal: "alice", cacheKey: nil)
        let fallbackBob = c.scopeKey(model: "m", principal: "bob", cacheKey: nil)
        XCTAssertNotEqual(
            fallback, fallbackBob, "no cache_key ⇒ fall back to principal")
        let otherModel = c.scopeKey(
            model: "m2", principal: "alice", cacheKey: "shared")
        XCTAssertNotEqual(k1, otherModel, "model id always part of the key")
    }

    /// `.both` mode: principal AND cache_key both narrow the scope.
    func testScopeKeyBothMode() {
        let c = PrefixKVCache(maxEntries: 4, scope: .both)
        let base = c.scopeKey(model: "m", principal: "alice", cacheKey: "k")
        XCTAssertEqual(
            base, c.scopeKey(model: "m", principal: "alice", cacheKey: "k"))
        XCTAssertNotEqual(
            base, c.scopeKey(model: "m", principal: "bob", cacheKey: "k"))
        XCTAssertNotEqual(
            base, c.scopeKey(model: "m", principal: "alice", cacheKey: "k2"))
    }

    // MARK: - ADR 024 T3 idle-encryption flag wiring

    /// The `encryptIdle` init flag drives whether idle entries are sealed; the
    /// seal/restore numerics are MLX-gated (the bit-identical gate), but the
    /// flag plumbing is pure and pinned here.
    func testEncryptIdleFlagWiring() {
        XCTAssertFalse(
            PrefixKVCache(maxEntries: 4).encryptsIdleEntries,
            "default is plaintext (off)")
        XCTAssertTrue(
            PrefixKVCache(maxEntries: 4, encryptIdle: true).encryptsIdleEntries)
    }

    // MARK: - commonPrefixLength edge cases

    func testCommonPrefixLength() {
        XCTAssertEqual(PrefixKVCache.commonPrefixLength([], []), 0)
        XCTAssertEqual(PrefixKVCache.commonPrefixLength([1, 2, 3], []), 0)
        XCTAssertEqual(
            PrefixKVCache.commonPrefixLength([1, 2, 3], [1, 2, 3]), 3,
            "identical")
        XCTAssertEqual(
            PrefixKVCache.commonPrefixLength([1, 2, 3, 4], [1, 2, 9]), 2,
            "diverge at index 2")
        XCTAssertEqual(
            PrefixKVCache.commonPrefixLength([1, 2], [1, 2, 3, 4]), 2,
            "one is a prefix of the other")
        XCTAssertEqual(
            PrefixKVCache.commonPrefixLength([9, 1], [1, 9]), 0,
            "diverge at index 0")
    }

    // MARK: - eviction policy (MEMORY): PrefixCachePolicy

    private func meta(bytes: Int, age: TimeInterval, ref: Int) -> PrefixCachePolicy.Meta {
        // `age` seconds in the past from a fixed `now`.
        PrefixCachePolicy.Meta(
            byteEstimate: bytes, lastUsed: Self.now.addingTimeInterval(-age),
            refcount: ref)
    }
    private static let now = Date(timeIntervalSince1970: 1_000_000)

    /// idle-TTL sweep: only refcount==0 AND idle beyond the TTL; ttl<=0
    /// disables it.
    func testIdleVictimSelection() {
        let metas = [
            meta(bytes: 1, age: 1000, ref: 0),  // idle + free → victim
            meta(bytes: 1, age: 1000, ref: 1),  // idle but IN USE → protected
            meta(bytes: 1, age: 10, ref: 0),    // fresh → kept
        ]
        let v = PrefixCachePolicy.idleVictimIndices(
            metas, now: Self.now, idleTTL: 600)
        XCTAssertEqual(v, [0], "only the idle, refcount==0 entry is swept")
        XCTAssertEqual(
            PrefixCachePolicy.idleVictimIndices(metas, now: Self.now, idleTTL: 0),
            [], "idleTTL<=0 disables the sweep")
    }

    /// count cap and byte cap; byte cap inert at maxBytes==0.
    func testOverCap() {
        let three = [
            meta(bytes: 10, age: 1, ref: 0),
            meta(bytes: 10, age: 2, ref: 0),
            meta(bytes: 10, age: 3, ref: 0),
        ]
        XCTAssertTrue(
            PrefixCachePolicy.isOverCap(three, maxEntries: 2, maxBytes: 0),
            "count cap exceeded")
        XCTAssertFalse(
            PrefixCachePolicy.isOverCap(three, maxEntries: 3, maxBytes: 0),
            "count cap inert at the limit; byte cap off")
        XCTAssertTrue(
            PrefixCachePolicy.isOverCap(three, maxEntries: 99, maxBytes: 25),
            "byte cap exceeded (30 > 25)")
        XCTAssertFalse(
            PrefixCachePolicy.isOverCap(three, maxEntries: 99, maxBytes: 0),
            "maxBytes==0 ⇒ byte cap inert")
    }

    /// LRU victim is the oldest refcount==0 entry while over cap; ties go to
    /// the lowest index; an in-use entry is never the victim.
    func testLRUVictimSelection() {
        let metas = [
            meta(bytes: 10, age: 5, ref: 0),    // oldest free
            meta(bytes: 10, age: 100, ref: 1),  // older but IN USE
            meta(bytes: 10, age: 2, ref: 0),    // newer free
        ]
        // Over the count cap (3 > 1): the oldest FREE entry (index 0) goes,
        // never the in-use index 1 even though it is older.
        XCTAssertEqual(
            PrefixCachePolicy.lruVictimIndex(metas, maxEntries: 1, maxBytes: 0),
            0)
        // Under cap ⇒ no victim.
        XCTAssertNil(
            PrefixCachePolicy.lruVictimIndex(metas, maxEntries: 9, maxBytes: 0))
    }

    /// Over cap but every entry is in use ⇒ nothing evictable (the eviction
    /// loop must terminate, not spin) — the refcount-protection invariant.
    func testLRUVictimAllInUseIsNil() {
        let metas = [
            meta(bytes: 10, age: 5, ref: 1),
            meta(bytes: 10, age: 9, ref: 2),
        ]
        XCTAssertNil(
            PrefixCachePolicy.lruVictimIndex(metas, maxEntries: 1, maxBytes: 0),
            "all in-flight ⇒ no victim, loop terminates")
    }

    /// LRU tie on lastUsed resolves to the lowest index (matches the prior
    /// `enumerated().min` order, which keeps same-input determinism).
    func testLRUVictimTieBreaksLowestIndex() {
        let metas = [
            meta(bytes: 10, age: 50, ref: 0),
            meta(bytes: 10, age: 50, ref: 0),  // same age as index 0
            meta(bytes: 10, age: 1, ref: 0),
        ]
        XCTAssertEqual(
            PrefixCachePolicy.lruVictimIndex(metas, maxEntries: 2, maxBytes: 0),
            0, "tie on lastUsed ⇒ lowest index")
    }

    /// flushIdle drops every refcount==0 entry, protecting in-flight ones.
    func testFlushIdleIndices() {
        let metas = [
            meta(bytes: 1, age: 1, ref: 0),
            meta(bytes: 1, age: 1, ref: 2),  // in use
            meta(bytes: 1, age: 1, ref: 0),
        ]
        XCTAssertEqual(
            PrefixCachePolicy.flushIdleIndices(metas), [0, 2],
            "only free entries flushed")
    }
}
