import Foundation
import MLX
import XCTest

@testable import AthenaStore

/// M7.2 — built-in vector DB. Cosine ranking is exact and
/// deterministic; gated on MLX/Metal (xcodebuild) since `query`
/// computes scores on MLX.
final class VectorStoreTests: XCTestCase {

    private func freshStore() throws -> (AthenaStore, URL) {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("athena-vs-\(UUID()).sqlite")
        return (try AthenaStore(path: url), url)
    }

    func testUpsertDeleteStatsCapCI() async throws {
        // CI-safe: no `query` (MLX) here — persistence/cap/stats only.
        let (s, url) = try freshStore()
        defer { try? FileManager.default.removeItem(at: url) }
        // cap = 2 vectors of dim 3 (×4 bytes) = 24 B.
        let vs = VectorStore(store: s, capBytes: 24)

        try await vs.upsert(id: "a", vector: [1, 0, 0], metadata: nil)
        try await vs.upsert(
            id: "b", vector: [0, 1, 0],
            metadata: Data(#"{"t":1}"#.utf8))
        var st = await vs.stats()
        XCTAssertEqual(st.count, 2)
        XCTAssertEqual(st.dim, 3)
        XCTAssertEqual(st.bytes, 24)

        // Third NEW vector exceeds the 24 B cap.
        do {
            try await vs.upsert(id: "c", vector: [0, 0, 1], metadata: nil)
            XCTFail("expected capExceeded")
        } catch let e as VectorStore.VectorError {
            guard case .capExceeded = e else {
                return XCTFail("wrong error \(e)")
            }
        }
        // Updating an EXISTING id is allowed (not new).
        try await vs.upsert(id: "a", vector: [2, 0, 0], metadata: nil)

        // Dim mismatch rejected.
        do {
            try await vs.upsert(id: "d", vector: [1, 1], metadata: nil)
            XCTFail("expected dimMismatch")
        } catch let e as VectorStore.VectorError {
            guard case .dimMismatch = e else {
                return XCTFail("wrong error \(e)")
            }
        }

        let del = await vs.delete(id: "b")
        XCTAssertTrue(del)
        st = await vs.stats()
        XCTAssertEqual(st.count, 1)
        // Persisted through to SQLite.
        let raw = await s.getVector(id: "a")
        XCTAssertEqual(raw?.vector, [2, 0, 0])
        let goneB = await s.getVector(id: "b")
        XCTAssertNil(goneB)
    }

    /// H10 (M68.3) — a zero-length vector is rejected (it would otherwise be
    /// adopted as the authoritative store `dim` on an empty store). CI-safe.
    func testZeroLengthVectorRejectedH10() async throws {
        let (s, url) = try freshStore()
        defer { try? FileManager.default.removeItem(at: url) }
        let vs = VectorStore(store: s, capBytes: 1 << 20)
        // Empty store: an empty vector must NOT silently set dim=0.
        do {
            try await vs.upsert(id: "z", vector: [], metadata: nil)
            XCTFail("expected zeroLengthVector")
        } catch let e as VectorStore.VectorError {
            guard case .zeroLengthVector = e else {
                return XCTFail("wrong error \(e)")
            }
        }
        // Nothing was stored, so the dim is still unset; a real vector works.
        try await vs.upsert(id: "a", vector: [1, 0, 0], metadata: nil)
        let st = await vs.stats()
        XCTAssertEqual(st.count, 1)
        XCTAssertEqual(st.dim, 3)
    }

    /// H4 (M68.3) — concurrent NEW upserts must not overrun the byte cap by
    /// both passing a cap check that straddled the `putVector` await. With a
    /// cap of exactly 2 rows (dim 3 × 4 B = 24 B), firing 8 distinct-id
    /// upserts concurrently must admit at most 2 and reject the rest. CI-safe.
    func testConcurrentUpsertsRespectCapH4() async throws {
        let (s, url) = try freshStore()
        defer { try? FileManager.default.removeItem(at: url) }
        let vs = VectorStore(store: s, capBytes: 24)  // exactly 2 dim-3 rows

        let admitted = await withTaskGroup(of: Bool.self) { group in
            for i in 0..<8 {
                group.addTask {
                    do {
                        try await vs.upsert(
                            id: "v\(i)", vector: [Float(i), 0, 0],
                            metadata: nil)
                        return true
                    } catch { return false }
                }
            }
            var n = 0
            for await ok in group where ok { n += 1 }
            return n
        }

        XCTAssertEqual(
            admitted, 2,
            "the reservation must let exactly cap-many new upserts in")
        let st = await vs.stats()
        XCTAssertEqual(st.count, 2, "store never exceeds the 2-row cap")
        XCTAssertLessThanOrEqual(st.bytes, 24)
    }

    /// H5 (M66.6 / ADR 006) — owner-scoping at the cache/stats/delete
    /// layer. CI-safe: exercises everything EXCEPT `query` (MLX), so no
    /// Metal needed; the gated `testQueryOwnerScopedGated` covers query.
    func testOwnerScopingCI() async throws {
        let (s, url) = try freshStore()
        defer { try? FileManager.default.removeItem(at: url) }
        let vs = VectorStore(store: s, capBytes: 1 << 20)
        let alice = VectorStore.Caller(
            principal: "u:alice", isAdmin: false, enforced: true)
        let bob = VectorStore.Caller(
            principal: "u:bob", isAdmin: false, enforced: true)
        let admin = VectorStore.Caller(
            principal: "u:admin", isAdmin: true, enforced: true)

        try await vs.upsert(
            id: "av", vector: [1, 0, 0], metadata: nil, caller: alice)
        try await vs.upsert(
            id: "bv", vector: [0, 1, 0], metadata: nil, caller: bob)

        // Each tenant's stats show only their own vector; admin sees both.
        let aStats = await vs.stats(caller: alice)
        XCTAssertEqual(aStats.count, 1)
        let bStats = await vs.stats(caller: bob)
        XCTAssertEqual(bStats.count, 1)
        let adminStats = await vs.stats(caller: admin)
        XCTAssertEqual(adminStats.count, 2)

        // Bob cannot delete Alice's vector (false ⇒ 404 at the server),
        // and it survives.
        let crossDel = await vs.delete(id: "av", caller: bob)
        XCTAssertFalse(crossDel)
        let survived = await s.getVector(id: "av")
        XCTAssertNotNil(survived)

        // Bob cannot overwrite Alice's id either.
        do {
            try await vs.upsert(
                id: "av", vector: [9, 9, 9], metadata: nil, caller: bob)
            XCTFail("expected ownerConflict")
        } catch let e as VectorStore.VectorError {
            guard case .ownerConflict = e else {
                return XCTFail("wrong error \(e)")
            }
        }
        // Alice's vector is unchanged after the rejected overwrite.
        let unchanged = await s.getVector(id: "av")?.vector
        XCTAssertEqual(unchanged, [1, 0, 0])

        // Alice deletes her own; admin deletes Bob's.
        let aDel = await vs.delete(id: "av", caller: alice)
        XCTAssertTrue(aDel)
        let bDel = await vs.delete(id: "bv", caller: admin)
        XCTAssertTrue(bDel)
        let endStats = await vs.stats(caller: admin)
        XCTAssertEqual(endStats.count, 0)
    }

    /// H5 — a legacy NULL-owner row (written pre-migration / auth-off) is
    /// admin-only: invisible to a scoped tenant, visible to admin.
    func testLegacyNullOwnerIsAdminOnlyCI() async throws {
        let (s, url) = try freshStore()
        defer { try? FileManager.default.removeItem(at: url) }
        // Write directly with NULL owner (the pre-migration shape).
        try await s.putVector(id: "legacy", vector: [1, 0], metadata: nil)
        let vs = VectorStore(store: s, capBytes: 1 << 20)
        let tenant = VectorStore.Caller(
            principal: "u:t", isAdmin: false, enforced: true)
        let admin = VectorStore.Caller(
            principal: "u:a", isAdmin: true, enforced: true)
        let tStats = await vs.stats(caller: tenant)
        XCTAssertEqual(tStats.count, 0)
        let adminStats = await vs.stats(caller: admin)
        XCTAssertEqual(adminStats.count, 1)
        // A scoped tenant can't delete it; admin can.
        let tDel = await vs.delete(id: "legacy", caller: tenant)
        XCTAssertFalse(tDel)
        let aDel = await vs.delete(id: "legacy", caller: admin)
        XCTAssertTrue(aDel)
    }

    /// H5 — `query` returns only the caller's visible vectors. Gated (MLX).
    func testQueryOwnerScopedGated() async throws {
        guard
            ProcessInfo.processInfo.environment["ATHENA_RUN_MODEL_TESTS"]
                == "1"
        else { throw XCTSkip("set ATHENA_RUN_MODEL_TESTS=1 (MLX/Metal)") }
        let (s, url) = try freshStore()
        defer { try? FileManager.default.removeItem(at: url) }
        let vs = VectorStore(store: s, capBytes: 1 << 20)
        let alice = VectorStore.Caller(
            principal: "u:alice", isAdmin: false, enforced: true)
        let bob = VectorStore.Caller(
            principal: "u:bob", isAdmin: false, enforced: true)
        try await vs.upsert(
            id: "av", vector: [1, 0, 0], metadata: nil, caller: alice)
        try await vs.upsert(
            id: "bv", vector: [1, 0, 0], metadata: nil, caller: bob)
        // Bob's query never returns Alice's identical-vector row.
        let bobHits = await vs.query(
            vector: [1, 0, 0], k: 10, caller: bob)
        XCTAssertEqual(bobHits.map(\.id), ["bv"])
        // Admin sees both.
        let adminHits = await vs.query(
            vector: [1, 0, 0], k: 10,
            caller: VectorStore.Caller(
                principal: "u:a", isAdmin: true, enforced: true))
        XCTAssertEqual(Set(adminHits.map(\.id)), ["av", "bv"])
    }

    func testCosineRankingGated() async throws {
        guard
            ProcessInfo.processInfo.environment["ATHENA_RUN_MODEL_TESTS"]
                == "1"
        else { throw XCTSkip("set ATHENA_RUN_MODEL_TESTS=1 (MLX/Metal)") }
        let (s, url) = try freshStore()
        defer { try? FileManager.default.removeItem(at: url) }
        let vs = VectorStore(store: s, capBytes: 1 << 20)

        try await vs.upsert(id: "x", vector: [1, 0, 0], metadata: nil)
        try await vs.upsert(id: "y", vector: [0, 1, 0], metadata: nil)
        try await vs.upsert(
            id: "z", vector: [0.9, 0.1, 0], metadata: nil)

        let hits = await vs.query(vector: [1, 0, 0], k: 2)
        XCTAssertEqual(hits.count, 2)
        // [1,0,0] is identical to x and closest-after to z.
        XCTAssertEqual(hits[0].id, "x")
        XCTAssertEqual(hits[1].id, "z")
        XCTAssertEqual(hits[0].score, 1.0, accuracy: 1e-3)
        XCTAssertGreaterThan(hits[0].score, hits[1].score)
    }

    /// M50.5 — regression for the allocator-pool leak class M46.6
    /// caught in the embedder. `query` materializes an N×dim norm
    /// matrix + matmul intermediates per call; over sustained search
    /// load those accumulate in MLX's pool. Gated on MLX/Metal.
    func testQueryPoolStaysBoundedAcrossManyCalls() async throws {
        guard
            ProcessInfo.processInfo.environment["ATHENA_RUN_MODEL_TESTS"]
                == "1"
        else { throw XCTSkip("set ATHENA_RUN_MODEL_TESTS=1 (MLX/Metal)") }
        let (s, url) = try freshStore()
        defer { try? FileManager.default.removeItem(at: url) }
        let vs = VectorStore(store: s, capBytes: 16 << 20)

        // Populate ~1k vectors of dim 256 — representative of a small
        // RAG index, large enough that per-query MLX matmul allocates
        // material amounts of pool memory.
        let dim = 256
        var rng = SystemRandomNumberGenerator()
        for i in 0..<1024 {
            var v = [Float](repeating: 0, count: dim)
            for j in 0..<dim {
                v[j] = Float(Int(rng.next() % 1000)) / 1000.0 - 0.5
            }
            try await vs.upsert(
                id: "id-\(i)", vector: v, metadata: nil)
        }
        var qv = [Float](repeating: 0, count: dim)
        for j in 0..<dim {
            qv[j] = Float(Int(rng.next() % 1000)) / 1000.0 - 0.5
        }

        // Warmup so first-call lazy allocations settle.
        _ = await vs.query(vector: qv, k: 5)
        MLX.Memory.clearCache()
        let baseline = MLX.Memory.cacheMemory

        for _ in 0..<32 { _ = await vs.query(vector: qv, k: 5) }

        let after = MLX.Memory.cacheMemory
        // Without M50.5's clear, the pool scales linearly with the
        // per-query matmul footprint × 32.
        let ceiling = 64 * 1024 * 1024
        XCTAssertLessThan(
            after - baseline, ceiling,
            "MLX cache pool drifted \(after - baseline) bytes "
            + "above baseline after 32 vector queries (M50.5 leak)")
    }

    // MARK: - M70.2 L3/L4 — ranking + dim-mismatch on a CI path
    //
    // The cosine SCORING is MLX (gated above). The H5 owner-filter + H7
    // id-tie-break ranking is pure Swift, extracted to `rankTopK`, so its
    // contract is unit-testable on CI without a Metal device.

    /// L3 — owner-scoping + the H7 stable id tie-break + top-k, over fixture
    /// scores (no MLX). CI-safe.
    func testRankTopKTieBreakAndScopingCI() {
        typealias Row = (id: String, vec: [Float], meta: Data?, owner: String?)
        let rows: [Row] = [
            ("b", [], nil, "u:alice"),
            ("a", [], nil, "u:alice"),
            ("c", [], nil, "u:bob"),
            ("d", [], nil, nil),  // legacy NULL-owner row
        ]
        // a and b tie at 0.5 → ascending-id tie-break (a before b); c highest.
        let scores: [Float] = [/*b*/ 0.5, /*a*/ 0.5, /*c*/ 0.9, /*d*/ 0.7]

        let admin = VectorStore.Caller(
            principal: "u:x", isAdmin: true, enforced: true)
        let all = VectorStore.rankTopK(
            rows: rows, scores: scores, k: 10, caller: admin)
        XCTAssertEqual(
            all.map(\.id), ["c", "d", "a", "b"],
            "score desc, equal scores broken by ascending id")
        // Stable: a second identical call yields the identical order.
        let all2 = VectorStore.rankTopK(
            rows: rows, scores: scores, k: 10, caller: admin)
        XCTAssertEqual(all2.map(\.id), all.map(\.id))
        // Top-k is a prefix of the full ranking.
        let top2 = VectorStore.rankTopK(
            rows: rows, scores: scores, k: 2, caller: admin)
        XCTAssertEqual(top2.map(\.id), ["c", "d"])

        // Scoped tenant (alice) sees only her rows — bob's and the NULL-owner
        // row are filtered out; her two tie at 0.5 → id asc.
        let alice = VectorStore.Caller(
            principal: "u:alice", isAdmin: false, enforced: true)
        let aliceHits = VectorStore.rankTopK(
            rows: rows, scores: scores, k: 10, caller: alice)
        XCTAssertEqual(aliceHits.map(\.id), ["a", "b"])
        XCTAssertEqual(aliceHits.first?.score, 0.5)

        // Auth-off (.unscoped) sees everything, including the NULL-owner row.
        let un = VectorStore.rankTopK(
            rows: rows, scores: scores, k: 10, caller: .unscoped)
        XCTAssertEqual(Set(un.map(\.id)), ["a", "b", "c", "d"])
    }

    /// L4 — `query` short-circuits to `[]` on a dim mismatch / k<=0 / empty
    /// store, BEFORE constructing any MLXArray, so it is CI-safe (no Metal).
    func testQueryDimMismatchReturnsEmptyCI() async throws {
        let (s, url) = try freshStore()
        defer { try? FileManager.default.removeItem(at: url) }
        let vs = VectorStore(store: s, capBytes: 1 << 20)
        try await vs.upsert(id: "a", vector: [1, 0, 0], metadata: nil)  // dim 3

        let mismatch = await vs.query(vector: [1, 0], k: 5)  // dim 2 ≠ 3
        XCTAssertTrue(mismatch.isEmpty, "wrong-dim query returns []")
        let zeroK = await vs.query(vector: [1, 0, 0], k: 0)
        XCTAssertTrue(zeroK.isEmpty, "k<=0 returns []")

        let (s2, url2) = try freshStore()
        defer { try? FileManager.default.removeItem(at: url2) }
        let empty = VectorStore(store: s2, capBytes: 1 << 20)
        let none = await empty.query(vector: [1, 0, 0], k: 5)
        XCTAssertTrue(none.isEmpty, "empty store returns []")
    }
}
