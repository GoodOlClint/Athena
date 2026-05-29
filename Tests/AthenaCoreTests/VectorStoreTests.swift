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
}
