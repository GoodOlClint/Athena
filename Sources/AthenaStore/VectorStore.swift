import Foundation
import MLX

/// Built-in vector DB: durable rows in `AthenaStore` (SQLite) + a
/// resident MLX matrix for cosine top-k. The resident working set is
/// MLX so the memory governor can rule it via `capBytes` (operator
/// intent). Internal store is the default; an external vector DB is an
/// opt-in *replacement* (not wired here). M7.2.
public actor VectorStore {
    public struct Hit: Sendable, Codable {
        public let id: String
        public let score: Float
        public let metadata: Data?
    }
    public struct Stats: Sendable, Codable {
        public let count: Int
        public let dim: Int
        public let bytes: Int
        public let capBytes: Int
    }
    public enum VectorError: Error, CustomStringConvertible {
        case dimMismatch(expected: Int, got: Int)
        case capExceeded(requestedBytes: Int, capBytes: Int)
        public var description: String {
            switch self {
            case let .dimMismatch(e, g):
                return "vector dim \(g) ≠ store dim \(e)"
            case let .capExceeded(r, c):
                return "vector store cap: needs \(r) B, cap \(c) B"
            }
        }
    }

    private let store: AthenaStore
    private let capBytes: Int
    /// In-memory mirror (id-ordered) of the persisted vectors; the
    /// query matrix is rebuilt from it lazily.
    private var cache: [(id: String, vec: [Float], meta: Data?)] = []
    private var loaded = false

    public init(store: AthenaStore, capBytes: Int) {
        self.store = store
        self.capBytes = capBytes
    }

    private func ensureLoaded() async {
        if loaded { return }
        cache = await store.allVectors().map {
            (id: $0.id, vec: $0.vector, meta: $0.metadata)
        }
        loaded = true
    }

    private var dim: Int { cache.first?.vec.count ?? 0 }

    private func residentBytes(adding n: Int = 0) -> Int {
        (cache.count + n) * (dim == 0 ? 0 : dim) * 4
    }

    public func upsert(
        id: String, vector: [Float], metadata: Data?
    ) async throws {
        await ensureLoaded()
        if let d = cache.first?.vec.count, d != vector.count {
            throw VectorError.dimMismatch(expected: d, got: vector.count)
        }
        let isNew = !cache.contains { $0.id == id }
        if isNew {
            let need = (cache.count + 1) * vector.count * 4
            if need > capBytes {
                throw VectorError.capExceeded(
                    requestedBytes: need, capBytes: capBytes)
            }
        }
        try await store.putVector(
            id: id, vector: vector, metadata: metadata)
        if let i = cache.firstIndex(where: { $0.id == id }) {
            cache[i] = (id, vector, metadata)
        } else {
            cache.append((id, vector, metadata))
            cache.sort { $0.id < $1.id }
        }
    }

    @discardableResult
    public func delete(id: String) async -> Bool {
        await ensureLoaded()
        let ok = await store.deleteVector(id: id)
        // NH2 (M66.1): only drop the row from the resident cache when the
        // persisted delete actually succeeded. `deleteVector` returns false
        // both for an absent id and for a genuine SQLite failure; clearing
        // the cache unconditionally would, on a real failure, desync the
        // cache from the store for the actor's lifetime (queries omit a row
        // still on disk; a re-upsert is miscounted as new).
        if ok { cache.removeAll { $0.id == id } }
        return ok
    }

    /// Cosine top-`k`. Scores computed on MLX (governed working set);
    /// ranked in Swift (k ≪ N) to avoid MLX top-k API churn.
    public func query(vector q: [Float], k: Int) async -> [Hit] {
        await ensureLoaded()
        guard !cache.isEmpty, q.count == dim, k > 0 else { return [] }
        let n = cache.count
        let flat = cache.flatMap { $0.vec }
        let m = MLXArray(flat, [n, dim])
        let qv = MLXArray(q, [dim, 1])
        let mNorm =
            m
            / MLX.sqrt(
                (m * m).sum(axis: 1, keepDims: true) + 1e-12)
        let qNorm =
            qv / MLX.sqrt((qv * qv).sum() + 1e-12)
        let scores = MLX.matmul(mNorm, qNorm).reshaped([n])
        scores.eval()
        let s = scores.asArray(Float.self)
        // End-of-query allocator-pool flush (M50.5). Each query builds
        // an N×dim resident matrix + norms + matmul intermediates; over
        // sustained search load those accumulate in MLX's pool exactly
        // like the embedder did pre-M46.6. `s` is already a Swift
        // [Float] so the ranking below doesn't need the MLXArrays.
        MLX.Memory.clearCache()
        return zip(cache, s)
            .map { Hit(id: $0.0.id, score: $0.1, metadata: $0.0.meta) }
            .sorted { $0.score > $1.score }
            .prefix(k)
            .map { $0 }
    }

    /// Age-based retention (M34.2): delete persisted vectors whose
    /// last-write time is older than `cutoff`, then drop the resident
    /// cache so the next query rebuilds the matrix without the pruned
    /// rows. Returns the count removed. Driven opportunistically by the
    /// server on each upsert when a vector TTL is configured.
    @discardableResult
    public func sweepExpired(olderThan cutoff: Double) async -> Int {
        let removed = (try? await store.pruneVectors(olderThan: cutoff)) ?? 0
        if removed > 0 {
            cache = []
            loaded = false
        }
        return removed
    }

    public func stats() async -> Stats {
        await ensureLoaded()
        return Stats(
            count: cache.count, dim: dim,
            bytes: residentBytes(), capBytes: capBytes)
    }
}
