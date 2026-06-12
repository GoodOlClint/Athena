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
        case ownerConflict(id: String)
        public var description: String {
            switch self {
            case let .dimMismatch(e, g):
                return "vector dim \(g) ≠ store dim \(e)"
            case let .capExceeded(r, c):
                return "vector store cap: needs \(r) B, cap \(c) B"
            case let .ownerConflict(id):
                return "vector '\(id)' is owned by another principal"
            }
        }
    }

    /// The authenticated caller for an owner-scoped op (H5 / M66.6 / ADR
    /// 006). `enforced` = auth on; `isAdmin` ⇒ sees across owners; auth-off
    /// uses `.unscoped` (single trusted operator sees everything).
    public struct Caller: Sendable {
        public let principal: String?
        public let isAdmin: Bool
        public let enforced: Bool
        public init(principal: String?, isAdmin: Bool, enforced: Bool) {
            self.principal = principal
            self.isAdmin = isAdmin
            self.enforced = enforced
        }
        public static let unscoped = Caller(
            principal: nil, isAdmin: true, enforced: false)
    }

    private let store: AthenaStore
    private let capBytes: Int
    /// In-memory mirror (id-ordered) of the persisted vectors, each with
    /// its owner; the query matrix is rebuilt from the caller-visible
    /// subset lazily.
    private var cache:
        [(id: String, vec: [Float], meta: Data?, owner: String?)] = []
    private var loaded = false

    public init(store: AthenaStore, capBytes: Int) {
        self.store = store
        self.capBytes = capBytes
    }

    private func ensureLoaded() async {
        if loaded { return }
        cache = await store.allVectors().map {
            (id: $0.id, vec: $0.vector, meta: $0.metadata, owner: $0.owner)
        }
        loaded = true
    }

    /// H5: may `caller` see a row owned by `rowOwner`? Admin / auth-off
    /// see all; a scoped caller sees only its own rows; a legacy NULL-owner
    /// row is admin-only (never matches a scoped caller).
    private func canSee(_ rowOwner: String?, _ caller: Caller) -> Bool {
        if !caller.enforced || caller.isAdmin { return true }
        guard let rowOwner else { return false }
        return rowOwner == caller.principal
    }

    private var dim: Int { cache.first?.vec.count ?? 0 }

    public func upsert(
        id: String, vector: [Float], metadata: Data?,
        caller: Caller = .unscoped
    ) async throws {
        await ensureLoaded()
        if let d = cache.first?.vec.count, d != vector.count {
            throw VectorError.dimMismatch(expected: d, got: vector.count)
        }
        // H5: an upsert to an id another principal owns must NOT silently
        // re-own / overwrite it (the cross-principal-overwrite half of the
        // finding). Reject; a fresh id is fine.
        let existing = cache.first { $0.id == id }
        if let existing, !canSee(existing.owner, caller) {
            throw VectorError.ownerConflict(id: id)
        }
        let isNew = existing == nil
        if isNew {
            let need = (cache.count + 1) * vector.count * 4
            if need > capBytes {
                throw VectorError.capExceeded(
                    requestedBytes: need, capBytes: capBytes)
            }
        }
        // Auth-off keeps NULL owner (legacy/shared); an enforced caller
        // stamps its principal.
        let owner = caller.enforced ? caller.principal : nil
        try await store.putVector(
            id: id, vector: vector, metadata: metadata, owner: owner)
        if let i = cache.firstIndex(where: { $0.id == id }) {
            cache[i] = (id, vector, metadata, owner)
        } else {
            cache.append((id, vector, metadata, owner))
            cache.sort { $0.id < $1.id }
        }
    }

    @discardableResult
    public func delete(id: String, caller: Caller = .unscoped) async -> Bool
    {
        await ensureLoaded()
        // H5: a non-owner gets the same false (⇒ 404) as a missing id, so
        // existence of another tenant's vector isn't revealed.
        if let existing = cache.first(where: { $0.id == id }),
            !canSee(existing.owner, caller)
        {
            return false
        }
        // Defense-in-depth: scope the persisted delete too (nil = admin /
        // auth-off see all).
        let scope =
            (caller.isAdmin || !caller.enforced) ? nil : caller.principal
        let ok = await store.deleteVector(id: id, owner: scope)
        // NH2 (M66.1): only drop the row from the resident cache when the
        // persisted delete actually succeeded — else a real SQLite failure
        // would desync the cache from the store for the actor's lifetime.
        if ok { cache.removeAll { $0.id == id } }
        return ok
    }

    /// Cosine top-`k` over the caller's VISIBLE vectors (H5). Scores
    /// computed on MLX (governed working set); ranked in Swift (k ≪ N) to
    /// avoid MLX top-k API churn.
    public func query(
        vector q: [Float], k: Int, caller: Caller = .unscoped
    ) async -> [Hit] {
        await ensureLoaded()
        let visible = cache.filter { canSee($0.owner, caller) }
        guard !visible.isEmpty, q.count == dim, k > 0 else { return [] }
        let n = visible.count
        let flat = visible.flatMap { $0.vec }
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
        return zip(visible, s)
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

    /// Store stats over the caller's VISIBLE vectors (H5); `capBytes` is
    /// the shared global cap.
    public func stats(caller: Caller = .unscoped) async -> Stats {
        await ensureLoaded()
        let visible = cache.filter { canSee($0.owner, caller) }
        let d = dim
        return Stats(
            count: visible.count, dim: d,
            bytes: visible.count * (d == 0 ? 0 : d) * 4,
            capBytes: capBytes)
    }
}
