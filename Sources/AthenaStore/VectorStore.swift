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
        /// H10 — a zero-length vector is always invalid: it carries no
        /// embedding and, on an empty store, would silently become the
        /// authoritative `dim` (0), making every later dim-check vacuous.
        case zeroLengthVector
        public var description: String {
            switch self {
            case let .dimMismatch(e, g):
                return "vector dim \(g) ≠ store dim \(e)"
            case let .capExceeded(r, c):
                return "vector store cap: needs \(r) B, cap \(c) B"
            case let .ownerConflict(id):
                return "vector '\(id)' is owned by another principal"
            case .zeroLengthVector:
                return "vector has zero length"
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
    /// H8 (M69.3) — id → its index in `cache`, so `upsert` resolves an
    /// existing row in O(1) instead of two O(n) `firstIndex`/`first` scans.
    /// Rebuilt on load and on any structural mutation (`cache` is no longer
    /// kept id-sorted — query ranks by score with an id tie-break, so cache
    /// order is irrelevant to results; this drops the per-upsert O(n log n)
    /// `cache.sort`).
    private var idIndex: [String: Int] = [:]
    /// H3 (M69.3) — the resident ROW-NORMALIZED `n × dim` matrix (in `cache`
    /// order), cached so a query doesn't rebuild a full `n × dim` MLXArray via
    /// `flatMap` and re-normalize every row on EVERY call (~1 GB/query at
    /// 100k×2560). Invalidated (set nil) on any cache mutation and rebuilt
    /// lazily by the next query. Owner-scoping (H5) is applied AFTER scoring
    /// in Swift, so one matrix serves every caller (the common
    /// admin/auth-off/single-owner case scores nothing it can't see anyway).
    private var normMatrix: MLXArray?
    /// H4 (M68.3) — memoizes the one-time cache hydration as a Task so
    /// concurrent first-touch callers COALESCE onto a single
    /// `store.allVectors()` load instead of each running it and clobbering
    /// the others' result (the nil-check + assignment below are actor-atomic,
    /// no `await` between). Cleared once loaded.
    private var loadTask: Task<Void, Never>?
    /// H4 — count of NEW upserts whose `store.putVector` is in flight but not
    /// yet reflected in `cache`. The cap check straddles the `putVector`
    /// await, so two concurrent new upserts both saw the same `cache.count`
    /// and both passed a per-row cap check that only one should have — the
    /// store then overran `capBytes`. Reserving the slot synchronously (before
    /// the await) makes a concurrent upsert see this in-flight count and be
    /// rejected, so the cap holds without serializing every upsert.
    private var pendingNew = 0

    public init(store: AthenaStore, capBytes: Int) {
        self.store = store
        self.capBytes = capBytes
    }

    private func ensureLoaded() async {
        if loaded { return }
        if let t = loadTask {
            await t.value
            return
        }
        let t = Task { [self] in
            let rows = await store.allVectors()
            cache = rows.map {
                (id: $0.id, vec: $0.vector, meta: $0.metadata,
                    owner: $0.owner)
            }
            rebuildIdIndex()
            normMatrix = nil  // H3 — rebuilt lazily by the next query
            loaded = true
        }
        loadTask = t
        await t.value
        loadTask = nil
    }

    /// H8 — rebuild `idIndex` from the current `cache` order (after a load or
    /// a structural mutation that shifted indices, e.g. a delete).
    private func rebuildIdIndex() {
        idIndex.removeAll(keepingCapacity: true)
        for (i, row) in cache.enumerated() { idIndex[row.id] = i }
    }

    /// H5: may `caller` see a row owned by `rowOwner`? Admin / auth-off
    /// see all; a scoped caller sees only its own rows; a legacy NULL-owner
    /// row is admin-only (never matches a scoped caller).
    static func canSee(_ rowOwner: String?, _ caller: Caller) -> Bool {
        if !caller.enforced || caller.isAdmin { return true }
        guard let rowOwner else { return false }
        return rowOwner == caller.principal
    }

    /// H5 + H7 ranking core (pure, no MLX): owner-filter the scored rows, sort
    /// by score DESC with an ascending-id tie-break so equal-score ties are
    /// STABLE (the pre-H7 `sorted` was unstable → ties reordered run-to-run),
    /// then take the top `k`. Extracted from `query` so the
    /// ranking/tie-break/owner-scoping contract is unit-testable on CI without
    /// a Metal device (M70.2 L3); `query` calls it with the MLX-computed
    /// `scores` (one per cache row, in cache order).
    static func rankTopK(
        rows: [(id: String, vec: [Float], meta: Data?, owner: String?)],
        scores: [Float], k: Int, caller: Caller
    ) -> [Hit] {
        var hits: [Hit] = []
        hits.reserveCapacity(rows.count)
        for i in rows.indices where canSee(rows[i].owner, caller) {
            hits.append(
                Hit(id: rows[i].id, score: scores[i], metadata: rows[i].meta))
        }
        hits.sort {
            $0.score != $1.score ? $0.score > $1.score : $0.id < $1.id
        }
        return Array(hits.prefix(k))
    }

    private var dim: Int { cache.first?.vec.count ?? 0 }

    public func upsert(
        id: String, vector: [Float], metadata: Data?,
        caller: Caller = .unscoped
    ) async throws {
        await ensureLoaded()
        // H10 — reject a zero-length vector before the empty-cache path can
        // silently adopt it as the authoritative store `dim` (0).
        guard !vector.isEmpty else { throw VectorError.zeroLengthVector }
        if let d = cache.first?.vec.count, d != vector.count {
            throw VectorError.dimMismatch(expected: d, got: vector.count)
        }
        // H5: an upsert to an id another principal owns must NOT silently
        // re-own / overwrite it (the cross-principal-overwrite half of the
        // finding). Reject; a fresh id is fine. H8 — resolve the existing row
        // in O(1) via `idIndex` instead of an O(n) scan.
        let existingIndex = idIndex[id]
        let existing = existingIndex.map { cache[$0] }
        if let existing, !Self.canSee(existing.owner, caller) {
            throw VectorError.ownerConflict(id: id)
        }
        let isNew = existing == nil
        if isNew {
            // H4 — count this row AND any new upserts already in flight
            // (`pendingNew`), then reserve the slot synchronously (before the
            // `putVector` await) so a concurrent new upsert sees the
            // reservation and can't co-pass a cap check only one should.
            let need = (cache.count + pendingNew + 1) * vector.count * 4
            if need > capBytes {
                throw VectorError.capExceeded(
                    requestedBytes: need, capBytes: capBytes)
            }
            pendingNew += 1
        }
        // Auth-off keeps NULL owner (legacy/shared); an enforced caller
        // stamps its principal.
        let owner = caller.enforced ? caller.principal : nil
        do {
            try await store.putVector(
                id: id, vector: vector, metadata: metadata, owner: owner)
        } catch {
            if isNew { pendingNew -= 1 }  // H4 — release the reservation
            throw error
        }
        // H8 — re-resolve via `idIndex` AFTER the `putVector` await (a
        // concurrent same-id upsert may have inserted it while we were
        // suspended; the pre-await `existingIndex` could be stale). O(1) vs
        // the prior post-await O(n) `firstIndex` re-scan.
        if let i = idIndex[id] {
            cache[i] = (id, vector, metadata, owner)
        } else {
            // Append at the end (cache is no longer kept id-sorted; the
            // per-upsert `cache.sort` was O(n log n) and order doesn't affect
            // query results, which rank by score + an id tie-break).
            cache.append((id, vector, metadata, owner))
            idIndex[id] = cache.count - 1
        }
        normMatrix = nil  // H3 — the resident matrix is now stale
        // H4 — the row is now reflected in `cache.count`; drop the reservation.
        if isNew { pendingNew -= 1 }
    }

    @discardableResult
    public func delete(id: String, caller: Caller = .unscoped) async -> Bool
    {
        await ensureLoaded()
        // H5: a non-owner gets the same false (⇒ 404) as a missing id, so
        // existence of another tenant's vector isn't revealed.
        if let existing = cache.first(where: { $0.id == id }),
            !Self.canSee(existing.owner, caller)
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
        if ok {
            cache.removeAll { $0.id == id }
            rebuildIdIndex()  // H8 — indices shifted
            normMatrix = nil  // H3 — resident matrix is now stale
        }
        return ok
    }

    /// Cosine top-`k` over the caller's VISIBLE vectors (H5). Scores
    /// computed on MLX (governed working set); ranked in Swift (k ≪ N) to
    /// avoid MLX top-k API churn.
    public func query(
        vector q: [Float], k: Int, caller: Caller = .unscoped
    ) async -> [Hit] {
        await ensureLoaded()
        let d = dim
        guard !cache.isEmpty, q.count == d, k > 0 else { return [] }
        let n = cache.count
        // H3 — build the ROW-NORMALIZED matrix ONCE over the full cache and
        // keep it resident; reuse it across queries until a mutation
        // invalidates it. Owner-scoping (H5) is applied AFTER scoring so one
        // cached matrix serves every caller.
        let mNorm: MLXArray
        if let cached = normMatrix {
            mNorm = cached
        } else {
            let flat = cache.flatMap { $0.vec }
            let m = MLXArray(flat, [n, d])
            mNorm =
                m / MLX.sqrt((m * m).sum(axis: 1, keepDims: true) + 1e-12)
            mNorm.eval()
            normMatrix = mNorm
        }
        let qv = MLXArray(q, [d, 1])
        let qNorm = qv / MLX.sqrt((qv * qv).sum() + 1e-12)
        let scores = MLX.matmul(mNorm, qNorm).reshaped([n])
        scores.eval()
        let s = scores.asArray(Float.self)
        // M50.5 — trim only the per-query transients (q-norm + matmul output);
        // the resident `mNorm` is held by `normMatrix` (a live reference), so
        // `clearCache` cannot reclaim it — that's the H3 win over rebuilding +
        // re-normalizing the whole matrix on every call.
        MLX.Memory.clearCache()
        // H5 + H7 — keep only the caller's visible rows, ranked by score with
        // an id tie-break so equal-score ties are stable (the prior `sorted`
        // was unstable, so ties reordered run-to-run). Pure ranking core,
        // extracted for CI testability (M70.2 L3).
        return Self.rankTopK(rows: cache, scores: s, k: k, caller: caller)
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
            idIndex.removeAll(keepingCapacity: true)
            normMatrix = nil
            loaded = false
        }
        return removed
    }

    /// Store stats over the caller's VISIBLE vectors (H5); `capBytes` is
    /// the shared global cap.
    public func stats(caller: Caller = .unscoped) async -> Stats {
        await ensureLoaded()
        let visible = cache.filter { Self.canSee($0.owner, caller) }
        let d = dim
        return Stats(
            count: visible.count, dim: d,
            bytes: visible.count * (d == 0 ? 0 : d) * 4,
            capBytes: capBytes)
    }
}
