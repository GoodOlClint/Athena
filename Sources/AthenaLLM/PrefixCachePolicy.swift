import Foundation

/// NC4 (M70.3) — the prefix-cache eviction policy, MLX-free.
///
/// `PrefixKVCache.Entry` couples the eviction-relevant metadata
/// (`byteEstimate`/`lastUsed`/`refcount`) to the `MLXArray`/`KVCache` tensors
/// it caches, and the only way to build one is `store` (which needs a real
/// backbone) — so the count/byte/idle/refcount eviction ORDER, the
/// load-bearing MEMORY invariant, had no CI coverage. The decisions are pure
/// metadata math; this is the single source the instance methods route
/// through (so production can't drift from the tested policy) and it operates
/// on a tensor-free metadata view, so it runs under `swift test`.
enum PrefixCachePolicy {

    /// Metadata-only view of one entry, in `entries` order (indices returned
    /// below are positions in that array).
    struct Meta {
        var byteEstimate: Int
        var lastUsed: Date
        var refcount: Int
    }

    /// Idle-TTL sweep: indices of entries that are NOT in use (`refcount==0`)
    /// AND have been idle longer than `idleTTL`. `idleTTL <= 0` disables the
    /// sweep (returns none). Mirrors the prior `removeAll` predicate (a single
    /// pass — all expired entries at once).
    static func idleVictimIndices(
        _ metas: [Meta], now: Date, idleTTL: TimeInterval
    ) -> [Int] {
        guard idleTTL > 0 else { return [] }
        return metas.indices.filter {
            metas[$0].refcount == 0
                && now.timeIntervalSince(metas[$0].lastUsed) > idleTTL
        }
    }

    /// Over EITHER cap: count > maxEntries, or (when `maxBytes > 0`) total
    /// bytes > maxBytes. `maxBytes == 0` ⇒ byte cap inert (count-only).
    static func isOverCap(
        _ metas: [Meta], maxEntries: Int, maxBytes: Int
    ) -> Bool {
        if metas.count > maxEntries { return true }
        if maxBytes > 0 {
            return metas.reduce(0) { $0 + $1.byteEstimate } > maxBytes
        }
        return false
    }

    /// The single LRU victim (lowest `lastUsed` among `refcount==0`) WHILE
    /// over a cap; `nil` when under both caps or nothing is evictable (every
    /// remaining entry is in use). The caller loops, removing one and
    /// re-evaluating — identical to the prior `while overCap()` body. On a
    /// `lastUsed` tie the lowest index wins (ascending `indices` + `min`),
    /// matching the prior `enumerated().min`.
    static func lruVictimIndex(
        _ metas: [Meta], maxEntries: Int, maxBytes: Int
    ) -> Int? {
        guard isOverCap(metas, maxEntries: maxEntries, maxBytes: maxBytes)
        else { return nil }
        return metas.indices
            .filter { metas[$0].refcount == 0 }
            .min(by: { metas[$0].lastUsed < metas[$1].lastUsed })
    }

    /// flushIdle / pressure relief: indices of every entry not currently in
    /// use. In-flight entries (`refcount > 0`) survive.
    static func flushIdleIndices(_ metas: [Meta]) -> [Int] {
        metas.indices.filter { metas[$0].refcount == 0 }
    }
}
