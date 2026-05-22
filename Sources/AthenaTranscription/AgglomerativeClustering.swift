import Foundation

/// Average-linkage (UPGMA) agglomerative clustering over cosine distance,
/// for embedding-based speaker diarization (M25.3). Pure + Sendable —
/// no MLX, unit-testable on synthetic vectors.
///
/// Inputs are speaker embeddings (assumed L2-normalized, so cosine
/// similarity = dot product; distance = 1 − cos ∈ [0, 2]). Either a fixed
/// `numClusters` is requested, or clustering stops when the closest pair
/// of clusters is farther apart than `threshold` (auto speaker count).
public enum AgglomerativeClustering {
    /// - Parameters:
    ///   - embeddings: row vectors (L2-normalized).
    ///   - numClusters: exact cluster count, or nil to auto-stop at
    ///     `threshold`.
    ///   - threshold: cosine-distance stop bound for auto mode
    ///     (merge while the closest pair is ≤ threshold).
    ///   - maxClusters: optional cap on the auto-mode count.
    /// - Returns: a 0-based contiguous label per input row.
    public static func cluster(
        _ embeddings: [[Float]],
        numClusters: Int? = nil,
        threshold: Float = 0.75,
        maxClusters: Int? = nil
    ) -> [Int] {
        let n = embeddings.count
        if n == 0 { return [] }
        if n == 1 { return [0] }

        // Pairwise cosine distance (1 − dot for unit vectors; we
        // normalize defensively in case inputs aren't unit-length).
        var norm = embeddings.map { v -> [Float] in
            let m = max(sqrt(v.reduce(0) { $0 + $1 * $1 }), 1e-9)
            return v.map { $0 / m }
        }
        var dist = [Float](repeating: 0, count: n * n)
        for i in 0..<n {
            for j in (i + 1)..<n {
                var dot: Float = 0
                let a = norm[i]
                let b = norm[j]
                for k in 0..<a.count { dot += a[k] * b[k] }
                let d = 1 - dot
                dist[i * n + j] = d
                dist[j * n + i] = d
            }
        }
        norm.removeAll(keepingCapacity: false)

        var active = [Bool](repeating: true, count: n)
        var size = [Int](repeating: 1, count: n)
        // Representative cluster id per point; merges fold j → i.
        var label = Array(0..<n)
        var activeCount = n

        let target = numClusters.map { max(1, min($0, n)) }

        while activeCount > 1 {
            // Closest active pair.
            var bestI = -1
            var bestJ = -1
            var bestD = Float.greatestFiniteMagnitude
            for i in 0..<n where active[i] {
                for j in (i + 1)..<n where active[j] {
                    let d = dist[i * n + j]
                    if d < bestD {
                        bestD = d
                        bestI = i
                        bestJ = j
                    }
                }
            }
            if bestI < 0 { break }

            // Decide whether to merge this closest pair (checked before
            // merging). Exact count: merge until we reach it. Auto: merge
            // while within `threshold`, but a `maxClusters` cap forces
            // merging past the threshold until the cap is met.
            let shouldMerge: Bool
            if let t = target {
                shouldMerge = activeCount > t
            } else if let cap = maxClusters, activeCount > cap {
                shouldMerge = true
            } else {
                shouldMerge = bestD <= threshold
            }
            if !shouldMerge { break }

            // UPGMA Lance-Williams update: merge bestJ into bestI.
            let si = Float(size[bestI])
            let sj = Float(size[bestJ])
            for k in 0..<n where active[k] && k != bestI && k != bestJ {
                let dik = dist[bestI * n + k]
                let djk = dist[bestJ * n + k]
                let merged = (si * dik + sj * djk) / (si + sj)
                dist[bestI * n + k] = merged
                dist[k * n + bestI] = merged
            }
            size[bestI] += size[bestJ]
            active[bestJ] = false
            for p in 0..<n where label[p] == bestJ { label[p] = bestI }
            activeCount -= 1
        }

        // Compact representative ids → contiguous 0-based labels.
        var remap: [Int: Int] = [:]
        var next = 0
        var out = [Int](repeating: 0, count: n)
        for p in 0..<n {
            if let id = remap[label[p]] {
                out[p] = id
            } else {
                remap[label[p]] = next
                out[p] = next
                next += 1
            }
        }
        return out
    }
}
