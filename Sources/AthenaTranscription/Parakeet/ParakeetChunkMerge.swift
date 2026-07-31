import Foundation

/// Stitches the per-chunk TDT token streams of a long-audio transcription back
/// into one ordered list (ADR 020 S4). MLX-free — chunks are decoded
/// independently (each `transcribeWindow` call resets decoder state; the
/// overlap region gives the model lead-in context), and merging the two token
/// lists at their time overlap is pure list work — so it is unit-pinned under
/// `swift test` (ADR 008/009).
///
/// Ported from the Python reference (`alignment.py`): try a longest *contiguous*
/// run of matching (id, ~start) token pairs in the overlap; if that fails, fall
/// back to a longest *common subsequence*; if neither finds enough anchors,
/// cut both lists at the overlap midpoint. Matching the reference keeps the
/// stitched timeline faithful to `parakeet-mlx`.
public enum ParakeetChunkMerge {
    private typealias Token = ParakeetAlignment.Token

    /// Merge `a` (accumulated) with `b` (next chunk, already time-offset).
    public static func merge(
        _ a: [ParakeetAlignment.Token], _ b: [ParakeetAlignment.Token],
        overlapDuration: Double
    ) -> [ParakeetAlignment.Token] {
        if a.isEmpty || b.isEmpty { return a.isEmpty ? b : a }
        if let r = longestContiguous(a, b, overlapDuration: overlapDuration) {
            return r
        }
        return longestCommonSubsequence(a, b, overlapDuration: overlapDuration)
    }

    /// True when two overlap tokens are the "same" emission: same vocab id and
    /// start times within half the overlap window.
    private static func aligned(
        _ x: Token, _ y: Token, _ overlapDuration: Double
    ) -> Bool {
        x.id == y.id && abs(x.start - y.start) < overlapDuration / 2
    }

    private static func midpointCut(
        _ a: [Token], _ b: [Token], aEnd: Double, bStart: Double
    ) -> [Token] {
        let cutoff = (aEnd + bStart) / 2
        return a.filter { $0.end <= cutoff } + b.filter { $0.start >= cutoff }
    }

    /// Longest contiguous matching run. Returns nil to signal "not enough
    /// anchors" (the reference raises; the caller then tries LCS).
    private static func longestContiguous(
        _ a: [Token], _ b: [Token], overlapDuration: Double
    ) -> [Token]? {
        let aEnd = a.last!.end
        let bStart = b.first!.start
        if aEnd <= bStart { return a + b }

        let overlapA = a.filter { $0.end > bStart - overlapDuration }
        let overlapB = b.filter { $0.start < aEnd + overlapDuration }
        let enoughPairs = overlapA.count / 2
        if overlapA.count < 2 || overlapB.count < 2 {
            return midpointCut(a, b, aEnd: aEnd, bStart: bStart)
        }

        var best: [(Int, Int)] = []
        for i in 0 ..< overlapA.count {
            for j in 0 ..< overlapB.count
            where aligned(overlapA[i], overlapB[j], overlapDuration) {
                var current: [(Int, Int)] = []
                var k = i, l = j
                while k < overlapA.count, l < overlapB.count,
                    aligned(overlapA[k], overlapB[l], overlapDuration)
                {
                    current.append((k, l))
                    k += 1
                    l += 1
                }
                if current.count > best.count { best = current }
            }
        }

        guard best.count >= enoughPairs, !best.isEmpty else { return nil }
        return assemble(a, b, overlapACount: overlapA.count, pairs: best)
    }

    /// Longest common subsequence of aligned overlap tokens (DP), then assemble.
    /// Never fails — falls back to the midpoint cut when no pairs match.
    private static func longestCommonSubsequence(
        _ a: [Token], _ b: [Token], overlapDuration: Double
    ) -> [Token] {
        let aEnd = a.last!.end
        let bStart = b.first!.start
        if aEnd <= bStart { return a + b }

        let overlapA = a.filter { $0.end > bStart - overlapDuration }
        let overlapB = b.filter { $0.start < aEnd + overlapDuration }
        if overlapA.count < 2 || overlapB.count < 2 {
            return midpointCut(a, b, aEnd: aEnd, bStart: bStart)
        }

        let m = overlapA.count, n = overlapB.count
        var dp = Array(repeating: Array(repeating: 0, count: n + 1), count: m + 1)
        for i in 1 ... m {
            for j in 1 ... n {
                if aligned(overlapA[i - 1], overlapB[j - 1], overlapDuration) {
                    dp[i][j] = dp[i - 1][j - 1] + 1
                } else {
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
                }
            }
        }

        var pairs: [(Int, Int)] = []
        var i = m, j = n
        while i > 0, j > 0 {
            if aligned(overlapA[i - 1], overlapB[j - 1], overlapDuration) {
                pairs.append((i - 1, j - 1))
                i -= 1
                j -= 1
            } else if dp[i - 1][j] > dp[i][j - 1] {
                i -= 1
            } else {
                j -= 1
            }
        }
        pairs.reverse()
        if pairs.isEmpty {
            return midpointCut(a, b, aEnd: aEnd, bStart: bStart)
        }
        return assemble(a, b, overlapACount: overlapA.count, pairs: pairs)
    }

    /// Shared assembly: prefix of `a` before the first anchor, then each anchor
    /// from `a` with the longer of the two gap runs between anchors, then the
    /// suffix of `b` after the last anchor. `pairs` index into the overlap
    /// slices; `overlapACount` maps overlap-A indices back into `a`.
    private static func assemble(
        _ a: [Token], _ b: [Token], overlapACount: Int, pairs: [(Int, Int)]
    ) -> [Token] {
        let aStartIdx = a.count - overlapACount
        let idxA = pairs.map { aStartIdx + $0.0 }
        let idxB = pairs.map { $0.1 }

        var result: [Token] = []
        result.append(contentsOf: a[0 ..< idxA[0]])
        for k in 0 ..< pairs.count {
            result.append(a[idxA[k]])
            if k < pairs.count - 1 {
                let gapA = Array(a[(idxA[k] + 1) ..< idxA[k + 1]])
                let gapB = Array(b[(idxB[k] + 1) ..< idxB[k + 1]])
                result.append(contentsOf: gapB.count > gapA.count ? gapB : gapA)
            }
        }
        result.append(contentsOf: b[(idxB.last! + 1)...])
        return result
    }
}
