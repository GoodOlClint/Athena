import Foundation

/// Pure, model-free math for cross-attention word-timestamp alignment
/// (M26.2). Mirrors `openai/whisper` `timing.py`: per-head softmax over
/// audio frames, normalization across the token axis, a median filter,
/// the head average, then DTW over the negated weight matrix. Kept free
/// of MLX so the alignment math is unit-tested in CI; the model forward
/// that produces the raw weights lives in `WhisperWordAlign`.
enum WordAlignMath {

    /// In-place softmax of each row over the last axis.
    static func softmaxRows(_ m: inout [[Double]]) {
        for r in m.indices {
            guard let mx = m[r].max() else { continue }
            var sum = 0.0
            for c in m[r].indices {
                let e = Foundation.exp(m[r][c] - mx)
                m[r][c] = e
                sum += e
            }
            if sum > 0 { for c in m[r].indices { m[r][c] /= sum } }
        }
    }

    /// Reflect-padded 1-D median filter (numpy `mode="reflect"`), width
    /// odd. Even widths fall through unchanged; rows shorter than the
    /// half-window clamp the pad so no index escapes the row.
    static func medianFilter(_ row: [Double], width: Int) -> [Double] {
        let n = row.count
        guard width > 1, width % 2 == 1, n > 1 else { return row }
        let pad = min(width / 2, n - 1)
        func at(_ i: Int) -> Double {
            if i < 0 { return row[-i] }
            if i >= n { return row[2 * n - 2 - i] }
            return row[i]
        }
        var out = [Double](repeating: 0, count: n)
        for i in 0 ..< n {
            var win = [Double]()
            win.reserveCapacity(2 * pad + 1)
            for k in -pad ... pad { win.append(at(i + k)) }
            win.sort()
            out[i] = win[win.count / 2]
        }
        return out
    }

    /// `weights` = `[head][token][frame]`. Per head: softmax over frames,
    /// normalize each frame column across tokens `(w-mean)/std`, median-
    /// filter each token row over frames. Then average over heads →
    /// `[token][frame]`.
    static func reduceWeights(
        _ weights: [[[Double]]], medfiltWidth: Int
    ) -> [[Double]] {
        let A = weights.count
        guard A > 0, let first = weights.first, !first.isEmpty else {
            return []
        }
        let T = first.count
        let F = first[0].count
        var heads = weights
        for a in 0 ..< A {
            softmaxRows(&heads[a])
            // Normalize each frame column across the token axis.
            for f in 0 ..< F {
                var mean = 0.0
                for t in 0 ..< T { mean += heads[a][t][f] }
                mean /= Double(T)
                var varSum = 0.0
                for t in 0 ..< T {
                    let d = heads[a][t][f] - mean
                    varSum += d * d
                }
                let std = max(Foundation.sqrt(varSum / Double(T)), 1e-8)
                for t in 0 ..< T {
                    heads[a][t][f] = (heads[a][t][f] - mean) / std
                }
            }
            for t in 0 ..< T {
                heads[a][t] = medianFilter(heads[a][t], width: medfiltWidth)
            }
        }
        var matrix = [[Double]](
            repeating: [Double](repeating: 0, count: F), count: T)
        for t in 0 ..< T {
            for f in 0 ..< F {
                var s = 0.0
                for a in 0 ..< A { s += heads[a][t][f] }
                matrix[t][f] = s / Double(A)
            }
        }
        return matrix
    }

    /// Dynamic time warp over a `[N][M]` cost matrix (rows = tokens,
    /// cols = audio frames). Returns the monotonic alignment path as
    /// parallel `(textIndex, timeIndex)` arrays in forward order. Moves:
    /// diagonal (match), up (token advance), left (frame advance).
    static func dtw(_ cost: [[Double]]) -> (text: [Int], time: [Int]) {
        let N = cost.count
        guard N > 0, let row0 = cost.first, !row0.isEmpty else {
            return ([], [])
        }
        let M = row0.count
        let inf = Double.greatestFiniteMagnitude
        var D = [[Double]](
            repeating: [Double](repeating: inf, count: M + 1),
            count: N + 1)
        var trace = [[Int8]](
            repeating: [Int8](repeating: -1, count: M + 1), count: N + 1)
        D[0][0] = 0
        for i in 1 ... N {
            for j in 1 ... M {
                let d0 = D[i - 1][j - 1]  // diagonal
                let d1 = D[i - 1][j]  // up   (token advances)
                let d2 = D[i][j - 1]  // left (frame advances)
                var best = d0
                var t: Int8 = 0
                if d1 < best {
                    best = d1
                    t = 1
                }
                if d2 < best {
                    best = d2
                    t = 2
                }
                D[i][j] = cost[i - 1][j - 1] + best
                trace[i][j] = t
            }
        }
        var i = N
        var j = M
        var ti = [Int]()
        var tj = [Int]()
        while i > 0 && j > 0 {
            ti.append(i - 1)
            tj.append(j - 1)
            switch trace[i][j] {
            case 0:
                i -= 1
                j -= 1
            case 1: i -= 1
            default: j -= 1
            }
        }
        // If the path hit the left wall first, record the remaining
        // tokens at frame 0 so every token row gets a time.
        while i > 0 {
            ti.append(i - 1)
            tj.append(0)
            i -= 1
        }
        return (ti.reversed(), tj.reversed())
    }

    /// First (earliest) frame index each of `n` token rows aligns to,
    /// from a DTW path. Non-decreasing in the token index.
    static func firstFrames(
        textIdx: [Int], timeIdx: [Int], n: Int
    ) -> [Int] {
        var frame = [Int](repeating: 0, count: n)
        var seen = [Bool](repeating: false, count: n)
        for k in textIdx.indices {
            let t = textIdx[k]
            if t >= 0, t < n, !seen[t] {
                frame[t] = timeIdx[k]
                seen[t] = true
            }
        }
        // Fill any unseen rows by carrying the previous frame forward.
        var last = 0
        for t in 0 ..< n {
            if seen[t] { last = frame[t] } else { frame[t] = last }
        }
        return frame
    }

    /// Merge unicode-safe subwords into words and their token ranges.
    /// A new word begins at the first subword, on a leading space, or on
    /// a pure-punctuation subword; otherwise the subword extends the
    /// current word. Mirrors whisper `split_tokens_on_spaces`.
    static func mergeWords(
        _ subwords: [(text: String, range: Range<Int>)]
    ) -> [(word: String, range: Range<Int>)] {
        var words: [(word: String, range: Range<Int>)] = []
        for sw in subwords {
            let trimmed = sw.text.trimmingCharacters(
                in: .whitespacesAndNewlines)
            let withSpace = sw.text.hasPrefix(" ")
            let punctuation =
                !trimmed.isEmpty
                && trimmed.allSatisfy { $0.isPunctuation || $0.isSymbol }
            if words.isEmpty || withSpace || punctuation {
                words.append((sw.text, sw.range))
            } else {
                let last = words.removeLast()
                words.append(
                    (last.word + sw.text,
                        last.range.lowerBound ..< sw.range.upperBound))
            }
        }
        return words
    }
}
