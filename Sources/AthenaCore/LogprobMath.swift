import Foundation

/// One generated token's logprob record (ADR 013 §4 / C2). MLX-free + Sendable
/// so it crosses the actor boundary and the server layer reads it without any
/// MLX dependency. `token`/`bytes` are filled by the decode path (it owns the
/// tokenizer); the numeric `logprob`/`top` come from `LogprobMath`.
public struct TokenLogprob: Sendable, Equatable {
    public let token: String
    public let logprob: Float
    public let bytes: [Int]
    public let top: [TopLogprob]
    public init(token: String, logprob: Float, bytes: [Int], top: [TopLogprob]) {
        self.token = token
        self.logprob = logprob
        self.bytes = bytes
        self.top = top
    }
}

/// One `top_logprobs` alternative for a generated position.
public struct TopLogprob: Sendable, Equatable {
    public let token: String
    public let logprob: Float
    public let bytes: [Int]
    public init(token: String, logprob: Float, bytes: [Int]) {
        self.token = token
        self.logprob = logprob
        self.bytes = bytes
    }
}

/// Pure logprob math over a raw logit row — the MLX-free reference for the
/// decode paths' per-token logprob capture (ADR 009: the decision/algebra is
/// unit-pinned; the real paths compute the same thing with MLX kernels over the
/// full-vocab `MLXArray`, and the stub engine uses this directly on a small
/// synthetic vocab). `log_softmax(x)[i] = x[i] − logSumExp(x)`, so the top-K by
/// logprob is exactly the top-K by raw logit; the shared offset (−logSumExp)
/// preserves the ordering.
public enum LogprobMath {
    /// - Parameters:
    ///   - row: the full logit row (one position, length = vocab).
    ///   - chosen: the selected token id (its logprob is always returned).
    ///   - topK: how many top alternatives to return (0 ⇒ none); clamped to
    ///     `[0, row.count]`.
    /// - Returns: the chosen token's logprob and the top-K `(id, logprob)`
    ///   pairs, descending by logprob (ties broken by lower id). An empty row
    ///   ⇒ `(0, [])`; an out-of-range `chosen` ⇒ logprob 0.
    public static func fromLogitRow(
        _ row: [Float], chosen: Int, topK: Int
    ) -> (logprob: Float, top: [(token: Int, logprob: Float)]) {
        guard !row.isEmpty else { return (0, []) }
        // logSumExp, shifted by the max for numerical stability.
        let m = row.max()!
        var sumExp: Float = 0
        for x in row { sumExp += Foundation.exp(x - m) }
        let lse = m + Foundation.log(sumExp)

        let chosenLp =
            (chosen >= 0 && chosen < row.count) ? row[chosen] - lse : 0

        let k = max(0, min(topK, row.count))
        guard k > 0 else { return (chosenLp, []) }
        // Indices of the k largest logits, descending; ties → lower index.
        let ranked = row.indices.sorted { a, b in
            row[a] != row[b] ? row[a] > row[b] : a < b
        }
        let top = ranked.prefix(k).map {
            (token: $0, logprob: row[$0] - lse)
        }
        return (chosenLp, top)
    }
}
