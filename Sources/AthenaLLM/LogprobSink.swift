import AthenaCore
import Foundation
import MLX

/// Accumulates per-token logprobs during a single (actor-confined) generation
/// (ADR 013 §4 / C2). A reference type so the value-type `GuidedDecoder` and
/// `LogitProcessor` structs all append to one instance. The numeric work reuses
/// the unit-tested `LogprobMath`; token-id → string/bytes decode is deferred to
/// the module (which owns the tokenizer), which reads `committed`.
///
/// Confined to one generation Task (no concurrent access), hence
/// `@unchecked Sendable`.
public final class LogprobSink: @unchecked Sendable {
    /// One emitted token's numeric capture: the chosen id, its logprob, and the
    /// top-K `(id, logprob)` alternatives (descending). Strings are added later.
    public struct Raw: Sendable {
        public let chosen: Int
        public let logprob: Float
        public let top: [(token: Int, logprob: Float)]
    }

    public let topLogprobs: Int
    private var pending: Raw?
    private var stashedRow: [Float]?
    public private(set) var committed: [Raw] = []

    public init(topLogprobs: Int) { self.topLogprobs = max(0, topLogprobs) }

    /// One-shot capture from a `(1, vocab)` logits slice + the chosen token —
    /// the greedy/guided pick paths know both at the selection site.
    public func prepare(slice: MLXArray, chosen: Int) {
        let r = LogprobMath.fromLogitRow(
            slice.asArray(Float.self), chosen: chosen, topK: topLogprobs)
        pending = Raw(chosen: chosen, logprob: r.logprob, top: r.top)
    }

    /// Two-phase capture for the `LogitProcessor` path: stash the row at
    /// `process(logits:)`, finalize with the sampled token at `didSample`.
    public func stash(slice: MLXArray) { stashedRow = slice.asArray(Float.self) }

    public func finalizeStashed(chosen: Int) {
        guard let row = stashedRow else { return }
        let r = LogprobMath.fromLogitRow(row, chosen: chosen, topK: topLogprobs)
        pending = Raw(chosen: chosen, logprob: r.logprob, top: r.top)
        stashedRow = nil
    }

    /// Promote the pending capture to committed — the token was emitted (so an
    /// EOS / dropped pick whose pending is never kept is discarded).
    public func keep() {
        if let p = pending { committed.append(p); pending = nil }
    }

    /// Drop the leading `i` entries to align with a structured response whose
    /// IDLE (pre-JSON) prefix is stripped (mirrors `GuidedGreedy.result`).
    public func sliceFrom(_ i: Int) {
        if i > 0 && i <= committed.count { committed = Array(committed[i...]) }
    }
}
