import AthenaStructured
import Foundation
import MLX

/// Deferred structured enforcement (operator-chosen "phase flag in our
/// loop"; resolves issue #2 and underpins tool-aware Patch 4).
///
/// IDLE: the model generates unconstrained (it may emit a
/// `<think>…</think>` or `<tool_call>` prefix); tokens are picked by
/// plain argmax and NOT fed to the Guide. The shim's `advance` only
/// mutates the Guide when the token is a valid transition, so each
/// committed token is probed: the first token that the fresh Guide
/// accepts (the JSON value's opener) flips us to ENFORCING.
/// ENFORCING: picks are Guide-masked and every committed token advances
/// the Guide (monotonic — still no rollback, the named risk stays
/// avoided by construction).
///
/// `jsonStarted` lets the caller return ONLY the enforced JSON span as
/// the response content (the unconstrained prefix is dropped, so
/// `response_format` output stays spec-compliant even with thinking).
/// No guide ⇒ fully unconstrained passthrough (unstructured path).
struct GuidedDecoder {
    let guide: StructuredGuide?
    let vocab: Int
    private(set) var enforcing: Bool
    private(set) var jsonStarted = false
    private var maskBuf: [UInt8] = []

    init(guide: StructuredGuide?, vocab: Int) {
        self.guide = guide
        self.vocab = vocab
        self.enforcing = false
    }

    /// Pick a token id from a `(1, vocab)` logits slice.
    mutating func pick(_ slice: MLXArray) -> Int {
        if let guide, enforcing {
            return SpeculativeGeneration.guidedArgmax(
                slice, vocab: vocab, guide: guide, maskBuf: &maskBuf)
        }
        return argMax(slice, axis: -1).item(Int.self)
    }

    /// Record a committed token; advance the Guide / flip the phase.
    /// Returns true iff this token started the enforced JSON span.
    @discardableResult
    mutating func commit(_ token: Int) -> Bool {
        guard let guide else { return false }
        if enforcing {
            _ = guide.advance(UInt32(token))
            return false
        }
        // IDLE: `advance` returns true (and consumes) only for a valid
        // JSON-start transition; false leaves the Guide untouched.
        if guide.advance(UInt32(token)) {
            enforcing = true
            jsonStarted = true
            return true
        }
        return false
    }
}
