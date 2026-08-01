import AthenaStructured
import Foundation

/// L5 / NC6 (M70.3) — the schema-mask seam, MLX-free.
///
/// The guided path forces the next token into the Guide's currently
/// allowed set: take the per-step allowed bitmask, build an
/// additive logit mask (`0` for allowed ids, `-inf` for the rest), add it to
/// the logits, and argmax. The pre-S0 vendored guided-greedy loop and
/// `GuidedSubstrate.GuidedLogitProcessor.process` had their own copies of the
/// bit-unpacking loop; this is the single source so they can't drift, and —
/// being MLX-free — it makes the masking decision unit-testable with scripted
/// off-schema logits (the audit's L5 fix: "scripted off-schema logits through
/// the seam"). The `MLXArray` math (`logits + MLXArray(add)`, `argMax`) stays
/// in the callers; only the pure bit→`[Float]` unpack and the equivalent
/// argmax live here.
enum GuidedMask {

    /// Build the additive logit mask from a Guide allowed-bitmask: index `i`
    /// is `0` (kept) iff bit `i` of byte `i>>3` is set, else `-.infinity`
    /// (suppressed). This is the exact array both MLX callers add to their
    /// last-position logits slice, so it is the production seam.
    static func additiveMask(allowed mask: [UInt8], vocab: Int) -> [Float] {
        var add = [Float](repeating: -.infinity, count: vocab)
        for i in 0 ..< vocab where (mask[i >> 3] >> UInt8(i & 7)) & 1 == 1 {
            add[i] = 0
        }
        return add
    }

    /// Pure mirror of the guided greedy pick: `argmax(logits + additiveMask)`.
    /// Tie-break is the lowest index (strict `>` keeps the first maximum),
    /// matching MLX `argMax`, which returns the first occurrence of the max —
    /// so a CI assertion on this value reflects what the MLX path commits.
    /// A token whose mask bit is clear can never win (its logit is `-inf`),
    /// so the pick is always schema-valid even when the unmasked argmax is
    /// off-schema. With an all-zero mask (nothing allowed) the result is the
    /// lowest index (all `-inf` tie) — the degenerate case the Guide's design
    /// prevents from arising mid-walk.
    static func maskedArgmax(
        logits: [Float], allowed mask: [UInt8], vocab: Int
    ) -> Int {
        let add = additiveMask(allowed: mask, vocab: vocab)
        var best = -Float.infinity
        var idx = 0
        for i in 0 ..< vocab {
            let v = logits[i] + add[i]
            if v > best { best = v; idx = i }
        }
        return idx
    }
}
