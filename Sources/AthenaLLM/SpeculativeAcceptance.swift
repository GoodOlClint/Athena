import Foundation

/// L1 (M70.3) — the greedy MTP speculative accept decision, MLX-free.
///
/// `SpeculativeGeneration.generate`'s load-bearing invariant (the most
/// load-bearing in the codebase) is that the emitted token sequence is
/// bit-identical to non-speculative greedy decoding — MTP only changes speed.
/// The numeric half (the backbone's masked-argmax equalling a reference) stays
/// env-gated, but the accept/reject+bonus SEQUENCING that makes the output
/// greedy regardless of what the draft proposes is pure: at each iteration the
/// loop computes `verifyPred` = the backbone's greedy pick after the confirmed
/// token, and
///   - accepts the draft (commits draft, then a bonus greedy pick) iff
///     `draft == verifyPred` — i.e. the draft guessed the greedy token, so
///     committing it commits the greedy token, and the bonus is the next
///     greedy token (the backbone's pick at the draft position);
///   - else rejects (commits `verifyPred`, the greedy token, and discards the
///     draft's downstream via the KV/Mamba rollback).
/// Either branch commits exactly the greedy token(s), so any draft strategy
/// yields the same sequence — speculation is a speedup, not a quality knob.
///
/// This is the single decision the production loop branches on, so it can't
/// drift from `SpeculativeAcceptanceTests`' parity simulation. The bonus is
/// computed lazily INSIDE the accept branch (an MLX `pick`), so this seam
/// deliberately does not take it — forcing it eagerly would add an argmax on
/// every reject.
enum SpeculativeAcceptance {
    /// Accept the draft iff it equals the backbone's greedy verify prediction.
    static func accepts(draft: Int, verifyPred: Int) -> Bool {
        draft == verifyPred
    }
}
