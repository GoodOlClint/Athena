import Foundation

/// NC5 (M70.3) — the cancellation early-break predicate the four AthenaLLM
/// decode loops poll.
///
/// `SpeculativeGeneration`, `SpeculativeSamplingGenerate`, `GuidedGreedy`, and
/// `GuidedSubstrate` each ran `if DecodeProgress.counter?.isCancelled == true {
/// break }` at the top of their decode loop (M60.5) to free the GPU on a
/// client disconnect / M33 deadline — the whole point of the milestone (the
/// original M60 wedge symptom was decoding to `maxTokens` for a dead request).
/// Those four copies had no test: the only `DecodeProgressCounter` test stub
/// never overrode `isCancelled`, so it picked up the protocol default `false`
/// and the cancel branch was never exercised. A refactor that dropped or
/// inverted the check would regress the abort-on-cancellation invariant with
/// green CI.
///
/// Centralizing the check here (a) makes the four loops share ONE predicate
/// (they can't drift) and (b) makes it CI-testable with a stub counter whose
/// flag flips, against the loop idiom, with no MLX. The `MLXArray` decode body
/// stays in the loops; only the boolean stop decision lives here.
public enum DecodeLoopControl {
    /// True iff the in-flight generation has been asked to stop (disconnect /
    /// deadline) via `DecodeProgress.counter?.cancelGeneration()`. `nil`
    /// counter (no metered context) ⇒ `false`, the production default outside
    /// `collectMetered`.
    public static func isCancelled() -> Bool {
        // ADR 029 / M1 — the counter (a TaskLocal set by `collectMetered`) is
        // the primary disconnect/deadline signal, but it only works because the
        // generate Task is created INSIDE the `withValue` scope so it inherits
        // the value. `Task.isCancelled` is a backstop that does not depend on
        // that inheritance, so a cancelled request still stops the decode (and
        // frees the inference slot) even if a future refactor moves the Task
        // out of the scope or the metered context is absent.
        DecodeProgress.counter?.isCancelled == true || Task.isCancelled
    }
}
