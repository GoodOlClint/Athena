import Foundation

/// M46.8 — per-token progress signal for decode loops that don't stream
/// individual token events back to the serve path.
///
/// The structured-output paths (GuidedGreedy, GuidedSubstrate) and the
/// speculative paths (SpeculativeGeneration, SpeculativeSampling) run a
/// fully synchronous internal loop inside `container.perform { ... }`,
/// then emit ONE `.text` event at completion with the whole decoded
/// string. From `collectMetered`'s point of view, no `.text` events
/// arrive until the loop finishes — so the M46.1/M46.7 heartbeat sees
/// `tokens=0` for the entire decode and reports `tokens_per_sec=0.0`.
/// That's the "alive but no progress visible" failure mode the consuming application
/// hit: the model genuinely IS producing tokens, the operator just
/// can't tell from the log.
///
/// The fix is a single `@TaskLocal` counter that the decode loops
/// increment on every internal commit. The heartbeat task reads from
/// the same counter via `DecodeProgress.counter?.incrementToken()`. No
/// protocol/signature plumbing required across the layers because
/// `TaskLocal` propagates across `await` and into actor calls.
///
/// AthenaCore stays dependency-free — only a protocol declaration here,
/// no `import Logging`, no `import MLX`. The serve path (which has the
/// concrete counter type) sets the TaskLocal; the AthenaLLM decode
/// loops call `incrementToken()` through the protocol.
public protocol DecodeProgressCounter: AnyObject, Sendable {
    /// One token has been committed by the decode loop. Implementations
    /// should be cheap (NSLock-isolated `tokens += 1` is the canonical
    /// shape — the heartbeat reader pays for the lock, not the
    /// per-token writer).
    func incrementToken()

    /// M48.4 — publish prefill chunk progress so the heartbeat can
    /// distinguish "stuck in prefill" from "stuck just-after-prefill"
    /// from "decoding slowly." Decode loops should call this after
    /// each prefill chunk has been submitted (not awaited — MLX's
    /// asyncEval queues the work; this counts "submitted to the
    /// scheduler"). `total` is the total chunk count for THIS
    /// prefill; `completed` is how many have been submitted so far
    /// (1-indexed at first call, equal to `total` at the last call).
    /// Default: no-op so conformers that don't care about prefill
    /// granularity stay valid.
    func recordPrefillChunk(completed: Int, total: Int)

    /// M49.3 — annotate the current setup sub-stage so the heartbeat
    /// reports `phase=setup:<stage>` (e.g. `setup:compile-dfa`,
    /// `setup:build-vocab`) instead of just `phase=setup`. The setup
    /// phase covers everything before the first prefill chunk lands:
    /// chat prep, prompt tokenization, vocab-token build,
    /// structured-index DFA compile, KV-cache init. A "stuck-in-setup"
    /// heartbeat is unactionable without knowing which sub-step is
    /// running — the DFA compile in particular can take minutes for a
    /// large schema on a cache miss. Pass nil to clear the annotation
    /// (typically via `defer`). Default: no-op so conformers that
    /// don't care about setup granularity stay valid.
    func setSetupStage(_ stage: String?)

    /// M60.5 — request that the in-flight generation stop early. Set by the
    /// serve path when it observes task cancellation (a client disconnect, or
    /// the M33 request deadline) so the synchronous decode loops free the GPU
    /// instead of decoding all the way to `maxTokens` for a request no one is
    /// waiting on. Default: no-op.
    func cancelGeneration()

    /// M60.5 — whether `cancelGeneration()` has been called. The decode loops
    /// poll this each iteration and `break`. Default: `false`.
    var isCancelled: Bool { get }
}

extension DecodeProgressCounter {
    public func recordPrefillChunk(completed: Int, total: Int) {}
    public func setSetupStage(_ stage: String?) {}
    public func cancelGeneration() {}
    public var isCancelled: Bool { false }
}

public enum DecodeProgress {
    /// Set by `AthenaServer.collectMetered` for the lifetime of one
    /// metered generation; read by the synchronous decode loops in
    /// `AthenaLLM` (GuidedGreedy, GuidedSubstrate, SpeculativeGeneration,
    /// SpeculativeSampling). nil ⇒ no counter is interested in
    /// per-iteration progress (e.g. the substrate-streamed path is
    /// already incrementing via `.text` events).
    @TaskLocal public static var counter: (any DecodeProgressCounter)?
}

/// M49.2 — coarse phase of a metered generation as observed by the
/// heartbeat. Distinguishes the three operationally-interesting
/// states: setup (waiting on request prep / DFA compile / vocab
/// build), prefill (CPU+GPU on the prompt chunks), and decode
/// (committing tokens). Lets `athena logs` answer "is it hung in
/// setup or just running long?" at a glance, without inferring from
/// "no prefill field yet."
///
/// Derived from the counter snapshot (tokens + prefill completion);
/// no separate state needed. Raw values match the log field
/// convention (lowercase enum case).
public enum DecodePhase: String, Sendable {
    case setup
    case prefill
    case decode

    /// Compute the current phase from a counter snapshot.
    ///
    /// - `tokens > 0` ⇒ `.decode` (at least one token has been
    ///   committed; we're past prefill regardless of whether prefill
    ///   was tracked).
    /// - `prefillTotal > 0 && prefillCompleted < prefillTotal` ⇒
    ///   `.prefill` (chunks submitted but not all done).
    /// - `prefillTotal > 0 && prefillCompleted == prefillTotal` ⇒
    ///   `.decode` (prefill done; decode loop about to commit or
    ///   already between iterations — the transient window is
    ///   sub-second in practice).
    /// - everything else ⇒ `.setup`.
    public static func from(
        tokens: Int, prefillCompleted: Int, prefillTotal: Int
    ) -> DecodePhase {
        if tokens > 0 { return .decode }
        if prefillTotal > 0 {
            return prefillCompleted < prefillTotal
                ? .prefill : .decode
        }
        return .setup
    }
}
