import Foundation

/// M46.8 — per-token progress signal for decode loops that don't stream
/// individual token events back to the serve path.
///
/// The structured-output path (`GuidedSubstrate`, which also serves
/// logprob capture — and, pre-publication-S0, the vendored speculative
/// loops) runs a fully synchronous internal loop inside `container.perform { ... }`,
/// then emit ONE `.text` event at completion with the whole decoded
/// string. From `collectMetered`'s point of view, no `.text` events
/// arrive until the loop finishes — so the M46.1/M46.7 heartbeat sees
/// `tokens=0` for the entire decode and reports `tokens_per_sec=0.0`.
/// That's the "alive but no progress visible" failure mode a downstream client
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

    /// M49.3 — annotate the current setup sub-stage so the heartbeat
    /// reports `phase=setup:<stage>` (e.g. `setup:compile-dfa`,
    /// `setup:build-vocab`) instead of just `phase=setup`. The setup
    /// phase covers chat prep, prompt tokenization, vocab-token build,
    /// structured-index DFA compile, and KV-cache init — and, since #47,
    /// **the prefill too**: no producer publishes prefill counts any more, so
    /// `phase=setup` persists until the first committed token. Read a long
    /// `phase=setup` as "still in setup OR prefilling", not as "stuck in the
    /// DFA compile". A "stuck-in-setup"
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
    public func setSetupStage(_ stage: String?) {}
    public func cancelGeneration() {}
    public var isCancelled: Bool { false }
}

public enum DecodeProgress {
    /// Set by `AthenaServer.collectMetered` for the lifetime of one
    /// metered generation; read by the synchronous decode loop in
    /// `AthenaLLM` (`GuidedSubstrate`, incl. logprob capture). nil ⇒ no
    /// counter is interested in
    /// per-iteration progress (e.g. the substrate-streamed path is
    /// already incrementing via `.text` events).
    @TaskLocal public static var counter: (any DecodeProgressCounter)?
}

/// M49.2 — coarse phase of a metered generation as observed by the
/// heartbeat. Two states: setup (request prep, DFA compile, vocab build —
/// **and the prompt prefill**) and decode (committing tokens). Lets
/// `athena logs` answer "is it hung or just running long?" at a glance.
///
/// There used to be a third, `.prefill`, fed by `recordPrefillChunk`.
/// Publication S0 deleted the only producer, and #47 removed the rest:
/// `GuidedSubstrate` hands prefill to the substrate's
/// `TokenIterator(prefillStepSize:)`, which exposes no per-chunk callback, so
/// there is nowhere to hook one. Checked against the mlx tracker on
/// 2026-08-01 — no upstream work is scheduled that would restore a producer,
/// so the case was deleted rather than kept as unreachable public API.
/// Restoring `phase=prefill` (and the `prefill=n/m` heartbeat field) requires
/// an upstream per-chunk hook first.
///
/// Consequence for reading a heartbeat: a long `phase=setup` means "still in
/// setup OR prefilling". On a large prompt, prefill is the likely answer.
///
/// Raw values match the log field convention (lowercase enum case).
public enum DecodePhase: String, Sendable, CaseIterable {
    case setup
    case decode

    /// Compute the current phase from a counter snapshot: any committed
    /// token means we are decoding; otherwise we are still in setup (which
    /// now includes the prefill — see the type doc).
    public static func from(tokens: Int) -> DecodePhase {
        tokens > 0 ? .decode : .setup
    }
}
