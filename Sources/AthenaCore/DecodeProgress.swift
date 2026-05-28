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
}

extension DecodeProgressCounter {
    public func recordPrefillChunk(completed: Int, total: Int) {}
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
