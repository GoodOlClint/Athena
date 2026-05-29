import Foundation

/// Per-request inference deadline (M33.1). The serve path bounds a single
/// generation by `request_timeout_secs` so a runaway decode is capped by
/// wall-clock, not only by `max_tokens` — the caller gets a classified
/// 504 (`AthenaError.requestTimedOut`) and the work is cancelled so it
/// stops consuming the budget/worker. A non-positive timeout disables the
/// deadline (opt-in, off by default — same posture as the M29 abuse caps).

/// Run `body` under a wall-clock deadline. `seconds <= 0` ⇒ unbounded
/// (runs `body` directly). On expiry it throws
/// `AthenaError.requestTimedOut(seconds:)` and cancels `body`; because the
/// generation streams are `AsyncStream`s whose `onTermination` cancels the
/// underlying model task, that cancellation propagates down to the decode.
public func withInferenceDeadline<T: Sendable>(
    seconds: Int,
    _ body: @Sendable @escaping () async throws -> T
) async throws -> T {
    guard seconds > 0 else { return try await body() }
    return try await withDeadlineNanos(
        UInt64(seconds) * 1_000_000_000,
        timeout: .requestTimedOut(seconds: seconds), body)
}

/// Nanosecond core of ``withInferenceDeadline(seconds:_:)``. Internal so
/// the deadline race can be unit-tested at sub-second granularity without
/// a real one-second sleep.
///
/// M49.5.2 — `body` is now throwing so the consumer can propagate
/// classified errors (e.g. `schemaTooComplex`) without losing them
/// through a non-throwing wrapper. The timer task's own throw of
/// `timeout` continues to race the body's completion exactly as before.
func withDeadlineNanos<T: Sendable>(
    _ nanos: UInt64, timeout: AthenaError,
    _ body: @Sendable @escaping () async throws -> T
) async throws -> T {
    try await withThrowingTaskGroup(of: T.self) { group in
        group.addTask { try await body() }
        group.addTask {
            try await Task.sleep(nanoseconds: nanos)
            throw timeout
        }
        // First child to finish wins: the body's result, or the timer's
        // throw, or the body's own throw. Cancelling the group tears
        // down the loser — the body task on timeout (stopping
        // generation), the timer on success or body-throw.
        let result = try await group.next()!
        group.cancelAll()
        return result
    }
}

/// Wrap a streaming source so it stops after `seconds` (0 ⇒ unbounded).
/// Used for the SSE/NDJSON paths, where the 200 + headers are already
/// committed so a 504 is impossible — instead the deadline truncates the
/// stream cleanly (cancelling the source, which cancels generation), and
/// the caller's finalizer closes the wire (`[DONE]` / `done:true`). The
/// reliability win — bounding a runaway decode — holds for streaming too.
///
/// M46.1 — `onTimerFired` is invoked exactly once when the deadline
/// truncates the stream (timer races ahead of the source completing).
/// AthenaCore stays dependency-free so callers (the daemon target's
/// streaming handlers) inject a `Logger` warning through this closure,
/// keeping the truncation visible to `log show`. The closure is NOT
/// invoked when the source finishes naturally before the deadline.
public func deadlineBounded<E: Sendable>(
    seconds: Int, _ source: AsyncStream<E>,
    onTimerFired: (@Sendable () -> Void)? = nil
) -> AsyncStream<E> {
    guard seconds > 0 else { return source }
    return deadlineBoundedNanos(
        UInt64(seconds) * 1_000_000_000, source,
        onTimerFired: onTimerFired)
}

/// Nanosecond core of ``deadlineBounded(seconds:_:)`` — internal for
/// sub-second unit testing.
func deadlineBoundedNanos<E: Sendable>(
    _ nanos: UInt64, _ source: AsyncStream<E>,
    onTimerFired: (@Sendable () -> Void)? = nil
) -> AsyncStream<E> {
    AsyncStream<E> { continuation in
        // Shared race-state for the timer-vs-pump completion order:
        // whichever wins records itself, and only the timer's
        // "I won" branch fires `onTimerFired`. NSLock-isolated to
        // make the read-modify-write Sendable-safe across the two
        // child tasks.
        let state = StreamTruncationState()
        let pump = Task {
            for await element in source {
                continuation.yield(element)
            }
            state.markPumpFinished()
            continuation.finish()
        }
        let timer = Task {
            try? await Task.sleep(nanoseconds: nanos)
            // If the pump already finished, the stream ran to natural
            // completion — no truncation, no callback. Otherwise the
            // timer is the actual end of the stream and the truncation
            // is real.
            if state.markTimerFiredIfFirst() {
                onTimerFired?()
            }
            pump.cancel()  // ends iteration → source onTermination → cancel
            continuation.finish()  // idempotent if pump already finished
        }
        continuation.onTermination = { _ in
            pump.cancel()
            timer.cancel()
        }
    }
}

/// Tracks the race between the pump (source) finishing and the deadline
/// timer firing so `onTimerFired` fires only on a real truncation, never
/// on a natural completion. `@unchecked Sendable` because all access is
/// `NSLock`-isolated. M46.1 — kept internal to `deadlineBoundedNanos`.
private final class StreamTruncationState: @unchecked Sendable {
    private let lock = NSLock()
    private var pumpFinished = false
    private var timerFired = false

    func markPumpFinished() {
        lock.lock()
        defer { lock.unlock() }
        pumpFinished = true
    }

    /// True iff this call is the first terminal event (the pump hadn't
    /// already finished) — the caller should treat this as a truncation
    /// and fire its log callback. Subsequent calls return false so the
    /// callback can't double-fire even under improbable scheduling.
    func markTimerFiredIfFirst() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        guard !pumpFinished && !timerFired else { return false }
        timerFired = true
        return true
    }
}
