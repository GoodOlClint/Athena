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
    _ body: @Sendable @escaping () async -> T
) async throws -> T {
    guard seconds > 0 else { return await body() }
    return try await withDeadlineNanos(
        UInt64(seconds) * 1_000_000_000,
        timeout: .requestTimedOut(seconds: seconds), body)
}

/// Nanosecond core of ``withInferenceDeadline(seconds:_:)``. Internal so
/// the deadline race can be unit-tested at sub-second granularity without
/// a real one-second sleep.
func withDeadlineNanos<T: Sendable>(
    _ nanos: UInt64, timeout: AthenaError,
    _ body: @Sendable @escaping () async -> T
) async throws -> T {
    try await withThrowingTaskGroup(of: T.self) { group in
        group.addTask { await body() }
        group.addTask {
            try await Task.sleep(nanoseconds: nanos)
            throw timeout
        }
        // First child to finish wins: the body's result, or the timer's
        // throw. Cancelling the group tears down the loser — the body
        // task on timeout (stopping generation), the timer on success.
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
public func deadlineBounded<E: Sendable>(
    seconds: Int, _ source: AsyncStream<E>
) -> AsyncStream<E> {
    guard seconds > 0 else { return source }
    return deadlineBoundedNanos(
        UInt64(seconds) * 1_000_000_000, source)
}

/// Nanosecond core of ``deadlineBounded(seconds:_:)`` — internal for
/// sub-second unit testing.
func deadlineBoundedNanos<E: Sendable>(
    _ nanos: UInt64, _ source: AsyncStream<E>
) -> AsyncStream<E> {
    AsyncStream<E> { continuation in
        let pump = Task {
            for await element in source {
                continuation.yield(element)
            }
            continuation.finish()
        }
        let timer = Task {
            try? await Task.sleep(nanoseconds: nanos)
            pump.cancel()  // ends iteration → source onTermination → cancel
            continuation.finish()  // idempotent if pump already finished
        }
        continuation.onTermination = { _ in
            pump.cancel()
            timer.cancel()
        }
    }
}
