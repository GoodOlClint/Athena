import Foundation

/// ADR 029 — the process-global **inference execution gate**: one
/// Metal-executing tenant at a time.
///
/// The `MemoryGovernor` enforces the ADR-011 single slot only as a *memory*
/// reservation per module class; it does **not** serialize execution. So an LLM
/// decode and an audio transcription (or a warm rebind racing an in-flight
/// decode) can drive MLX kernels on the one Metal pool concurrently — the
/// "never compose at the inference layer" hazard. This gate adds the missing
/// *execution* exclusivity: every Metal-executing op AND every model
/// rebind/load-swap runs through `withExclusiveExecution`, so only one holds the
/// device at a time. It composes with — and is orthogonal to — the governor.
///
/// A FIFO async semaphore (fair, no starvation), acquired for the execution
/// span only (NOT across the governor's cold-load wait — that is I/O, not
/// execution). Cancellation-aware: a queued waiter whose task is cancelled
/// leaves the queue with a `CancellationError` and never acquires.
///
/// MLX-free and unit-pinned (ADR 008/009): the serialization/FIFO/cancellation
/// invariants are tested without a Metal device.
public actor InferenceGate {
    public static let shared = InferenceGate()
    public init() {}

    /// Default-on revert knob (ADR 029). When `false`, `withExclusiveExecution`
    /// runs the work directly with no serialization — the pre-029 behavior.
    /// Write-once at daemon boot before any request (a plain global, like
    /// `GovernorMemory.serveCacheBounded`).
    public nonisolated(unsafe) static var enabled = true

    private var held = false
    private var waiters: [(ticket: UInt64, cont: CheckedContinuation<Void, Error>)] = []
    private var nextTicket: UInt64 = 0

    /// Run `work` under exclusive Metal execution. `work` runs in the CALLER's
    /// task context (so its cancellation propagates), with the gate held for its
    /// whole span and released on return/throw/cancel. nonisolated so `work`
    /// does not serialize on this actor — only the acquire/release bookkeeping
    /// does.
    nonisolated public func withExclusiveExecution<T: Sendable>(
        _ work: @Sendable () async throws -> T
    ) async throws -> T {
        guard Self.enabled else { return try await work() }
        try await acquire()
        // Release inline (awaited) on BOTH exits so the gate is provably free
        // before we return — a `defer` can't await, and a fire-and-forget
        // `Task` would leave the gate transiently "held" after we return. A
        // cancelled task still runs these actor hops (cancellation is
        // cooperative; an actor call doesn't auto-throw), so the gate never
        // leaks on cancel.
        do {
            let result = try await work()
            await release()
            return result
        } catch {
            await release()
            throw error
        }
    }

    /// Acquire the gate, suspending FIFO behind any current holder + waiters.
    func acquire() async throws {
        guard Self.enabled else { return }
        try Task.checkCancellation()
        if !held && waiters.isEmpty {
            held = true
            return
        }
        let ticket = nextTicket
        nextTicket &+= 1
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation {
                (cont: CheckedContinuation<Void, Error>) in
                // Runs synchronously on this actor → check-then-enqueue is
                // atomic against release()/other acquires.
                waiters.append((ticket, cont))
            }
        } onCancel: {
            Task { await self.cancelWaiter(ticket) }
        }
    }

    /// Release the gate: hand it to the next FIFO waiter, or mark it free.
    func release() {
        guard Self.enabled else { return }
        if waiters.isEmpty {
            held = false
            return
        }
        // Hand off: `held` stays true — ownership moves to the resumed waiter.
        let next = waiters.removeFirst()
        next.cont.resume()
    }

    private func cancelWaiter(_ ticket: UInt64) {
        guard let i = waiters.firstIndex(where: { $0.ticket == ticket }) else {
            return  // already resumed (acquired) — its work checks cancellation
        }
        let w = waiters.remove(at: i)
        w.cont.resume(throwing: CancellationError())
    }

    /// Test-only: number of queued waiters (not the holder).
    var waiterCount: Int { waiters.count }
    /// Test-only: whether the gate is currently held.
    var isHeld: Bool { held }
}
