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

    // ADR 038 slice 1 — production observability. Until now the gate's depth
    // was test-only (`waiterCount`/`isHeld`), so FIFO starvation — a long
    // transcription or convert ahead of a chat request — was invisible, and
    // there was no evidence base for the batching trigger (routine ≥2-deep
    // contention). These counters back `stats()` → /healthz + /metrics.
    /// Monotonic nanos when the current holder acquired (undefined while free).
    private var heldSinceNanos: UInt64 = 0
    /// High-water queue depth since boot (FIFO starvation is a peak, not a mean).
    private var maxWaiters = 0
    /// Total successful acquisitions (includes uncontended fast-path).
    private var acquisitions = 0
    /// Acquisitions that had to queue (the contention signal).
    private var contended = 0
    /// Bounded reservoir of queue wait times (ms); flat memory, like
    /// `AthenaMetrics`' latency window. p95≈0 ⇒ no contention.
    private var waitSamples: [Double] = []
    private let waitCap = 1024

    /// ADR 038 — a point-in-time read of the gate for /healthz + /metrics.
    public struct GateStats: Sendable {
        public let held: Bool
        public let waiters: Int
        public let heldMs: Double
        public let maxWaiters: Int
        public let acquisitions: Int
        public let contended: Int
        public let waitP50Ms: Double
        public let waitP95Ms: Double
        public let waitMaxMs: Double
    }

    private func recordWait(_ ms: Double) {
        waitSamples.append(ms)
        if waitSamples.count > waitCap {
            waitSamples.removeFirst(waitSamples.count - waitCap)
        }
    }

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
        // ADR 030 Part 2 (WP2) — fresh slate: a fault recorded during THIS span
        // is unambiguously ours (the gate guarantees single-tenant execution).
        MetalFaultLatch.shared.clear()
        // Release inline (awaited) on BOTH exits so the gate is provably free
        // before we return — a `defer` can't await, and a fire-and-forget
        // `Task` would leave the gate transiently "held" after we return. A
        // cancelled task still runs these actor hops (cancellation is
        // cooperative; an actor call doesn't auto-throw), so the gate never
        // leaks on cancel.
        // Capture the outcome first, so the release below is the ONLY one on
        // every path. The previous shape put `try throwIfMetalFaulted()` inside
        // the `do`, so on the success-with-latched-fault path its throw was
        // caught by the attached `catch`, which released a second time for one
        // acquisition — handing the gate to a second waiter alongside the first,
        // or clearing `held` while the real holder still ran. That is the ADR
        // 029 exclusivity violation this type exists to prevent, on the exact
        // path ADR 030 Part 2 added the latch for.
        let outcome: Result<T, Error>
        do {
            outcome = .success(try await work())
        } catch {
            outcome = .failure(error)
        }
        await release()
        // ADR 030 Part 2 — an async allocation fault fired on MLX's worker
        // thread during this span (the handler recorded it instead of aborting
        // the daemon). The eval barrier inside `work` has completed, so the
        // latch is set by now; convert it to a classified 503 so the request
        // degrades and the process stays up.
        //
        // Checked after the release on BOTH paths, and before surfacing the raw
        // error, so a recorded allocation fault still wins over a thrown one —
        // a throw is often a cascade artifact of the decode loop touching the
        // invalid arrays before it broke on the latch.
        try throwIfMetalFaulted()
        return try outcome.get()
    }

    /// ADR 030 Part 2 (WP2) — if a recognized MLX allocation fault was latched
    /// during the just-finished span, consume it and throw a classified 503.
    private nonisolated func throwIfMetalFaulted() throws {
        if let fault = MetalFaultLatch.shared.take() {
            throw AthenaError.metalOutOfMemory(module: nil, detail: fault)
        }
    }

    /// Acquire the gate, suspending FIFO behind any current holder + waiters.
    ///
    /// Deliberately does NOT consult `enabled`: the knob is read exactly once
    /// per span, by `withExclusiveExecution` above, so acquire/release can never
    /// disagree about whether this span owns the gate. Reading it here too made
    /// the pair race a mid-span flip — see `release()`.
    ///
    /// `private`, with `release()`, so that "read once per span" is enforced by
    /// the compiler rather than by this comment: a direct call from elsewhere in
    /// AthenaCore would reintroduce the asymmetry with no guard at either end.
    private func acquire() async throws {
        try Task.checkCancellation()
        if !held && waiters.isEmpty {
            held = true
            heldSinceNanos = DispatchTime.now().uptimeNanoseconds
            acquisitions += 1
            recordWait(0)  // uncontended
            return
        }
        let ticket = nextTicket
        nextTicket &+= 1
        let enqueuedNanos = DispatchTime.now().uptimeNanoseconds
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation {
                (cont: CheckedContinuation<Void, Error>) in
                // Runs synchronously on this actor → check-then-enqueue is
                // atomic against release()/other acquires.
                waiters.append((ticket, cont))
                if waiters.count > maxWaiters { maxWaiters = waiters.count }
            }
        } onCancel: {
            Task { await self.cancelWaiter(ticket) }
        }
        // Resumed ⇒ release() handed the gate to us (`held` stayed true); this
        // is the instant we become the holder. A cancelled waiter throws above
        // and never reaches here, so it counts as neither an acquisition nor a
        // wait sample.
        let now = DispatchTime.now().uptimeNanoseconds
        heldSinceNanos = now
        acquisitions += 1
        contended += 1
        recordWait(Double(now &- enqueuedNanos) / 1e6)
    }

    /// Release the gate: hand it to the next FIFO waiter, or mark it free.
    ///
    /// Never consults `enabled`. It used to, and that stranded the gate: a span
    /// acquired while enabled whose `enabled` flipped `true → false` mid-flight
    /// released into a no-op, leaving `held == true` with no holder — every
    /// later acquire then queued behind a phantom and was never resumed. The
    /// guard was also unreachable on the path it was written for, since
    /// `withExclusiveExecution` returns before releasing when the gate is off.
    private func release() {
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

    /// ADR 038 — the production observability read. Disabled gate (revert knob)
    /// ⇒ all-zero/unheld, since nothing is serialized to observe.
    public func stats() -> GateStats {
        let heldMs =
            held
            ? Double(DispatchTime.now().uptimeNanoseconds &- heldSinceNanos) / 1e6
            : 0
        let sorted = waitSamples.sorted()
        return GateStats(
            held: held, waiters: waiters.count, heldMs: heldMs,
            maxWaiters: maxWaiters, acquisitions: acquisitions,
            contended: contended,
            waitP50Ms: Self.percentile(sorted, 0.5),
            waitP95Ms: Self.percentile(sorted, 0.95),
            waitMaxMs: sorted.last ?? 0)
    }

    /// Nearest-rank percentile over an ascending sample. Twin of
    /// `AthenaMetrics.percentile` (kept local so AthenaCore takes no dep on
    /// AthenaServerKit). ponytail: 4-line dup, fold if a third copy appears.
    static func percentile(_ sorted: [Double], _ p: Double) -> Double {
        guard !sorted.isEmpty else { return 0 }
        let rank = Int((Double(sorted.count) * p).rounded(.up))
        return sorted[min(sorted.count - 1, max(0, rank - 1))]
    }

    /// ADR 038 — render gate stats as Prometheus text 0.0.4 (appended to
    /// `/metrics`). Pure, so it is unit-testable without the actor.
    public static func prometheus(_ s: GateStats) -> String {
        var out = ""
        out += "# HELP athena_inference_gate_waiters Requests queued behind "
        out += "the inference execution gate (current depth).\n"
        out += "# TYPE athena_inference_gate_waiters gauge\n"
        out += "athena_inference_gate_waiters \(s.waiters)\n"
        out += "# HELP athena_inference_gate_held Whether the gate is held "
        out += "(1) or free (0).\n"
        out += "# TYPE athena_inference_gate_held gauge\n"
        out += "athena_inference_gate_held \(s.held ? 1 : 0)\n"
        out += "# HELP athena_inference_gate_held_ms How long the current "
        out += "holder has held the gate (ms; 0 when free).\n"
        out += "# TYPE athena_inference_gate_held_ms gauge\n"
        out += "athena_inference_gate_held_ms \(s.heldMs)\n"
        out += "# HELP athena_inference_gate_max_waiters High-water queue "
        out += "depth since boot.\n"
        out += "# TYPE athena_inference_gate_max_waiters gauge\n"
        out += "athena_inference_gate_max_waiters \(s.maxWaiters)\n"
        out += "# HELP athena_inference_gate_acquisitions_total Successful "
        out += "gate acquisitions (includes uncontended).\n"
        out += "# TYPE athena_inference_gate_acquisitions_total counter\n"
        out += "athena_inference_gate_acquisitions_total \(s.acquisitions)\n"
        out += "# HELP athena_inference_gate_contended_total Acquisitions "
        out += "that had to queue (the contention signal).\n"
        out += "# TYPE athena_inference_gate_contended_total counter\n"
        out += "athena_inference_gate_contended_total \(s.contended)\n"
        out += "# HELP athena_inference_gate_wait_ms Queue wait time (ms) "
        out += "over a recent bounded window.\n"
        out += "# TYPE athena_inference_gate_wait_ms summary\n"
        out += "athena_inference_gate_wait_ms{quantile=\"0.5\"} \(s.waitP50Ms)\n"
        out += "athena_inference_gate_wait_ms{quantile=\"0.95\"} \(s.waitP95Ms)\n"
        out += "athena_inference_gate_wait_ms_max \(s.waitMaxMs)\n"
        return out
    }

    /// Test-only: number of queued waiters (not the holder).
    var waiterCount: Int { waiters.count }
    /// Test-only: whether the gate is currently held.
    var isHeld: Bool { held }
}
