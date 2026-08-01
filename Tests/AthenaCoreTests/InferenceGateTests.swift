import XCTest

@testable import AthenaCore

/// ADR 029 — the inference execution gate is MLX-free, so its serialization /
/// FIFO / cancellation / revert-knob invariants run in CI (ADR 009).
final class InferenceGateTests: XCTestCase {

    override func setUp() {
        InferenceGate.enabled = true
        MetalFaultLatch.shared.clear()
    }
    override func tearDown() {
        InferenceGate.enabled = true
        MetalFaultLatch.shared.clear()
    }

    private actor Concurrency {
        private(set) var maxActive = 0
        private var active = 0
        func enter() { active += 1; maxActive = max(maxActive, active) }
        func leave() { active -= 1 }
    }

    /// Deterministic handshake for ordering-sensitive gate tests (issue #3):
    /// fixed sleep windows flaked on slow shared runners when the assumed
    /// scheduling didn't happen in time. `HeldGate` holds the gate until
    /// released and signals when its body has provably entered; the shared
    /// `waitUntil` (AsyncTestWait.swift) polls an observable condition with a
    /// generous monotonic deadline instead of betting on a fixed window.
    private final class HeldGate {
        let task: Task<Void, Error>
        private let releaseC: AsyncStream<Void>.Continuation
        /// How the `entered` handshake finished. `timedOut` is distinct from
        /// `neverAcquired` on purpose: the first means the acquisition is still
        /// outstanding (a wedged or held-elsewhere gate), the second means the
        /// span exited without ever entering the body (acquire threw). They
        /// point at different bugs.
        private enum Handshake { case entered, neverAcquired, timedOut }

        /// Acquire `gate` and suspend inside the body until `release()`.
        /// Returns only after the body has provably entered (gate held) —
        /// verified by assertion, since a disabled gate runs bodies ungated.
        ///
        /// The handshake is bounded. Awaiting it outright is only safe when the
        /// gate is fresh and unheld, so `acquire()` takes the uncontended fast
        /// path — that is a property of the current call sites, not of this
        /// helper. On a contended gate an acquisition that never completes would
        /// suspend here until the CI job timeout, reporting nothing. A deadline
        /// turns that into one named failure.
        ///
        /// Fails (returns nil) rather than handing back a holder that does not
        /// hold: a failed acquisition leaves `task` cancelled, so a call site
        /// that carried on would rethrow `CancellationError` from its later
        /// `holder.task.value` and bury the named diagnostic under secondary
        /// noise pointing at the wrong thing. Call sites `guard let`.
        init?(
            _ gate: InferenceGate, seconds: Double = 10,
            file: StaticString = #filePath, line: UInt = #line
        ) async {
            let (entered, enteredC) = AsyncStream.makeStream(of: Void.self)
            let (release, releaseC) = AsyncStream.makeStream(of: Void.self)
            self.releaseC = releaseC
            self.task = Task {
                // Terminate `entered` on EVERY exit (incl. acquire-throws), so
                // a failed acquisition surfaces below immediately instead of
                // waiting out the whole deadline.
                defer { enteredC.finish() }
                try await gate.withExclusiveExecution {
                    enteredC.yield()
                    var it = release.makeAsyncIterator()
                    _ = await it.next()
                }
            }
            let outcome = await withTaskGroup(of: Handshake.self) { group in
                group.addTask {
                    var it = entered.makeAsyncIterator()
                    return await it.next() == nil ? .neverAcquired : .entered
                }
                group.addTask {
                    try? await Task.sleep(for: .seconds(seconds))
                    return .timedOut
                }
                let first = await group.next() ?? .timedOut
                group.cancelAll()
                return first
            }
            guard outcome == .entered else {
                XCTFail(
                    outcome == .timedOut
                        ? "HeldGate acquisition did not complete within "
                            + "\(seconds)s — gate wedged or held elsewhere"
                        : "HeldGate never acquired the gate",
                    file: file, line: line)
                // Unwind both ways it could still be parked: queued inside
                // `acquire()` (cancel drains it via `cancelWaiter`), or already
                // in the body awaiting a release that now never comes.
                task.cancel()
                releaseC.finish()
                // Join the unwind, like the waitUntil bail branches below: the
                // initializer must not return with its own task mid-cancel.
                // Bounded — `cancelWaiter` dequeues and throws regardless of
                // `held`, and `releaseC.finish()` unblocks a body already past
                // acquisition, so neither park outlives this await.
                _ = try? await task.value
                return nil
            }
            guard await gate.isHeld else {
                // Acquired, but the gate is not held — a disabled gate runs the
                // body ungated. Same cascade risk, so bail the same way.
                XCTFail(
                    "HeldGate returned without holding the gate",
                    file: file, line: line)
                releaseC.finish()
                _ = try? await task.value
                return nil
            }
        }
        func release() { releaseC.finish() }
        // A skipped release() must not park the holder task forever.
        deinit { releaseC.finish() }
    }

    /// Mutual exclusion: gated sections never overlap, even under contention,
    /// and the gate drains to empty afterward.
    func testSerializesConcurrentSections() async throws {
        let gate = InferenceGate()
        let track = Concurrency()
        await withThrowingTaskGroup(of: Void.self) { group in
            for _ in 0 ..< 20 {
                group.addTask {
                    try await gate.withExclusiveExecution {
                        await track.enter()
                        try? await Task.sleep(nanoseconds: 500_000)
                        await track.leave()
                    }
                }
            }
            try? await group.waitForAll()
        }
        let mx = await track.maxActive
        XCTAssertEqual(mx, 1, "gated sections overlapped")
        let held = await gate.isHeld
        let waiters = await gate.waiterCount
        XCTAssertFalse(held, "gate left held")
        XCTAssertEqual(waiters, 0, "gate left with queued waiters")
    }

    /// The revert knob: a disabled gate runs work directly and does NOT
    /// serialize (sections overlap).
    func testDisabledDoesNotSerialize() async throws {
        InferenceGate.enabled = false
        let gate = InferenceGate()
        let r = try await gate.withExclusiveExecution { 42 }
        XCTAssertEqual(r, 42)

        let track = Concurrency()
        await withTaskGroup(of: Void.self) { group in
            for _ in 0 ..< 6 {
                group.addTask {
                    try? await gate.withExclusiveExecution {
                        await track.enter()
                        try? await Task.sleep(nanoseconds: 3_000_000)
                        await track.leave()
                    }
                }
            }
        }
        let mx = await track.maxActive
        XCTAssertGreaterThan(mx, 1, "disabled gate should not serialize")
    }

    /// Value and error both propagate; a throwing section still releases the
    /// gate (the next acquire succeeds).
    func testErrorPropagatesAndReleases() async throws {
        struct Boom: Error {}
        let gate = InferenceGate()
        do {
            _ = try await gate.withExclusiveExecution { throw Boom() }
            XCTFail("error did not propagate")
        } catch is Boom {
        } catch { XCTFail("wrong error: \(error)") }
        // If the throw leaked the gate this would hang; it returns ⇒ released.
        let r = try await gate.withExclusiveExecution { 7 }
        XCTAssertEqual(r, 7)
        let held = await gate.isHeld
        XCTAssertFalse(held)
    }

    /// A queued waiter whose task is cancelled leaves the queue with a
    /// `CancellationError` and never blocks the holder or the gate.
    func testCancelledWaiterThrowsAndDrains() async throws {
        let gate = InferenceGate()
        // nil ⇒ acquisition already failed with its own named diagnostic;
        // carrying on would bury it under a secondary CancellationError.
        guard let holder = await HeldGate(gate) else { return }
        defer { holder.release() }
        let waiter = Task {
            try await gate.withExclusiveExecution { 1 }
        }
        guard await waitUntil("waiter enqueued", { await gate.waiterCount == 1 })
        else {
            // Don't leave the waiter running past this method: cancel it and
            // await the unwind, so the failure report isn't racing a live task.
            waiter.cancel()
            _ = try? await waiter.value
            return
        }
        waiter.cancel()
        do {
            _ = try await waiter.value
            XCTFail("cancelled waiter should throw")
        } catch is CancellationError {
        } catch { XCTFail("expected CancellationError, got \(error)") }
        holder.release()
        _ = try await holder.task.value
        let held = await gate.isHeld
        let waiters = await gate.waiterCount
        XCTAssertFalse(held)
        XCTAssertEqual(waiters, 0)
    }

    /// The revert knob flipping mid-span must not strand the gate. `release()`
    /// used to open with `guard Self.enabled`, so a span acquired while enabled
    /// released into a no-op once the knob went false — `held` stayed true with
    /// no holder and every later acquire queued forever. Reachable from this
    /// tier today: `MemoryGovernor.evictSync` spawns detached teardown tasks
    /// that outlive their test method, and `testDisabledDoesNotSerialize` sets
    /// `enabled = false` on its first line. Only alphabetical class ordering
    /// hid it.
    func testReleaseSurvivesDisabledFlipMidSpan() async throws {
        let gate = InferenceGate()
        guard let holder = await HeldGate(gate) else { return }  // while enabled
        defer { holder.release() }
        InferenceGate.enabled = false  // …flips while the span is in flight
        holder.release()
        _ = try await holder.task.value

        let held = await gate.isHeld
        let waiters = await gate.waiterCount
        XCTAssertFalse(held, "release() stranded the gate held after the flip")
        XCTAssertEqual(waiters, 0)

        // And the gate is genuinely reusable, not merely reported free. Bounded
        // so a regression fails with a diagnostic instead of hanging CI — the
        // stranded gate's symptom is an acquire that never returns.
        InferenceGate.enabled = true
        let reuse = Task { try await gate.withExclusiveExecution { 7 } }
        guard
            await waitUntil(
                "gate reacquired after the mid-span flip",
                { await gate.stats().acquisitions == 2 })
        else {
            reuse.cancel()
            _ = try? await reuse.value
            return
        }
        let r = try await reuse.value
        XCTAssertEqual(r, 7)
    }

    // MARK: - ADR 038 slice 1 — gate observability accounting

    /// Uncontended acquisitions: every acquire takes the fast path, so
    /// `contended`/`maxWaiters`/wait-p95 all stay 0 (⇒ "no contention" on
    /// /metrics), while `acquisitions` counts them all.
    func testUncontendedAccounting() async throws {
        let gate = InferenceGate()
        for _ in 0 ..< 5 { try await gate.withExclusiveExecution {} }
        let s = await gate.stats()
        XCTAssertEqual(s.acquisitions, 5)
        XCTAssertEqual(s.contended, 0)
        XCTAssertEqual(s.maxWaiters, 0)
        XCTAssertEqual(s.waiters, 0)
        XCTAssertFalse(s.held)
        XCTAssertEqual(s.waitP95Ms, 0)
    }

    /// Contention: three requests queue behind a holder → depth + high-water +
    /// contended count are visible mid-flight and after, and the holder itself
    /// is not counted as contended.
    func testContentionAccounting() async throws {
        let gate = InferenceGate()
        // nil ⇒ acquisition already failed with its own named diagnostic;
        // carrying on would bury it under a secondary CancellationError.
        guard let holder = await HeldGate(gate) else { return }
        defer { holder.release() }
        let waiters = (0 ..< 3).map { _ in
            Task { try await gate.withExclusiveExecution {} }
        }
        guard
            await waitUntil(
                "all three waiters enqueued", { await gate.waiterCount == 3 })
        else {
            for w in waiters { w.cancel() }
            for w in waiters { _ = try? await w.value }
            return
        }
        let mid = await gate.stats()
        XCTAssertTrue(mid.held)
        XCTAssertEqual(mid.waiters, 3, "expected 3 queued behind the holder")
        XCTAssertEqual(mid.maxWaiters, 3)

        holder.release()
        _ = try await holder.task.value
        for w in waiters { _ = try await w.value }
        let done = await gate.stats()
        XCTAssertEqual(done.acquisitions, 4, "holder + 3 waiters")
        XCTAssertEqual(done.contended, 3, "only the 3 that queued")
        XCTAssertEqual(done.maxWaiters, 3)
        XCTAssertEqual(done.waiters, 0)
        XCTAssertFalse(done.held)
    }

    /// The revert knob observes nothing: a disabled gate serializes nothing, so
    /// stats stay all-zero/unheld.
    func testDisabledGateObservesNothing() async throws {
        InferenceGate.enabled = false
        let gate = InferenceGate()
        _ = try await gate.withExclusiveExecution { 1 }
        let s = await gate.stats()
        XCTAssertFalse(s.held)
        XCTAssertEqual(s.acquisitions, 0)
        XCTAssertEqual(s.waiters, 0)
    }

    /// The Prometheus render is pure: names, TYPE lines, and values are exact.
    func testGatePrometheusRender() {
        let s = InferenceGate.GateStats(
            held: true, waiters: 2, heldMs: 12.5, maxWaiters: 3,
            acquisitions: 10, contended: 4,
            waitP50Ms: 1.0, waitP95Ms: 9.0, waitMaxMs: 15.0)
        let out = InferenceGate.prometheus(s)
        XCTAssertTrue(out.contains("athena_inference_gate_waiters 2"))
        XCTAssertTrue(out.contains("athena_inference_gate_held 1"))
        XCTAssertTrue(out.contains("athena_inference_gate_max_waiters 3"))
        XCTAssertTrue(out.contains("athena_inference_gate_acquisitions_total 10"))
        XCTAssertTrue(out.contains("athena_inference_gate_contended_total 4"))
        XCTAssertTrue(
            out.contains("athena_inference_gate_wait_ms{quantile=\"0.95\"} 9.0"))
        XCTAssertTrue(
            out.contains(
                "# TYPE athena_inference_gate_acquisitions_total counter"))
    }

    // MARK: - ADR 030 Part 2 (WP2) — degrade latched Metal faults

    /// A recognized allocation fault recorded during a gated span (as the global
    /// error handler does, off the worker thread) is converted to a classified
    /// `metalOutOfMemory` (503) on span exit — the daemon-abort path is gone.
    func testLatchedMetalFaultDegradesTo503WP2() async throws {
        let gate = InferenceGate()
        do {
            _ = try await gate.withExclusiveExecution {
                // Simulate the worker-thread handler recording a device-cap OOM.
                MetalFaultLatch.shared.record(
                    "[metal::malloc] Attempting to allocate 111 GB which is "
                        + "greater than the maximum allowed buffer size")
                return 0
            }
            XCTFail("a latched Metal fault must surface as an error")
        } catch let e as AthenaError {
            XCTAssertEqual(e.httpStatus, 503)
            XCTAssertEqual(e.code, "metal_oom")
        }
        // Latch consumed: a subsequent clean span succeeds (no stale fault).
        let r = try await gate.withExclusiveExecution { 5 }
        XCTAssertEqual(r, 5, "latch must not leak into the next span")
        let held = await gate.isHeld
        XCTAssertFalse(held, "gate left held after a degraded fault")
    }

    /// The recorded fault wins even when the span ALSO throws (the raw throw is
    /// usually a cascade artifact of touching the invalid arrays).
    func testLatchedFaultPreferredOverRawThrowWP2() async throws {
        struct Boom: Error {}
        let gate = InferenceGate()
        do {
            _ = try await gate.withExclusiveExecution {
                MetalFaultLatch.shared.record("metal::malloc device cap")
                throw Boom()
            }
            XCTFail("expected the classified Metal OOM")
        } catch let e as AthenaError {
            XCTAssertEqual(e.code, "metal_oom")
        } catch { XCTFail("raw throw leaked past the latch: \(error)") }
    }

    /// No fault ⇒ no throw: the happy path is byte-unchanged.
    func testNoFaultNoDegradeWP2() async throws {
        let gate = InferenceGate()
        let r = try await gate.withExclusiveExecution { 99 }
        XCTAssertEqual(r, 99)
    }

    // MARK: - MetalFaultLatch algebra

    func testMetalFaultLatchRecordTakeClear() {
        let latch = MetalFaultLatch()
        XCTAssertFalse(latch.isSet)
        latch.record("first")
        latch.record("second")  // first-writer-wins: keep the root cause
        XCTAssertTrue(latch.isSet)
        XCTAssertEqual(latch.take(), "first")
        XCTAssertNil(latch.take(), "take clears the slot")
        XCTAssertFalse(latch.isSet)
        latch.record("again")
        latch.clear()
        XCTAssertNil(latch.take())
    }
}
