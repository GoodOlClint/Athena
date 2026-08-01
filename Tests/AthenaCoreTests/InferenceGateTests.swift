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

        /// Where a failure arm reports. Defaults to `XCTFail`; issue #23
        /// substitutes a collector so the arms themselves can be tested.
        ///
        /// `XCTExpectFailure` cannot express what these tests need: they must
        /// assert the diagnostic *count* is exactly one (a second means the
        /// bail arm itself leaked) AND match its text AND keep asserting
        /// afterwards. Its scoped form is also synchronous, so it cannot
        /// `await` the construction under test. Injecting the sink turns each
        /// arm into an ordinary assertion.
        typealias FailureSink = @Sendable (String, StaticString, UInt) -> Void

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
            failureSink: FailureSink? = nil,
            file: StaticString = #filePath, line: UInt = #line
        ) async {
            let fail = failureSink ?? { XCTFail($0, file: $1, line: $2) }
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
                fail(
                    outcome == .timedOut
                        ? "HeldGate acquisition did not complete within "
                            + "\(seconds)s — gate wedged or held elsewhere"
                        : "HeldGate never acquired the gate",
                    file, line)
                // Unwind both ways it could still be parked: queued inside
                // `acquire()` (cancel drains it via `cancelWaiter`), or already
                // in the body awaiting a release that now never comes.
                task.cancel()
                releaseC.finish()
                // Join the unwind: the initializer must not return with its
                // own task mid-cancel. Bounded by a DEADLINE, not by argument
                // (#23) — the reasoning "cancelWaiter dequeues, so the park
                // cannot outlive this await" is only true while the
                // `task.cancel()` above is present. Delete that line and an
                // unbounded `await task.value` hangs to the CI job timeout
                // with no diagnostic; verified by mutation. The deadline turns
                // that regression into a named failure.
                await joinUnwind(fail, file, line)
                return nil
            }
            guard await gate.isHeld else {
                // Acquired, but the gate is not held — a disabled gate runs the
                // body ungated. Same cascade risk, so bail the same way.
                fail(
                    "HeldGate returned without holding the gate",
                    file, line)
                releaseC.finish()
                await joinUnwind(fail, file, line)
                return nil
            }
        }

        /// Await the holder task's unwind, bounded. A bail arm that leaves
        /// the task parked is a real defect in that arm — report it rather
        /// than suspending the whole suite waiting for something that never
        /// comes.
        ///
        /// Polls a flag set by a DETACHED observer rather than awaiting
        /// `task.value` inside a task group. A group must drain its children,
        /// and cancelling a child that is awaiting `task.value` does not
        /// interrupt that await (the awaited task is not itself cancelled), so
        /// the group-race version is exactly as unbounded as the plain await
        /// it replaced — verified by mutation. The observer here is abandoned,
        /// not joined, so this returns on the deadline no matter what.
        private func joinUnwind(
            _ fail: FailureSink, _ file: StaticString, _ line: UInt,
            seconds: Double = 10
        ) async {
            let done = Flag()
            // Deliberately not cancelled on exit: cancelling a task that is
            // awaiting `task.value` does not interrupt that await (the awaited
            // task is not itself cancelled), which is the same reason the
            // task-group version below did not work. A cancel here would be
            // theatre. The observer is abandoned; it retains only `done` and
            // `task`, never `self`, so it cannot keep the holder alive.
            _ = Task { [task] in
                _ = try? await task.value
                done.set()
            }
            let deadline = ContinuousClock.now + .seconds(seconds)
            while !done.isSet {
                if ContinuousClock.now > deadline {
                    fail(
                        "HeldGate bail did not unwind within \(seconds)s — "
                            + "the holder task is still parked, so this "
                            + "failure arm is missing a cancel or a release",
                        file, line)
                    return
                }
                try? await Task.sleep(for: .milliseconds(1))
            }
        }

        /// Minimal thread-safe boolean — the observer sets it off-actor.
        private final class Flag: @unchecked Sendable {
            private let lock = NSLock()
            private var value = false
            func set() {
                lock.lock()
                defer { lock.unlock() }
                value = true
            }
            var isSet: Bool {
                lock.lock()
                defer { lock.unlock() }
                return value
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

    /// Issue #20 — a span that SUCCEEDS while a Metal fault is latched must
    /// release exactly once.
    ///
    /// `try throwIfMetalFaulted()` used to sit inside the `do`, so its throw on
    /// the success path was caught by the attached `catch`, which released
    /// again. The final state is identical either way (both releases end with
    /// the queue drained), so this asserts the only thing that differs: with two
    /// waiters queued, a double release resumes BOTH, and their gated sections
    /// overlap — the ADR 029 violation the gate exists to prevent.
    ///
    /// Reachable on the designed path, not an exotic one: an async allocation
    /// fault fired on MLX's worker thread while `work()` returned normally is
    /// exactly what ADR 030 Part 2 added the latch for. It was masked because
    /// the one test that latches a fault has no queued waiters.
    func testLatchedFaultOnSuccessReleasesOnce() async throws {
        let gate = InferenceGate()
        let track = Concurrency()
        let (park, parkC) = AsyncStream.makeStream(of: Void.self)
        let (entered, enteredC) = AsyncStream.makeStream(of: Void.self)

        // Holder: parks until the waiters are queued, then succeeds having
        // latched a fault — the double-release trigger.
        let holder = Task {
            try await gate.withExclusiveExecution { () -> Int in
                enteredC.yield()
                var it = park.makeAsyncIterator()
                _ = await it.next()
                MetalFaultLatch.shared.record(
                    "[metal::malloc] Attempting to allocate 111 GB")
                return 0
            }
        }
        var enteredIt = entered.makeAsyncIterator()
        guard await enteredIt.next() != nil else {
            return XCTFail("holder never acquired the gate")
        }

        let waiters = (0 ..< 2).map { _ in
            Task {
                try await gate.withExclusiveExecution {
                    await track.enter()
                    try? await Task.sleep(for: .milliseconds(20))
                    await track.leave()
                }
            }
        }
        guard await waitUntil("both waiters queued", { await gate.waiterCount == 2 })
        else {
            parkC.finish()
            for w in waiters { w.cancel() }
            return
        }

        parkC.finish()
        // The holder surfaces the latched fault as a classified 503.
        do {
            _ = try await holder.value
            XCTFail("a latched fault must surface even on the success path")
        } catch let e as AthenaError {
            XCTAssertEqual(e.code, "metal_oom")
        }
        for w in waiters { _ = try await w.value }

        let overlapped = await track.maxActive
        XCTAssertEqual(
            overlapped, 1,
            "double release handed the gate to both waiters — two Metal "
                + "tenants executing concurrently (ADR 029)")
        let held = await gate.isHeld
        let queued = await gate.waiterCount
        XCTAssertFalse(held)
        XCTAssertEqual(queued, 0)
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

    // MARK: - HeldGate failure arms (issue #23)

    /// Thread-safe collector standing in for `XCTFail` so a failure arm can be
    /// asserted on instead of failing the test that exercises it.
    private final class FailureCollector: @unchecked Sendable {
        private let lock = NSLock()
        private var messages: [String] = []
        func record(_ m: String) {
            lock.lock()
            defer { lock.unlock() }
            messages.append(m)
        }
        var recorded: [String] {
            lock.lock()
            defer { lock.unlock() }
            return messages
        }
    }

    /// Arm 1 — the deadline arm (`.timedOut`). A `HeldGate` built on a gate
    /// someone else already holds can never enter its body, so the handshake
    /// must expire, return nil, name the wedged-or-held-elsewhere cause, and
    /// leave no waiter behind.
    ///
    /// The abandoned acquisition is the point: it is still queued inside
    /// `acquire()` when the deadline fires, so only `task.cancel()` →
    /// `cancelWaiter` dequeues it. Drop that cancel and `waiterCount` stays 1.
    ///
    /// CEILING (issue #23 asks for this to be stated when a hard bound is
    /// impractical, and it is — SwiftPM exposes no per-test
    /// `executionTimeAllowance`). The `seconds: 0.5` bound is delivered BY the
    /// deadline race inside `HeldGate.init`. It bounds a regression in the
    /// arm's *unwind* (verified: dropping `task.cancel()` fails in ~15 s), but
    /// NOT a regression in the race that produces `outcome` — break that and
    /// this test hangs to the 60-minute CI job timeout with no diagnostic.
    /// Verified by mutation, not assumed. Pinning the race itself would need
    /// the same construct under test to bound it, which is circular.
    func testHeldGateDeadlineArmFailsAndDrains() async throws {
        let gate = InferenceGate()
        guard let holder = await HeldGate(gate) else { return }
        defer { holder.release() }

        let sink = FailureCollector()
        let secondAttempt = await HeldGate(
            gate, seconds: 0.5, failureSink: { m, _, _ in sink.record(m) })

        XCTAssertNil(
            secondAttempt,
            "a HeldGate that never entered its body must not hand back a "
                + "holder — the call site would rethrow CancellationError "
                + "from holder.task.value and bury this diagnostic")
        XCTAssertEqual(
            sink.recorded.count, 1,
            "one diagnostic, naming one cause. A second means the bail arm "
                + "itself leaked — joinUnwind reports when the holder task is "
                + "still parked. Got: \(sink.recorded)")
        let msg = try XCTUnwrap(sink.recorded.first)
        XCTAssertTrue(
            msg.contains("did not complete within"),
            "the deadline arm must name the timeout, not the "
                + "never-acquired cause — they point at different bugs. "
                + "Got: \(msg)")
        await waitUntil("abandoned acquisition drains") {
            await gate.waiterCount == 0
        }
        let stillHeld = await gate.isHeld
        XCTAssertTrue(
            stillHeld,
            "the original holder still holds — the failed attempt must not "
                + "have released someone else's gate")
    }

    /// Arm 2 — the acquired-but-not-held arm, added by the #25 fold-in. A
    /// disabled gate runs the body ungated, so the handshake succeeds while
    /// `isHeld` is false.
    ///
    /// Deliberately asymmetric with arm 1 and NOT covered by it: the body has
    /// provably entered here, so there is nothing queued to cancel.
    ///
    /// Honest about what this pins. `deinit { releaseC.finish() }` also
    /// unwinds the task — a failable class initializer that returns nil after
    /// full initialization DOES run `deinit` (verified). So the arm's own
    /// `releaseC.finish()` is not required for correctness; what it buys is
    /// PROMPTNESS, because `joinUnwind` runs before `return nil` and therefore
    /// before `deinit`. Drop it and the bail still unwinds, just 10 s later
    /// and with a spurious second diagnostic. What this test genuinely pins is
    /// the arm's contract — returns nil, names the not-held cause, exactly one
    /// diagnostic — not a statement whose deletion would corrupt the gate.
    ///
    /// Contrast arm 1, where `task.cancel()` IS irreplaceable: `deinit` cannot
    /// dequeue a waiter parked inside `acquire()`.
    func testHeldGateNotHeldArmFailsAndDrains() async throws {
        InferenceGate.enabled = false
        defer { InferenceGate.enabled = true }
        let gate = InferenceGate()

        let sink = FailureCollector()
        let attempt = await HeldGate(
            gate, seconds: 5, failureSink: { m, _, _ in sink.record(m) })

        XCTAssertNil(
            attempt, "a holder that does not hold must not be handed back")
        XCTAssertEqual(
            sink.recorded.count, 1,
            "one diagnostic, naming one cause. A second is joinUnwind "
                + "reporting that the bail left the holder task parked past "
                + "its deadline. Got: \(sink.recorded)")
        let msg = try XCTUnwrap(sink.recorded.first)
        XCTAssertTrue(
            msg.contains("without holding the gate"),
            "this arm must name the not-held cause, not the deadline. "
                + "Got: \(msg)")
        // Vacuous by construction, and kept only as a premise check: a
        // disabled gate returns `try await work()` without calling `acquire()`
        // (InferenceGate.swift), so nothing is ever enqueued and this cannot
        // fail. The operator's "drains waiterCount to 0" criterion is pinned
        // for real by the deadline-arm test above, which has an actual
        // abandoned acquisition to drain. Said plainly so this is not read as
        // parity between the two arms.
        let waiters = await gate.waiterCount
        XCTAssertEqual(
            waiters, 0, "premise: a disabled gate enqueues nothing")
        let held = await gate.isHeld
        XCTAssertFalse(held)
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
