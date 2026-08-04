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
        ///
        /// Both arms are driven (#69): `.timedOut` by
        /// `testHeldGateDeadlineArmFailsAndDrains`, `.neverAcquired` by
        /// `testHeldGateNeverAcquiredArmFailsAndDrains` via the `failAcquire`
        /// injection below.
        ///
        /// That took correcting a wrong premise, recorded so it is not
        /// re-adopted. `acquire()` throws in only two places, both needing the
        /// HOLDER task cancelled: `Task.checkCancellation()` on entry, or
        /// `cancelWaiter` resuming a queued continuation. Tests DO reach the
        /// second — arm 1's own `task.cancel()` drives it. What no test can do
        /// is make `acquire()` throw BEFORE the handshake resolves, which is
        /// what selecting `.neverAcquired` that way would require, since the
        /// init owns the holder task and never exposes it beforehand.
        ///
        /// That narrower fact is still a different proposition from "no test
        /// can drive this arm", which is what the gap note used to claim.
        /// `.neverAcquired` is selected by the `entered` stream finishing
        /// WITHOUT a yield, and ANY early exit of the holder body produces
        /// that. Injecting one is deterministic (40/40), 0.003 s, and leaves
        /// `Sources/` untouched.
        ///
        /// FIXED in #81 — `.callerCancelled` is the third outcome, and it
        /// exists because the other two were both wrong under one condition.
        ///
        /// Under CALLER cancellation both task-group children can finish at
        /// once: the `entered` observer is cancelled, and if its stream ends
        /// WITHOUT a yield that selects `.neverAcquired`, while the sleeping
        /// sibling's `try?` swallows `CancellationError` and returns
        /// `.timedOut`. So `group.next()` was a race between two wrong answers
        /// about a gate that had been acquired normally. Measured over 200
        /// iterations before the fix: `.neverAcquired` 16 (8%) and `.timedOut`
        /// 28 (14%), every one of the 44 with `acquisitions == 1`. (The
        /// remaining ~78% resolved `.entered` — the observer usually drains the
        /// buffered yield before noticing cancellation — which is why the
        /// defect presented as an intermittent flake.)
        ///
        /// Fixing one arm would have left the other half of the misreports
        /// intact — and `.timedOut` was the WORSE half, since "gate wedged or
        /// held elsewhere" sends a reader hunting a deadlock that does not
        /// exist. So the fix is in the HANDSHAKE, not an arm: cancellation is
        /// detected once, after the race resolves but BEFORE either result is
        /// interpreted. Both original arms keep their documented meanings
        /// exactly, and neither can now be reported about a cancelled caller.
        ///
        /// Two consequences, stated rather than glossed. Cancellation takes
        /// PRECEDENCE: a genuine `.neverAcquired` or `.timedOut` occurring in a
        /// cancelled caller is reported as `.callerCancelled`, so those arms'
        /// reachability is narrowed even though their meanings are unchanged.
        /// And the discarded-holder case is the DOMINANT one, not an edge —
        /// ~80% of cancelled handshakes had already observed the yield and
        /// would previously have returned a working holder. Both are the
        /// intended trade: a cancelled caller is being torn down anyway, and
        /// removing the timing dependence is the point.
        ///
        /// Driven by `testHeldGateCallerCancellationIsNotBlamedOnTheGate`,
        /// which forces CALLER CANCELLATION; the child race it exposes is
        /// sampled, not forced — see that test's comment for the measurement
        /// (`.entered` 81 / `.neverAcquired` 9 / `.timedOut` 10 per 100, so the
        /// misreport condition appears roughly 1 in 5). The determinism belongs
        /// to the fix, which collapses all three to one outcome, not to the
        /// construction.
        private enum Handshake {
            case entered, neverAcquired, timedOut, callerCancelled
        }

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
            failureSink: FailureSink? = nil, failAcquire: Bool = false,
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
                // #69 — drive the `.neverAcquired` arm. What selects that arm
                // is the `entered` stream finishing WITHOUT a yield, which any
                // early exit of this body produces; it does not have to be
                // `acquire()` itself throwing. So the arm is reachable from
                // the test helper, with `Sources/` untouched — same shape as
                // the `failureSink` injection #23 added, and the same
                // justification (ADR 009: the arm is decision logic, and an
                // untested failure arm is not a failure arm).
                if failAcquire { throw CancellationError() }
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
                // #81 — decide cancellation ONCE, here, before either child's
                // result is interpreted. Cancelling the caller cancels both
                // children simultaneously, so `first` is a coin flip between
                // two arms that would each assert a cause about the GATE that
                // nothing has established. This closure runs in the caller's
                // context, so `Task.isCancelled` is exactly that condition.
                return Task.isCancelled ? .callerCancelled : first
            }
            guard outcome == .entered else {
                let cause: String
                switch outcome {
                case .timedOut:
                    cause =
                        "HeldGate acquisition did not complete within "
                        + "\(seconds)s — gate wedged or held elsewhere"
                case .neverAcquired:
                    cause = "HeldGate never acquired the gate"
                case .callerCancelled:
                    // Deliberately says nothing about the gate: the gate may
                    // well have been acquired healthily (#81 measured
                    // acquisitions == 1 on every misreport).
                    //
                    // Still REPORTS rather than returning nil quietly. A
                    // cancelled caller is not itself a defect, but at the call
                    // sites that EXPECT a holder — the
                    // `guard let ... else { return }` ones — a silent nil would
                    // skip the rest of that test and pass vacuously. The
                    // remaining sites bind the result and assert nil
                    // deliberately, so they are unaffected either way. Stated
                    // as two classes rather than a count of each, because a
                    // count in prose goes silently false the day someone adds
                    // a ninth call site.
                    cause =
                        "HeldGate handshake was abandoned because the CALLER "
                        + "was cancelled — this reports nothing about the "
                        + "gate, which may have been acquired normally"
                case .entered:
                    preconditionFailure("guarded above")
                }
                fail(cause, file, line)
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
                await joinUnwind(fail, file, line, seconds: Self.unwindBudget(seconds))
                return nil
            }
            guard await gate.isHeld else {
                // Acquired, but the gate is not held — a disabled gate runs the
                // body ungated. Same cascade risk, so bail the same way.
                fail(
                    "HeldGate returned without holding the gate",
                    file, line)
                releaseC.finish()
                await joinUnwind(fail, file, line, seconds: Self.unwindBudget(seconds))
                return nil
            }
        }

        /// Unwind budget DERIVED from the init's handshake bound (#69), not a
        /// fixed 10 s. A call site that deliberately picks a short bound to
        /// fail fast — `testHeldGateDeadlineArmFailsAndDrains` uses 0.5 s —
        /// would otherwise still pay the full 10 s on an unwind regression,
        /// partly defeating the point of choosing it.
        ///
        /// The floor is what keeps this fast rather than flaky. The span being
        /// bounded is `task.cancel()` + `releaseC.finish()` draining a handful
        /// of actor hops, plus `joinUnwind`'s poll tick (~2 ms in situ since
        /// #107 moved the pacing into an uncancellable Task; it was nominally
        /// 1 ms before, and 0 under cancellation). No direct
        /// measurement of THIS span exists; the floor is justified by analogy
        /// with a comparable one measured elsewhere in this repo — #28's
        /// actor-hop span, 2–14 ms even under 2x CPU oversubscription. 1 s is
        /// ~70x that, so a 0.5 s call site fails in ~1 s instead of ~10 s with
        /// margin to spare.
        ///
        /// Two call sites move, both intentionally: the 0.5 s site 10 → 1, and
        /// `testHeldGateNotHeldArmFailsAndDrains`'s 5 s site 10 → 5 (its
        /// unwind is `releaseC.finish()` plus a task exit on a DISABLED gate,
        /// sub-millisecond, so 5 s is four orders of margin). Every site
        /// taking the default keeps exactly the 10 s it has today — #68 raised
        /// `joinUnwind`'s own default 5 → 10 to match `waitUntil`, which has
        /// been 10 s since AsyncTestWait.swift was created.
        /// `testUnwindBudgetIsDerivedFromTheHandshake` pins this rather than
        /// leaving it as prose.
        static func unwindBudget(_ handshake: Double) -> Double {
            max(handshake, 1)
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
            let done = Locked(false)
            // Deliberately not cancelled on exit: cancelling a task that is
            // awaiting `task.value` does not interrupt that await (the awaited
            // task is not itself cancelled), which is the same reason the
            // task-group version below did not work. A cancel here would be
            // theatre. The observer is abandoned; it retains only `done` and
            // `task`, never `self`, so it cannot keep the holder alive.
            _ = Task { [task] in
                _ = try? await task.value
                done.mutate { $0 = true }
            }
            let deadline = ContinuousClock.now + .seconds(seconds)
            while !done.current {
                if ContinuousClock.now > deadline {
                    fail(
                        "HeldGate bail did not unwind within \(seconds)s — "
                            + "the holder task is still parked, so this "
                            + "failure arm is missing a cancel or a release",
                        file, line)
                    return
                }
                // Pace from a context that CANNOT be cancelled. `try? await
                // Task.sleep` looks like a 1 ms pause but is a no-op in a
                // cancelled task — sleep throws immediately and `try?`
                // swallows it — so the loop would spin at full tilt until the
                // holder unwinds or the budget expires. #106 introduced the
                // first path that reaches here from a CANCELLED caller (25×
                // per run), which is what made this reachable.
                //
                // MEASURED two ways, because the worst case and the typical
                // case are far apart and quoting only one misleads.
                //
                // Worst case — 200 polls in a cancelled task with `done` never
                // set, i.e. the loop running to its deadline:
                //   try? await Task.sleep(1ms)       0.00015 s   (no pacing)
                //   await Task { sleep(1ms) }.value  0.307 s     (~1.5 ms each)
                //
                // In situ — the 25 cancelled calls this file actually makes:
                // the OLD form spun 0–7 times per call, 47 polls total, 2–15 µs
                // per call. So on a normal runner the bug was invisible, which
                // is why it was recorded as a harmless ceiling for a while.
                // The NEW form paces once per call at ~2.0 ms.
                //
                // The reason to fix it anyway is that "invisible" depends on
                // the pool: under `LIBDISPATCH_COOPERATIVE_POOL_STRICT=1` the
                // old form fails deterministically, because a loop that never
                // yields cannot let the observer task run on a one-thread
                // pool. Not pinned by the default tier (#143).
                //
                // The fix rests on two facts this file already relies on: an
                // unstructured `Task {}` does not inherit cancellation, and
                // awaiting `task.value` is not interrupted by the awaiting
                // task's own cancellation (see `joinUnwind`'s header). Cost is
                // one Task allocation per poll, bounded by the deadline.
                await Task { try? await Task.sleep(for: .milliseconds(1)) }.value
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

    /// Collector standing in for `XCTFail` so a failure arm can be asserted on
    /// instead of failing the test that exercises it. Backed by the suite's
    /// shared `Locked` box (#70).
    private final class FailureCollector: Sendable {
        private let messages = Locked([String]())
        func record(_ m: String) { messages.mutate { $0.append(m) } }
        var recorded: [String] { messages.current }
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
    /// arm's *unwind* — dropping `task.cancel()` fails in
    /// `handshake + unwindBudget(handshake) + waitUntil`, i.e. 0.5 + 1 + 10
    /// (measured 11.5 s; it was 20.5 s before #69 derived the middle term) —
    /// but NOT a regression in the race that produces `outcome`: break that
    /// and this test hangs to the 60-minute CI job timeout with no diagnostic.
    ///
    /// The ceiling is stated as its COMPONENTS rather than a single number
    /// (#69). It previously read "~15 s", which was the arithmetic for a 5 s
    /// unwind deadline and went stale the moment #68 raised that to 10 s —
    /// and stale again when #69 derived it from the handshake. A reader would
    /// try to reproduce the one figure in this comment, fail, and reasonably
    /// doubt the "verified by mutation" claim the rest of it rests on. Naming
    /// the three spans keeps it true when any of them moves.
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

    /// Arm 4 — `.callerCancelled` (#81). Before the fix, cancelling the caller
    /// mid-handshake produced a coin flip between `.neverAcquired` ("never
    /// acquired the gate") and `.timedOut` ("gate wedged or held elsewhere"),
    /// both asserted about a gate that had been acquired normally — 8% / 14%
    /// over 200 iterations, all 44 with `acquisitions == 1`.
    ///
    /// **What is forced is CALLER CANCELLATION, which is #81's requirement —
    /// NOT the child race.** An earlier draft of this comment claimed the
    /// construction left "no race" in the handshake; measurement refuted it, so
    /// the true shape is recorded here instead. Probing `first` at the decision
    /// point over 100 iterations: `.entered` 81, `.neverAcquired` 9,
    /// `.timedOut` 10 (23/1/1 on a second 25-iteration sample). The observer
    /// child, though born cancelled, usually drains the already-buffered
    /// `enteredC.yield()` — `AsyncStream` returns pending elements before
    /// reporting termination — so it answers `.entered` most of the time. The
    /// #81 misreport condition is therefore SAMPLED at roughly 1 in 5.
    ///
    /// What IS deterministic is `Task.isCancelled == true` at the check
    /// (100/100), and that is what makes the OUTCOME deterministic: the fix
    /// collapses all three values of `first` to one. So the determinism belongs
    /// to the fix, not to this construction — which is also why the loop below
    /// is load-bearing rather than decorative.
    ///
    /// The construction rests on one Swift concurrency fact: an unstructured
    /// `Task {}` does NOT inherit cancellation from the task that created it,
    /// while `withTaskGroup` children DO. So building the `HeldGate` from an
    /// ALREADY-cancelled caller puts the two in opposite states — the holder
    /// task acquires the gate normally, the handshake children are born
    /// cancelled.
    ///
    /// Getting the caller reliably cancelled BEFORE it reaches `HeldGate` is
    /// the other half: it announces itself on-CPU, then parks on a stream
    /// nobody ever feeds, so the only thing that can resume it is its own
    /// cancellation. No sleeps, no windows. If that ever stopped holding, the
    /// `guard Task.isCancelled` below fails loudly rather than silently
    /// reverting this to a hopeful test (verified by mutation).
    ///
    /// Runs 25 iterations. Because the child race is sampled, a single pass
    /// proves little: the PARTIAL fix #81 warns against (distinguishing only
    /// `.neverAcquired`) survived iteration 1 in 8 of 9 runs, first failing at
    /// iterations 4, 2, 3, 1, 2, 3, 3, 2, 3. The loop is what turns a ~1-in-5
    /// detection into a reliable one.
    ///
    /// CEILING — RESOLVED (#107), kept because the failure it predicted was
    /// then reproduced deterministically. This is the first path to reach
    /// `joinUnwind` from a cancelled caller (25× per run, counted in situ), and
    /// `joinUnwind` used to pace on a bare `try? await Task.sleep`, which
    /// throws instantly in a cancelled task and so did not pace at all. On a
    /// normal runner that was harmless — 47 polls total across the 25 calls,
    /// 2–15 µs each — which is what "measured harmless, 0.002 s" recorded.
    ///
    /// Starve the pool and it stops being harmless, exactly in the shape this
    /// note predicted: under `LIBDISPATCH_COOPERATIVE_POOL_STRICT=1` the old
    /// pacing fails deterministically — every iteration reports a SECOND
    /// diagnostic ("bail did not unwind within 1.0s") and trips the
    /// `recorded.count == 1` assertion, rather than any arm assertion failing.
    /// The busy loop never yielded, so on a one-thread cooperative pool the
    /// observer task could not run. Reproduced by mutation, both directions.
    ///
    /// `joinUnwind` now paces from an uncancellable context (see its pacing
    /// comment), and the same strict-pool run is green. Nothing in the default
    /// tier pins this — see the note there.
    func testHeldGateCallerCancellationIsNotBlamedOnTheGate() async throws {
        // #81's premise is "the gate was acquired NORMALLY" — all 44 original
        // misreports had `acquisitions == 1`. Without pinning that, this test
        // cannot tell a cancelled-but-healthy handshake from one where the
        // acquisition genuinely failed, and would pass with `failAcquire: true`
        // (measured). Per-iteration `acquisitions == 1` is NOT deterministic —
        // the bail's `task.cancel()` can beat the holder to `acquire()`'s
        // `checkCancellation()` — so the deterministic facts are asserted every
        // iteration and the premise itself is asserted once, over the loop.
        var everAcquired = false
        for iteration in 1 ... 25 {
            let gate = InferenceGate()
            let collector = FailureCollector()
            let (onCPU, onCPUC) = AsyncStream.makeStream(of: Void.self)
            let (park, parkC) = AsyncStream.makeStream(of: Void.self)

            let caller = Task { () -> Bool in
                onCPUC.yield()
                var it = park.makeAsyncIterator()
                // Resumes only via cancellation — AsyncStream iteration is
                // cancellation-aware and nothing ever yields to `park`.
                _ = await it.next()
                // Belt-and-braces: what actually retains the continuation is
                // the closure CAPTURE, established when this closure is formed
                // — the post-await position does no work. Kept because a
                // finished stream would resume this uncancelled, but that is
                // NOT silent: the guard below fails loudly (verified by
                // forcing `parkC.finish()`, which reddens at iteration 2).
                withExtendedLifetime(parkC) {}
                guard Task.isCancelled else { return false }
                let attempt = await HeldGate(
                    gate, seconds: 1,
                    failureSink: { m, _, _ in collector.record(m) })
                return attempt == nil
            }

            var started = onCPU.makeAsyncIterator()
            _ = await started.next()  // the caller is on-CPU and about to park
            caller.cancel()
            let bailedWithoutHolder = await caller.value

            XCTAssertTrue(
                bailedWithoutHolder,
                "iteration \(iteration): the caller must have been cancelled "
                    + "before constructing HeldGate, and a HeldGate that never "
                    + "completed its handshake must not hand back a holder")
            XCTAssertEqual(
                collector.recorded.count, 1,
                "iteration \(iteration): one diagnostic, naming one cause. "
                    + "Got: \(collector.recorded)")
            let msg = try XCTUnwrap(collector.recorded.first)
            XCTAssertTrue(
                msg.contains("CALLER"),
                "iteration \(iteration): must name caller cancellation. "
                    + "Got: \(msg)")
            // The two regressions this pins, stated as what must NOT be said.
            // Fixing only `.neverAcquired` — the direction #81 was originally
            // filed for — leaves the `.timedOut` half live, and that half is
            // the worse one: it sends a reader hunting a deadlock.
            XCTAssertFalse(
                msg.contains("never acquired the gate"),
                "iteration \(iteration): the caller being cancelled says "
                    + "nothing about whether the gate was acquired. Got: \(msg)")
            XCTAssertFalse(
                msg.contains("wedged or held elsewhere"),
                "iteration \(iteration): nothing established that the gate is "
                    + "wedged — this arm is the majority half of the old "
                    + "misreport. Got: \(msg)")

            // Deterministic every iteration: whatever the holder did, the bail
            // must leave nothing behind. Nothing else in this file pins the
            // unwind on the cancelled-caller path.
            let stats = await gate.stats()
            XCTAssertFalse(
                stats.held,
                "iteration \(iteration): the bail must release the gate")
            XCTAssertEqual(
                stats.waiters, 0,
                "iteration \(iteration): no waiter left behind")
            if stats.acquisitions == 1 { everAcquired = true }
        }
        // The premise, asserted over the loop because it is not per-iteration
        // deterministic. This is what pins the unstructured-`Task {}` fact the
        // whole construction rests on: if the holder ever stopped acquiring
        // (e.g. because it started inheriting the caller's cancellation), the
        // test would still be green on every other assertion while no longer
        // exercising #81's actual condition at all.
        XCTAssertTrue(
            everAcquired,
            "no iteration observed acquisitions == 1, so none of them "
                + "reproduced #81's condition — a cancelled caller whose gate "
                + "was acquired NORMALLY. The construction depends on an "
                + "unstructured Task {} not inheriting cancellation; if that "
                + "stopped holding, every assertion above would still pass "
                + "while testing nothing.")
    }

    /// Arm 3 — the `.neverAcquired` arm. #69 first recorded this as a known
    /// gap on the premise that driving it needed `acquire()` to throw, which
    /// would mean a production change. The pre-submit review refuted that: the
    /// arm is selected by the `entered` stream finishing WITHOUT a yield, and
    /// any early exit of the holder body does that. `failAcquire` injects one.
    ///
    /// Asserts `acquisitions == 0` deliberately, and the assertion is KEPT now
    /// that #81 is fixed rather than retired with it. It was added because this
    /// arm used to also fire under caller cancellation, about a gate that HAD
    /// been acquired — so a test here could pass while pinning that misreport
    /// as correct. `.callerCancelled` now takes that case, so the assertion
    /// should never be what fails; it stays as the guard that would notice if
    /// the two conditions ever merged again.
    func testHeldGateNeverAcquiredArmFailsAndDrains() async throws {
        let gate = InferenceGate()
        let collector = FailureCollector()
        let attempt = await HeldGate(
            gate, seconds: 5,
            failureSink: { m, f, l in collector.record(m) },
            failAcquire: true)

        XCTAssertNil(attempt, "a holder that never acquired must not be handed back")
        XCTAssertEqual(
            collector.recorded.count, 1,
            "one diagnostic, naming one cause. A second means the bail arm "
                + "itself leaked. Got: \(collector.recorded)")
        XCTAssertEqual(
            collector.recorded.first, "HeldGate never acquired the gate",
            "the .neverAcquired arm must name acquisition, not the deadline")
        let stats = await gate.stats()
        XCTAssertEqual(
            stats.acquisitions, 0,
            "premise: the body never ran, so nothing was acquired. If this "
                + "fires, this arm has started reporting about a gate that "
                + "WAS acquired — the #81 misreport, fixed by the "
                + ".callerCancelled outcome — and this test would be pinning "
                + "it as correct")
        XCTAssertFalse(stats.held, "the gate must be free")
        XCTAssertEqual(stats.waiters, 0, "no waiter left behind")
    }

    /// #69 — the derived unwind budget's ARITHMETIC. Scoped deliberately, and
    /// the scope is worth stating because the obvious overclaim is tempting:
    /// this pins `unwindBudget` itself, NOT that either bail arm calls it.
    /// Reverting both `joinUnwind(…, seconds:)` call sites to the bare form
    /// leaves this test — and the whole suite — green, because on a healthy
    /// path `joinUnwind` returns the moment the holder unwinds and never
    /// consults its deadline. Only a broken unwind reaches the deadline.
    ///
    /// The WIRING is therefore verified by mutation, not by assertion:
    /// dropping `task.cancel()` from arm 1 fails in 11.5 s with the
    /// diagnostic "did not unwind within 1.0s" — the derived value, spelled
    /// out in the message — where the same mutation on `main` gave 20.5 s and
    /// "within 10.0s". Recorded here so the pair is legible together.
    func testUnwindBudgetIsDerivedFromTheHandshake() {
        XCTAssertEqual(
            HeldGate.unwindBudget(10), 10,
            "the default handshake must map to the 10 s every default call "
                + "site has today — this is the arithmetic, not the wiring")
        XCTAssertEqual(HeldGate.unwindBudget(5), 5, "the not-held arm's site")
        XCTAssertEqual(
            HeldGate.unwindBudget(0.5), 1,
            "a fast-fail site is floored, not left at its raw handshake — the "
                + "floor is what keeps a short bound from being flaky")
        XCTAssertEqual(
            HeldGate.unwindBudget(0), 1, "the floor holds at the degenerate end")
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
    /// before `deinit`. Drop it and the bail still unwinds, just
    /// `unwindBudget(5)` = 5 s later and with a spurious second diagnostic
    /// (measured 5.7 s; it was 10.8 s before #69 derived that budget from this
    /// call site's own `seconds: 5`). Stated as the expression rather than a
    /// bare number for the same reason arm 1's ceiling is — this figure had
    /// ALREADY gone stale once, silently, when #69 fixed its sibling.
    /// What this test genuinely pins is
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
