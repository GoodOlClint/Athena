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
        let holder = Task {
            try await gate.withExclusiveExecution {
                try? await Task.sleep(nanoseconds: 50_000_000)
            }
        }
        try? await Task.sleep(nanoseconds: 5_000_000)  // holder acquires
        let waiter = Task {
            try await gate.withExclusiveExecution { 1 }
        }
        try? await Task.sleep(nanoseconds: 5_000_000)  // waiter enqueues
        waiter.cancel()
        do {
            _ = try await waiter.value
            XCTFail("cancelled waiter should throw")
        } catch is CancellationError {
        } catch { XCTFail("expected CancellationError, got \(error)") }
        _ = try await holder.value
        let held = await gate.isHeld
        let waiters = await gate.waiterCount
        XCTAssertFalse(held)
        XCTAssertEqual(waiters, 0)
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
        let holder = Task {
            try await gate.withExclusiveExecution {
                try? await Task.sleep(nanoseconds: 60_000_000)
            }
        }
        try? await Task.sleep(nanoseconds: 5_000_000)  // holder acquires
        let waiters = (0 ..< 3).map { _ in
            Task { try await gate.withExclusiveExecution {} }
        }
        try? await Task.sleep(nanoseconds: 10_000_000)  // all three enqueue
        let mid = await gate.stats()
        XCTAssertTrue(mid.held)
        XCTAssertEqual(mid.waiters, 3, "expected 3 queued behind the holder")
        XCTAssertEqual(mid.maxWaiters, 3)

        _ = try await holder.value
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
