import XCTest

@testable import AthenaCore

/// ADR 029 — the inference execution gate is MLX-free, so its serialization /
/// FIFO / cancellation / revert-knob invariants run in CI (ADR 009).
final class InferenceGateTests: XCTestCase {

    override func setUp() { InferenceGate.enabled = true }
    override func tearDown() { InferenceGate.enabled = true }

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
            for _ in 0..<20 {
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
            for _ in 0..<6 {
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
}
