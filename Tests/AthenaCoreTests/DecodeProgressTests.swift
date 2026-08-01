import AthenaCore
import Foundation
import XCTest

/// M48.1 — TaskLocal propagation contract for `DecodeProgress.counter`.
/// Pins BOTH sides of the bug fixed in M48.1:
///
/// 1. A Task spawned BEFORE `DecodeProgress.$counter.withValue(...)`
///    does NOT see the binding — its TaskLocal table was captured at
///    spawn time and the later withValue scope can't reach into it.
///    The original M46.8 wiring (`events` constructed at the call
///    site, drained inside the withValue) hit exactly this shape and
///    silently broke the structured-path heartbeat for every release
///    from M46.8 (v0.10.63) through M47.2 (v0.10.68).
///
/// 2. A Task spawned INSIDE the withValue scope DOES see the binding
///    — withValue inherits to children created during its lifetime.
///    The M48.1 fix lifts the events-stream construction inside the
///    withValue closure so the AsyncStream's internal Task captures
///    the binding when it is spawned.
///
/// Pure, CI-safe — no MLX, no model, just the Swift concurrency
/// contract.
final class DecodeProgressTaskLocalTests: XCTestCase {

    /// Spawning a Task before binding the TaskLocal ⇒ the Task's
    /// table never picks up the binding. This is the bug shape.
    func testPreSpawnedTaskDoesNotSeeBinding() async {
        let counter = TestCounter()
        // Spawn the work stream WITHOUT a binding in scope.
        let events = AsyncStream<Int> { cont in
            Task {
                DecodeProgress.counter?.incrementToken()
                cont.yield(1)
                cont.finish()
            }
        }
        // Bind, then drain. The Task already started outside the
        // binding scope, so its incrementToken is a no-op.
        await DecodeProgress.$counter.withValue(counter) {
            for await _ in events {}
        }
        XCTAssertEqual(
            counter.tokens, 0,
            "Pre-spawned Task must NOT see a later TaskLocal binding "
                + "— this is what masked the structured-path heartbeat "
                + "in M46.8 → M47.2.")
    }

    /// Spawning a Task INSIDE the binding ⇒ the Task's table inherits
    /// the binding. This is the M48.1 fix shape.
    func testTaskSpawnedInsideBindingSeesIt() async {
        let counter = TestCounter()
        await DecodeProgress.$counter.withValue(counter) {
            // Build the stream inside the binding so the Task it
            // spawns inherits the TaskLocal table at spawn time.
            let events = AsyncStream<Int> { cont in
                Task {
                    DecodeProgress.counter?.incrementToken()
                    DecodeProgress.counter?.incrementToken()
                    cont.yield(1)
                    cont.finish()
                }
            }
            for await _ in events {}
        }
        XCTAssertEqual(
            counter.tokens, 2,
            "Task spawned inside withValue must inherit the binding "
                + "— this is what M48.1 restores for the heartbeat.")
    }

    /// Bonus: the binding is dynamic-scoped, not lexically captured.
    /// A `nonisolated` function called from inside withValue can
    /// read it without taking it as a parameter — that's the whole
    /// point of using a TaskLocal in the first place (no signature
    /// plumbing through five layers of generate/runSpeculative/
    /// container.perform/closure/decode loop).
    func testNonisolatedFunctionReadsBindingDynamically() {
        let counter = TestCounter()
        DecodeProgress.$counter.withValue(counter) {
            Self.incrementDynamically(times: 3)
        }
        XCTAssertEqual(counter.tokens, 3)
    }

    nonisolated private static func incrementDynamically(times: Int) {
        for _ in 0 ..< times {
            DecodeProgress.counter?.incrementToken()
        }
    }
}

/// M49.2 — phase classification for the heartbeat. Pins the three
/// state transitions an operator reads:
///   request entry → setup (no prefill data, no commits)
///                 → prefill (chunks submitted, none committed)
///                 → decode (any token committed, or prefill complete)
///
/// These pin the pure function, not a live path. No in-tree producer has
/// published prefill counts since publication S0 (#47 removed the last
/// vestiges of that plumbing), so every real caller now takes the defaulted
/// `prefillTotal: 0` route and the `.prefill` arm is unreachable in the
/// daemon. The arm is still worth pinning: `from` is public API of the
/// `AthenaCore` library product and must stay correct for a caller that can
/// supply the counts.
final class DecodePhaseTests: XCTestCase {

    func testNoSignalIsSetup() {
        XCTAssertEqual(
            DecodePhase.from(tokens: 0, prefillCompleted: 0, prefillTotal: 0),
            .setup,
            "Fresh request before any progress publishes ⇒ setup.")
    }

    func testPrefillInProgress() {
        XCTAssertEqual(
            DecodePhase.from(tokens: 0, prefillCompleted: 1, prefillTotal: 22),
            .prefill)
        XCTAssertEqual(
            DecodePhase.from(tokens: 0, prefillCompleted: 13, prefillTotal: 22),
            .prefill)
        XCTAssertEqual(
            DecodePhase.from(tokens: 0, prefillCompleted: 21, prefillTotal: 22),
            .prefill,
            "Anything strictly less than total ⇒ still prefill.")
    }

    func testPrefillCompleteWithoutTokensIsDecode() {
        XCTAssertEqual(
            DecodePhase.from(tokens: 0, prefillCompleted: 22, prefillTotal: 22),
            .decode,
            "Prefill done + no commit yet is the transient between "
                + "prefill and first decoded token — classify as decode "
                + "because we're past the prefill workload.")
    }

    func testAnyTokenIsDecode() {
        XCTAssertEqual(
            DecodePhase.from(tokens: 1, prefillCompleted: 22, prefillTotal: 22),
            .decode)
        XCTAssertEqual(
            DecodePhase.from(tokens: 1000, prefillCompleted: 22, prefillTotal: 22),
            .decode)
    }

    func testTokensWithoutPrefillStateIsDecode() {
        // Substrate-streamed (non-Guide) path doesn't publish prefill
        // chunks — it goes straight to incrementing tokens. Phase
        // must still resolve as decode.
        XCTAssertEqual(
            DecodePhase.from(tokens: 5, prefillCompleted: 0, prefillTotal: 0),
            .decode)
    }

    func testRawValuesMatchLogFieldConvention() {
        // Heartbeat log line embeds `.rawValue` directly. If these
        // change, operator dashboards/greps that key off
        // `phase=setup|prefill|decode` will silently break.
        XCTAssertEqual(DecodePhase.setup.rawValue, "setup")
        XCTAssertEqual(DecodePhase.prefill.rawValue, "prefill")
        XCTAssertEqual(DecodePhase.decode.rawValue, "decode")
    }
}

/// NSLock-isolated counter that satisfies `DecodeProgressCounter`.
/// Tests assert against `tokens` after draining the stream.
private final class TestCounter: @unchecked Sendable, DecodeProgressCounter {
    private let lock = NSLock()
    private var n = 0
    func incrementToken() {
        lock.lock(); defer { lock.unlock() }
        n += 1
    }
    var tokens: Int {
        lock.lock(); defer { lock.unlock() }
        return n
    }
}
