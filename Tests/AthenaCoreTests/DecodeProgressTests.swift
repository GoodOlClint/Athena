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

/// M49.2 — phase classification for the heartbeat. Two states an operator
/// reads: setup (no commits yet — request prep, DFA compile, vocab build, and
/// the prompt prefill) then decode (any token committed).
///
/// The `.prefill` case and its arms are gone as of #47: publication S0 deleted
/// the only producer of prefill counts, the substrate's
/// `TokenIterator(prefillStepSize:)` exposes no per-chunk hook to replace it,
/// and the mlx tracker has no upstream work scheduled that would restore one.
/// Keeping an unreachable case as public API was the alternative and was
/// rejected — nothing has shipped a release yet, so the source break is free.
final class DecodePhaseTests: XCTestCase {

    func testNoTokensIsSetup() {
        XCTAssertEqual(
            DecodePhase.from(tokens: 0), .setup,
            "Before the first committed token the request is in setup — "
                + "which now spans the prefill, so a long setup phase on a "
                + "large prompt is expected, not a hang.")
    }

    func testAnyTokenIsDecode() {
        XCTAssertEqual(DecodePhase.from(tokens: 1), .decode)
        XCTAssertEqual(DecodePhase.from(tokens: 1000), .decode)
    }

    func testRawValuesMatchLogFieldConvention() {
        // Heartbeat log line embeds `.rawValue` directly. If these change,
        // operator dashboards/greps that key off `phase=setup|decode` break.
        XCTAssertEqual(DecodePhase.setup.rawValue, "setup")
        XCTAssertEqual(DecodePhase.decode.rawValue, "decode")
        XCTAssertEqual(
            DecodePhase.allCases.map(\.rawValue), ["setup", "decode"],
            "a new phase must be a deliberate operator-surface change")
    }
}

/// Counter satisfying `DecodeProgressCounter`, backed by the suite's shared
/// `Locked` box (#70). Tests assert against `tokens` after draining the stream.
private final class TestCounter: @unchecked Sendable, DecodeProgressCounter {
    private let n = Locked(0)
    func incrementToken() { n.mutate { $0 += 1 } }
    var tokens: Int { n.current }
}
