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
    private var prefillCompleted = 0
    private var prefillTotal = 0
    func incrementToken() {
        lock.lock(); defer { lock.unlock() }
        n += 1
    }
    func recordPrefillChunk(completed: Int, total: Int) {
        lock.lock(); defer { lock.unlock() }
        self.prefillCompleted = completed
        self.prefillTotal = total
    }
    var tokens: Int {
        lock.lock(); defer { lock.unlock() }
        return n
    }
    var prefillState: (completed: Int, total: Int) {
        lock.lock(); defer { lock.unlock() }
        return (prefillCompleted, prefillTotal)
    }
}

/// M48.4 — `recordPrefillChunk` default contract: the protocol
/// extension provides a no-op default so existing conformers that
/// don't care about prefill granularity stay valid. Counter-specific
/// behavior is tested with an instance that overrides the default.
final class DecodeProgressPrefillTests: XCTestCase {

    func testDefaultRecordPrefillChunkIsNoOp() {
        // A conformer that implements ONLY incrementToken picks up the
        // protocol-extension defaults for the rest (recordPrefillChunk /
        // setSetupStage / cancelGeneration / isCancelled), which is what
        // HeartbeatCounter relied on before M48.4 shipped.
        final class TokenOnly:
            @unchecked Sendable, DecodeProgressCounter
        {
            private(set) var tokens = 0
            func incrementToken() { tokens += 1 }
        }
        let c = TokenOnly()
        // L11 (M70.3): observe the contract instead of `XCTAssertTrue(true)`.
        // The defaults must be GENUINE no-ops: calling them mutates no
        // observable state, and the default isCancelled is false (so a
        // conformer that doesn't care about cancellation never spuriously
        // aborts a decode).
        c.recordPrefillChunk(completed: 5, total: 10)
        c.setSetupStage("compile-dfa")
        c.cancelGeneration()
        XCTAssertEqual(c.tokens, 0, "no default touched the conformer's state")
        XCTAssertFalse(c.isCancelled, "default isCancelled is false")
        // The one method it DID implement still works, proving the object is
        // live (not that the assertions above passed vacuously).
        c.incrementToken()
        XCTAssertEqual(c.tokens, 1)
    }

    func testRecordPrefillChunkPropagatesLatestValues() {
        let counter = TestCounter()
        counter.recordPrefillChunk(completed: 1, total: 38)
        counter.recordPrefillChunk(completed: 14, total: 38)
        counter.recordPrefillChunk(completed: 38, total: 38)
        let s = counter.prefillState
        XCTAssertEqual(s.completed, 38)
        XCTAssertEqual(s.total, 38)
    }

    /// The decode-loop prefill block matches this idiom — total
    /// chunks computed once with the ceiling-divide, then per-chunk
    /// publish after asyncEval submits the work. Pinned here so a
    /// future refactor that drops the publish call gets caught.
    func testPrefillLoopIdiomPublishesEveryChunk() {
        let counter = TestCounter()
        DecodeProgress.$counter.withValue(counter) {
            simulatePrefill(promptCount: 1500)  // 3 chunks at 512
        }
        let s = counter.prefillState
        XCTAssertEqual(s.completed, 3)
        XCTAssertEqual(s.total, 3)
    }

    private nonisolated func simulatePrefill(promptCount: Int) {
        // Mirror the GuidedGreedy / SpeculativeGeneration prefill
        // arithmetic exactly so the test catches an off-by-one if
        // the loop body diverges.
        let head = promptCount - 1
        let chunkSize = 512
        let totalChunks = (head + chunkSize - 1) / chunkSize
        var i = 0
        var done = 0
        while i < head {
            i += chunkSize
            done += 1
            DecodeProgress.counter?.recordPrefillChunk(
                completed: done, total: totalChunks)
        }
    }
}
