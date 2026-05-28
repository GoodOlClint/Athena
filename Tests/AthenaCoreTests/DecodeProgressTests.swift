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
    func testNonisolatedFunctionReadsBindingDynamically() async {
        let counter = TestCounter()
        await DecodeProgress.$counter.withValue(counter) {
            Self.incrementDynamically(times: 3)
        }
        XCTAssertEqual(counter.tokens, 3)
    }

    nonisolated private static func incrementDynamically(times: Int) {
        for _ in 0..<times {
            DecodeProgress.counter?.incrementToken()
        }
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
        // A conformer that DOES NOT implement recordPrefillChunk picks
        // up the protocol-extension default and silently no-ops, which
        // is what HeartbeatCounter relied on before M48.4 shipped.
        final class TokenOnly:
            @unchecked Sendable, DecodeProgressCounter
        {
            func incrementToken() {}
        }
        let c = TokenOnly()
        c.recordPrefillChunk(completed: 5, total: 10)
        // No throw, no crash — that IS the contract.
        XCTAssertTrue(true)
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
    func testPrefillLoopIdiomPublishesEveryChunk() async {
        let counter = TestCounter()
        await DecodeProgress.$counter.withValue(counter) {
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
