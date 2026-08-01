import Foundation
import XCTest

@testable import AthenaCore

/// NC5 (M70.3) — the M60.5 cancellation early-break the AthenaLLM decode
/// loops (today `GuidedSubstrate` and logprob capture; pre-S0 also the
/// vendored speculative loops) share is now `DecodeLoopControl.isCancelled()`. The real
/// loop bodies are MLX-bound, but the predicate they poll and the loop idiom
/// itself are pure and pinned here with a stub counter whose flag flips
/// mid-decode — so a refactor that drops or inverts the check fails CI.
final class DecodeLoopControlTests: XCTestCase {

    /// A counter that reports cancelled once `tokens >= threshold` — models a
    /// disconnect/deadline firing partway through a generation.
    private final class CancelAfterCounter:
        @unchecked Sendable, DecodeProgressCounter
    {
        private let lock = NSLock()
        private var n = 0
        private let threshold: Int
        init(cancelAfter: Int) { self.threshold = cancelAfter }
        func incrementToken() {
            lock.lock(); defer { lock.unlock() }
            n += 1
        }
        var isCancelled: Bool {
            lock.lock(); defer { lock.unlock() }
            return n >= threshold
        }
        var tokens: Int {
            lock.lock(); defer { lock.unlock() }
            return n
        }
    }

    /// No counter bound (the production default outside `collectMetered`):
    /// `isCancelled()` is false, so generation is never spuriously aborted.
    func testNoCounterIsNotCancelled() {
        XCTAssertNil(DecodeProgress.counter)
        XCTAssertFalse(DecodeLoopControl.isCancelled())
    }

    /// The predicate reflects the bound counter's flag through the
    /// `@TaskLocal`, transitioning false→true as the counter flips.
    func testReflectsBoundCounterFlag() {
        let counter = CancelAfterCounter(cancelAfter: 2)
        DecodeProgress.$counter.withValue(counter) {
            XCTAssertFalse(DecodeLoopControl.isCancelled(), "0 tokens: live")
            counter.incrementToken()
            XCTAssertFalse(DecodeLoopControl.isCancelled(), "1 token: live")
            counter.incrementToken()
            XCTAssertTrue(
                DecodeLoopControl.isCancelled(), "2 tokens: cancel fired")
        }
        // Binding is scoped: outside withValue the predicate is false again.
        XCTAssertFalse(DecodeLoopControl.isCancelled())
    }

    /// The exact idiom all four decode loops run — `while out.count <
    /// maxTokens { if DecodeLoopControl.isCancelled() { break }; commit;
    /// incrementToken() }` — must STOP promptly when cancellation fires,
    /// nowhere near maxTokens (the M60 wedge the milestone fixed). Mirrors how
    /// DecodeProgressTests pins the prefill idiom.
    func testDecodeLoopIdiomBreaksOnCancellation() {
        let maxTokens = 1000
        let counter = CancelAfterCounter(cancelAfter: 3)
        var committed = 0
        DecodeProgress.$counter.withValue(counter) {
            decodeIdiom(maxTokens: maxTokens, counter: counter) {
                committed += 1
            }
        }
        XCTAssertEqual(
            committed, 3,
            "loop must break the iteration AFTER the 3rd commit flips the "
                + "flag — not run to maxTokens \(maxTokens)")
    }

    /// Without cancellation the same idiom runs to `maxTokens` — proves the
    /// early-break is the ONLY thing that stopped it short above.
    func testDecodeLoopIdiomRunsToMaxWithoutCancellation() {
        let maxTokens = 16
        let counter = CancelAfterCounter(cancelAfter: .max)  // never fires
        var committed = 0
        DecodeProgress.$counter.withValue(counter) {
            decodeIdiom(maxTokens: maxTokens, counter: counter) {
                committed += 1
            }
        }
        XCTAssertEqual(committed, maxTokens)
    }

    /// `nonisolated` so it reads the `@TaskLocal` dynamically, exactly as the
    /// synchronous AthenaLLM decode loops do.
    private nonisolated func decodeIdiom(
        maxTokens: Int, counter: DecodeProgressCounter, onCommit: () -> Void
    ) {
        var committed = 0
        while committed < maxTokens {
            if DecodeLoopControl.isCancelled() { break }
            committed += 1
            onCommit()
            counter.incrementToken()
        }
    }
}
