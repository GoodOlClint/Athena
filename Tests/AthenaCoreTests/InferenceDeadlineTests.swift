import XCTest

@testable import AthenaCore

/// M33.1 — the per-request inference deadline. Tests the nanosecond
/// cores at sub-second granularity so they stay fast + deterministic.
final class InferenceDeadlineTests: XCTestCase {

    func testFastBodyReturnsBeforeDeadline() async throws {
        let v = try await withDeadlineNanos(
            500_000_000, timeout: .requestTimedOut(seconds: 1)
        ) { 42 }
        XCTAssertEqual(v, 42)
    }

    func testSlowBodyThrowsClassifiedTimeout() async {
        do {
            _ = try await withDeadlineNanos(
                50_000_000, timeout: .requestTimedOut(seconds: 7)
            ) { () -> Int in
                try? await Task.sleep(nanoseconds: 2_000_000_000)
                return 0
            }
            XCTFail("expected the deadline to fire")
        } catch let e as AthenaError {
            XCTAssertEqual(e, .requestTimedOut(seconds: 7))
            XCTAssertEqual(e.httpStatus, 504)
            XCTAssertEqual(e.code, "inference_timeout")
        } catch {
            XCTFail("unexpected error \(error)")
        }
    }

    func testZeroSecondsIsUnbounded() async throws {
        // seconds <= 0 ⇒ the deadline is disabled; the body runs to
        // completion even though it outlives any sane test timeout window.
        let v = try await withInferenceDeadline(seconds: 0) {
            try? await Task.sleep(nanoseconds: 10_000_000)
            return "done"
        }
        XCTAssertEqual(v, "done")
    }

    func testStreamTruncatedAtDeadline() async {
        // Issue #4: the old form paced a 100-element source at 40 ms/element
        // under a 120 ms deadline and asserted `count > 0` — i.e. it bet the
        // first element would be produced inside the window. Instead: buffer
        // three elements up front and NEVER finish the source, so the ONLY way
        // this loop can terminate is the deadline truncating it, and the ready
        // elements are delivered regardless of scheduling.
        var count = 0
        for await _ in deadlineBoundedNanos(500_000_000, neverFinishing(3)) {
            count += 1
        }
        XCTAssertEqual(
            count, 3, "buffered elements delivered, then the deadline ended it")
    }

    func testStreamZeroSecondsPassesEverythingThrough() async {
        let source = AsyncStream<Int> { cont in
            for i in 0 ..< 5 { cont.yield(i) }
            cont.finish()
        }
        var got: [Int] = []
        for await x in deadlineBounded(seconds: 0, source) { got.append(x) }
        XCTAssertEqual(got, [0, 1, 2, 3, 4])
    }

    func testStreamTruncationFiresCallback() async {
        // M46.1 — when the deadline races ahead of the source, the
        // `onTimerFired` closure must fire exactly once so the caller's
        // logger can record the truncation (the daemon-side analog of
        // the M45.7 5xx legibility fix for the streaming path).
        // A source that never finishes ⇒ the timer always wins the race, with
        // no dependence on how fast the runner paces the source (issue #4).
        let fired = TestFlag()
        let bounded = deadlineBoundedNanos(200_000_000, neverFinishing(3)) {
            fired.set()
        }
        for await _ in bounded {}
        XCTAssertTrue(
            fired.value(),
            "onTimerFired must fire when the deadline truncates the stream"
        )
    }

    func testStreamFinishingNaturallyDoesNotFireCallback() async {
        // M46.1 — symmetric: a stream that finishes BEFORE the deadline
        // must NOT call onTimerFired. Otherwise the daemon log gets a
        // false-positive truncation line on every short, normal
        // completion.
        let source = AsyncStream<Int> { cont in
            for i in 0 ..< 3 { cont.yield(i) }
            cont.finish()
        }
        let fired = TestFlag()
        let start = ContinuousClock.now  // the timer starts at stream creation
        let bounded = deadlineBoundedNanos(2_000_000_000, source) {
            fired.set()
        }
        for await _ in bounded {}
        // Wait past the deadline so any latent timer would have fired —
        // anchored to a MONOTONIC start, so the 200 ms margin holds no matter
        // how long draining the source took (issue #4).
        try? await Task.sleep(
            until: start + .milliseconds(2_200), clock: .continuous)
        XCTAssertFalse(
            fired.value(),
            "onTimerFired must not fire when the stream finishes "
                + "naturally before the deadline"
        )
    }

    func testDownstreamCancelDoesNotFireCallbackNE6() async {
        // NE6 — a consumer that breaks early (client disconnect) BEFORE the
        // deadline must not get a spurious "deadline truncated" callback: the
        // cancelled timer must return before markTimerFiredIfFirst, so the
        // operator isn't misled into chasing a timeout that never happened.
        let source = AsyncStream<Int> { cont in
            let t = Task {
                var i = 0
                while !Task.isCancelled {
                    cont.yield(i)
                    i += 1
                    try? await Task.sleep(nanoseconds: 10_000_000)
                }
                cont.finish()
            }
            cont.onTermination = { _ in t.cancel() }
        }
        let fired = TestFlag()
        // A long deadline so it can't fire by elapsed time within the test —
        // the only way the callback could fire is the bug (timer firing on a
        // downstream cancel).
        let bounded = deadlineBoundedNanos(5_000_000_000, source) {
            fired.set()
        }
        var seen = 0
        for await _ in bounded {
            seen += 1
            if seen >= 1 { break }  // simulate a disconnect mid-stream
        }
        // Give the now-cancelled timer time to (wrongly) fire if the bug is back.
        try? await Task.sleep(nanoseconds: 200_000_000)
        XCTAssertFalse(
            fired.value(),
            "a downstream cancel must not fire the deadline-truncation callback"
        )
    }
}

/// A source that hands over `n` ready elements and then never finishes, so the
/// ONLY thing that can end the stream is the deadline — no bet on how fast the
/// runner paces a source (issue #4). The 30 s tail is a hang guard: if
/// truncation ever regressed, the caller fails on the element count instead of
/// blocking CI forever. It is cancelled on the (normal) truncation path.
private func neverFinishing(_ n: Int) -> AsyncStream<Int> {
    AsyncStream<Int> { cont in
        for i in 0 ..< n { cont.yield(i) }
        let hangGuard = Task {
            try? await Task.sleep(for: .seconds(30))
            cont.yield(-1)  // extra element ⇒ a clean count mismatch
            cont.finish()
        }
        cont.onTermination = { _ in hangGuard.cancel() }
    }
}

/// Bool for test assertions that touch state across child tasks (the
/// timer/pump closures inside deadlineBoundedNanos), backed by the suite's
/// shared `Locked` box (#70).
private final class TestFlag: Sendable {
    private let fired = Locked(false)
    func set() { fired.mutate { $0 = true } }
    func value() -> Bool { fired.current }
}
