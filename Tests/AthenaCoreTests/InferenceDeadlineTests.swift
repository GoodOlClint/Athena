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
        // A source that emits one element every 40 ms, bounded by a 120 ms
        // deadline, must stop well before its 100 elements are exhausted.
        let source = AsyncStream<Int> { cont in
            let t = Task {
                for i in 0..<100 {
                    if Task.isCancelled { break }
                    cont.yield(i)
                    try? await Task.sleep(nanoseconds: 40_000_000)
                }
                cont.finish()
            }
            cont.onTermination = { _ in t.cancel() }
        }
        var count = 0
        for await _ in deadlineBoundedNanos(120_000_000, source) {
            count += 1
        }
        XCTAssertGreaterThan(count, 0)
        XCTAssertLessThan(count, 100)
    }

    func testStreamZeroSecondsPassesEverythingThrough() async {
        let source = AsyncStream<Int> { cont in
            for i in 0..<5 { cont.yield(i) }
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
        let source = AsyncStream<Int> { cont in
            let t = Task {
                for i in 0..<100 {
                    if Task.isCancelled { break }
                    cont.yield(i)
                    try? await Task.sleep(nanoseconds: 40_000_000)
                }
                cont.finish()
            }
            cont.onTermination = { _ in t.cancel() }
        }
        let fired = TestFlag()
        let bounded = deadlineBoundedNanos(120_000_000, source) {
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
            for i in 0..<3 { cont.yield(i) }
            cont.finish()
        }
        let fired = TestFlag()
        let bounded = deadlineBoundedNanos(2_000_000_000, source) {
            fired.set()
        }
        for await _ in bounded {}
        // Wait past the deadline so any latent timer would have fired.
        try? await Task.sleep(nanoseconds: 2_100_000_000)
        XCTAssertFalse(
            fired.value(),
            "onTimerFired must not fire when the stream finishes "
                + "naturally before the deadline"
        )
    }
}

/// NSLock-isolated bool for test assertions that touch state across
/// child tasks (the timer/pump closures inside deadlineBoundedNanos).
private final class TestFlag: @unchecked Sendable {
    private let lock = NSLock()
    private var fired = false
    func set() {
        lock.lock()
        defer { lock.unlock() }
        fired = true
    }
    func value() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return fired
    }
}
