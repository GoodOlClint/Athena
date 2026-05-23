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
}
