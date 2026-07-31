import AthenaServerKit
import Foundation
import XCTest

/// Usability audit 2026-07-02 §1 — `athena logs` returned the *oldest*
/// `limit` entries of the window (drop-the-tail bug). `LogTail` is the pure,
/// MLX-free fix (ADR 008/009): drain the oldest-first stream, keep the newest
/// `limit`. These pins FAIL against the old take-first-N behavior and pass
/// against tail-N.
final class LogTailTests: XCTestCase {
    /// The DoD pin: a synthetic oldest-first stream of `limit + K` entries
    /// must yield the LAST `limit`, in order — not the first `limit`.
    func testKeepsNewestNotOldest() {
        let limit = 200
        let k = 137
        var tail = LogTail<Int>(limit: limit)
        for i in 0 ..< (limit + k) { tail.append(i) }  // 0…336, oldest-first
        XCTAssertEqual(tail.entries.count, limit)
        XCTAssertEqual(tail.entries.first, k)  // 137, not 0
        XCTAssertEqual(tail.entries.last, limit + k - 1)  // 336
        XCTAssertEqual(tail.entries, Array(k ..< (limit + k)))
        XCTAssertTrue(tail.truncated)
    }

    /// Under capacity: everything retained, order preserved, not truncated.
    func testUnderCapacityKeepsAllInOrder() {
        var tail = LogTail<Int>(limit: 200)
        for i in 0 ..< 50 { tail.append(i) }
        XCTAssertEqual(tail.entries, Array(0 ..< 50))
        XCTAssertFalse(tail.truncated)
    }

    /// Exactly at capacity is not truncation (nothing dropped).
    func testExactCapacityNotTruncated() {
        var tail = LogTail<Int>(limit: 10)
        for i in 0 ..< 10 { tail.append(i) }
        XCTAssertEqual(tail.entries, Array(0 ..< 10))
        XCTAssertFalse(tail.truncated)
    }

    /// A non-positive limit clamps to 1 (never a zero-capacity buffer).
    func testLimitClampsToOne() {
        var tail = LogTail<Int>(limit: 0)
        for i in 0 ..< 5 { tail.append(i) }
        XCTAssertEqual(tail.entries, [4])
        XCTAssertTrue(tail.truncated)
    }
}
