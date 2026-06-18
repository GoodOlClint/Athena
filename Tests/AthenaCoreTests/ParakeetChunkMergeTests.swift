import Foundation
import XCTest

@testable import AthenaTranscription

/// Pure (MLX-free) long-audio chunk stitching (ADR 020 S4). Always runs in CI
/// (ADR 008/009). Pins dedup of overlapping chunk tokens via the contiguous /
/// LCS / midpoint-cut ladder ported from the reference.
final class ParakeetChunkMergeTests: XCTestCase {
    private typealias Token = ParakeetAlignment.Token
    private func tok(_ id: Int, _ start: Double, _ dur: Double = 1) -> Token {
        Token(id: id, text: "t\(id)", start: start, duration: dur)
    }

    func testEmptyInputs() {
        let b = [tok(1, 0)]
        XCTAssertEqual(
            ParakeetChunkMerge.merge([], b, overlapDuration: 4).map(\.id), [1])
        XCTAssertEqual(
            ParakeetChunkMerge.merge(b, [], overlapDuration: 4).map(\.id), [1])
        XCTAssertEqual(
            ParakeetChunkMerge.merge([], [], overlapDuration: 4).count, 0)
    }

    func testNonOverlappingConcatenates() {
        // a ends before b starts → straight concat, no merge.
        let a = [tok(1, 0), tok(2, 2)]  // ends at 3
        let b = [tok(3, 5), tok(4, 7)]
        let r = ParakeetChunkMerge.merge(a, b, overlapDuration: 4)
        XCTAssertEqual(r.map(\.id), [1, 2, 3, 4])
    }

    func testContiguousOverlapDeduplicates() {
        // a and b share ids 3,4 in the time overlap → stitched once, not twice.
        let a = [tok(1, 0), tok(2, 2), tok(3, 4), tok(4, 6)]  // ends at 7
        let b = [tok(3, 4), tok(4, 6), tok(5, 8), tok(6, 10)]
        let r = ParakeetChunkMerge.merge(a, b, overlapDuration: 4)
        XCTAssertEqual(r.map(\.id), [1, 2, 3, 4, 5, 6])
        XCTAssertEqual(
            r.map(\.start), [0, 2, 4, 6, 8, 10])
    }

    func testNonMatchingOverlapFallsBackToMidpointCut() {
        // Overlapping in time but no shared ids → LCS finds nothing → cut at
        // the overlap midpoint.
        let a = [tok(1, 0), tok(2, 2)]  // ends at 3
        let b = [tok(9, 2.5), tok(10, 4)]  // starts at 2.5
        let r = ParakeetChunkMerge.merge(a, b, overlapDuration: 4)
        // cutoff = (3 + 2.5)/2 = 2.75 → keep a.end<=2.75 (id1) + b.start>=2.75 (id10)
        XCTAssertEqual(r.map(\.id), [1, 10])
    }

    func testMonotonicAfterMerge() {
        let a = [tok(1, 0), tok(2, 2), tok(3, 4), tok(4, 6)]
        let b = [tok(3, 4), tok(4, 6), tok(5, 8)]
        let r = ParakeetChunkMerge.merge(a, b, overlapDuration: 4)
        for i in 1..<r.count {
            XCTAssertLessThanOrEqual(r[i - 1].start, r[i].start)
        }
    }
}
