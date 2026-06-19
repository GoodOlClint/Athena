import Foundation
import XCTest

@testable import AthenaCore

/// C2 (ADR 013 §4) — `LogprobMath.fromLogitRow` is the MLX-free reference for
/// per-token logprob capture, so its algebra runs in CI (ADR 009). Uses small
/// synthetic logit rows; the real decode paths compute the same with MLX.
final class LogprobMathTests: XCTestCase {

    /// log_softmax of a uniform row: every logprob = −ln(n).
    func testUniformRowLogprobs() {
        let row: [Float] = [0, 0, 0, 0]  // n=4 ⇒ ln(4) ≈ 1.3863
        let (lp, top) = LogprobMath.fromLogitRow(row, chosen: 2, topK: 4)
        XCTAssertEqual(lp, -Float(log(4.0)), accuracy: 1e-5)
        XCTAssertEqual(top.count, 4)
        for t in top {
            XCTAssertEqual(t.logprob, -Float(log(4.0)), accuracy: 1e-5)
        }
    }

    /// All logprobs are ≤ 0 and the chosen token's logprob equals its
    /// log_softmax. The most-likely token has the highest (least negative).
    func testChosenLogprobAndOrdering() {
        let row: [Float] = [1, 3, 2]  // argmax = index 1
        let (lp, top) = LogprobMath.fromLogitRow(row, chosen: 0, topK: 3)
        // log_softmax[0] = 1 − logSumExp([1,3,2]).
        let lse = Float(log(exp(1.0) + exp(3.0) + exp(2.0)))
        XCTAssertEqual(lp, 1 - lse, accuracy: 1e-5)
        XCTAssertLessThanOrEqual(lp, 0)
        // top is descending by logprob: id 1 (logit 3) first, then 2, then 0.
        XCTAssertEqual(top.map { $0.token }, [1, 2, 0])
        XCTAssertEqual(top.first!.logprob, 3 - lse, accuracy: 1e-5)
        // strictly descending
        XCTAssertGreaterThan(top[0].logprob, top[1].logprob)
        XCTAssertGreaterThan(top[1].logprob, top[2].logprob)
    }

    func testTopKZeroYieldsNoAlternatives() {
        let (lp, top) = LogprobMath.fromLogitRow([1, 2, 3], chosen: 2, topK: 0)
        XCTAssertTrue(top.isEmpty)
        XCTAssertLessThanOrEqual(lp, 0)
    }

    func testTopKClampedToVocab() {
        let (_, top) = LogprobMath.fromLogitRow([5, 1], chosen: 0, topK: 10)
        XCTAssertEqual(top.count, 2)  // clamped to vocab=2
        XCTAssertEqual(top.map { $0.token }, [0, 1])  // 5 > 1
    }

    func testTiesBreakByLowerIndex() {
        // Equal logits ⇒ stable order by ascending index.
        let (_, top) = LogprobMath.fromLogitRow([2, 2, 2], chosen: 1, topK: 2)
        XCTAssertEqual(top.map { $0.token }, [0, 1])
    }

    func testEmptyRow() {
        let (lp, top) = LogprobMath.fromLogitRow([], chosen: 0, topK: 3)
        XCTAssertEqual(lp, 0)
        XCTAssertTrue(top.isEmpty)
    }

    func testOutOfRangeChosen() {
        let (lp, _) = LogprobMath.fromLogitRow([1, 2], chosen: 9, topK: 1)
        XCTAssertEqual(lp, 0)  // out-of-range ⇒ 0, no crash
    }
}
