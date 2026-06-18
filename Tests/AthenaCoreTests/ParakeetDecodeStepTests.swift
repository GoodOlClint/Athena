import Foundation
import XCTest

@testable import AthenaTranscription

/// Pure (MLX-free) greedy-TDT step control flow (ADR 020 S4, ADR 009).
/// Always runs in CI. Pins the frame-pointer advance + anti-stall guard.
final class ParakeetDecodeStepTests: XCTestCase {

    func testNonZeroDurationAdvancesAndResets() {
        // A real duration moves the pointer by that many frames and clears the
        // consecutive-zero counter regardless of its prior value.
        let r = ParakeetDecodeStep.advance(
            duration: 2, priorNewSymbols: 5, maxSymbols: 10)
        XCTAssertEqual(r.stepDelta, 2)
        XCTAssertEqual(r.newSymbols, 0)
    }

    func testZeroDurationHoldsPointerAndCounts() {
        // Zero duration keeps the pointer; counter increments.
        let r = ParakeetDecodeStep.advance(
            duration: 0, priorNewSymbols: 0, maxSymbols: 10)
        XCTAssertEqual(r.stepDelta, 0)
        XCTAssertEqual(r.newSymbols, 1)

        let r2 = ParakeetDecodeStep.advance(
            duration: 0, priorNewSymbols: 7, maxSymbols: 10)
        XCTAssertEqual(r2.stepDelta, 0)
        XCTAssertEqual(r2.newSymbols, 8)
    }

    func testAntiStallForcesAdvanceAtCap() {
        // Reaching maxSymbols consecutive zero-duration steps forces +1 frame
        // and resets, so the decode can't loop forever at one frame.
        let r = ParakeetDecodeStep.advance(
            duration: 0, priorNewSymbols: 9, maxSymbols: 10)
        XCTAssertEqual(r.stepDelta, 1)
        XCTAssertEqual(r.newSymbols, 0)
    }

    func testAntiStallBoundedDecodeTerminates() {
        // Simulate a pathological all-zero-duration stream: with the guard,
        // the pointer still advances ~1 frame every maxSymbols steps, so a
        // T-frame clip terminates in a bounded number of steps.
        let T = 100, maxSymbols = 10
        var step = 0, newSymbols = 0, iters = 0
        while step < T {
            let r = ParakeetDecodeStep.advance(
                duration: 0, priorNewSymbols: newSymbols, maxSymbols: maxSymbols)
            step += r.stepDelta
            newSymbols = r.newSymbols
            iters += 1
            XCTAssertLessThan(iters, T * maxSymbols + 10, "decode did not terminate")
        }
        XCTAssertGreaterThanOrEqual(step, T)
    }
}
