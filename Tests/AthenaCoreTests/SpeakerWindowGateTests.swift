import AthenaTranscription
import Foundation
import XCTest

/// ND15 — `SpeakerWindowGate.keptIndices` is the pure relative-silence gate
/// extracted from `MLXSpeakerEmbeddingModule` (M25.3), so its decision runs in
/// CI (ADR 009) without an embedding model. Keep windows ≥ 20% of the loudest;
/// silent audio keeps all; the loudest window always survives.
final class SpeakerWindowGateTests: XCTestCase {

    func testEmptyInput() {
        XCTAssertEqual(SpeakerWindowGate.keptIndices(rms: []), [])
    }

    func testUniformlySilentKeepsAll() {
        // maxRMS == 0 → the gate is disabled; every window is kept.
        XCTAssertEqual(
            SpeakerWindowGate.keptIndices(rms: [0, 0, 0]), [0, 1, 2])
    }

    func testUniformEnergyKeepsAll() {
        // All equal → each sits at the max, well above 20% of it.
        XCTAssertEqual(
            SpeakerWindowGate.keptIndices(rms: [0.4, 0.4, 0.4]), [0, 1, 2])
    }

    func testDropsBelowTwentyPercentOfLoudest() {
        // max=1.0, gate=0.2 → drop 0.1 and 0.0; keep 1.0 and 0.5.
        XCTAssertEqual(
            SpeakerWindowGate.keptIndices(rms: [1.0, 0.1, 0.5, 0.0]), [0, 2])
    }

    func testGateBoundaryIsInclusive() {
        // A window exactly at the gate (0.2 == 0.2*1.0) is kept (≥, not >).
        XCTAssertEqual(
            SpeakerWindowGate.keptIndices(rms: [1.0, 0.2]), [0, 1])
    }

    func testLoudestAlwaysSurvives() {
        // Even a lone loud spike among silence keeps at least that window.
        XCTAssertEqual(
            SpeakerWindowGate.keptIndices(rms: [0.0, 0.0, 1.0, 0.0]), [2])
    }

    func testCustomGateFraction() {
        // gateFraction 1.0 → only windows at the maximum survive.
        XCTAssertEqual(
            SpeakerWindowGate.keptIndices(
                rms: [1.0, 0.5, 1.0], gateFraction: 1.0),
            [0, 2])
    }

    func testOrderPreserved() {
        // Kept indices come back in ascending (input) order.
        let kept = SpeakerWindowGate.keptIndices(
            rms: [0.9, 0.05, 0.8, 0.3, 0.02])
        XCTAssertEqual(kept, kept.sorted())
        XCTAssertEqual(kept, [0, 2, 3])  // gate = 0.18
    }
}
