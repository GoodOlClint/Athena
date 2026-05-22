import Foundation
import XCTest

@testable import AthenaTranscription

/// M26.1 — the timestamp-segment parser + per-segment `avgLogprob` are
/// pure (no MLX execution), so they run in CI. Token streams mimic
/// Whisper's paired `<|t|>` markers around content tokens.
final class WhisperDecodeParseTests: XCTestCase {
    private let tb = WhisperDecode.timestampBegin
    private let step = WhisperDecode.timeStep  // 0.02 s

    /// Frame id for a timestamp at `seconds`.
    private func ts(_ seconds: Double) -> Int {
        tb + Int((seconds / step).rounded())
    }

    func testSingleSegmentBoundsAndAvgLogprob() {
        // <|0.00|> 100 200 <|2.00|>
        let gen = [ts(0), 100, 200, ts(2)]
        let lp = [-9.0, -0.5, -1.5, -9.0]  // only the two text logprobs count
        let segs = WhisperDecode.parseSegments(generated: gen, logprobs: lp)
        XCTAssertEqual(segs.count, 1)
        XCTAssertEqual(segs[0].start, 0, accuracy: 1e-9)
        XCTAssertEqual(segs[0].end, 2, accuracy: 1e-9)
        XCTAssertEqual(segs[0].tokens, [100, 200])
        XCTAssertEqual(segs[0].avgLogprob!, -1.0, accuracy: 1e-9)
    }

    func testTwoSegmentsPairedTimestamps() {
        // <|0.00|> a b <|2.00|> <|2.00|> c <|4.00|>
        let gen = [ts(0), 100, 200, ts(2), ts(2), 300, ts(4)]
        let lp = [-9, -2, -4, -9, -9, -3, -9].map(Double.init)
        let segs = WhisperDecode.parseSegments(generated: gen, logprobs: lp)
        XCTAssertEqual(segs.count, 2)
        XCTAssertEqual(segs[0].tokens, [100, 200])
        XCTAssertEqual(segs[0].avgLogprob!, -3.0, accuracy: 1e-9)
        XCTAssertEqual(segs[1].start, 2, accuracy: 1e-9)
        XCTAssertEqual(segs[1].end, 4, accuracy: 1e-9)
        XCTAssertEqual(segs[1].tokens, [300])
        XCTAssertEqual(segs[1].avgLogprob!, -3.0, accuracy: 1e-9)
    }

    /// Trailing content with no closing timestamp falls back to the
    /// window length so the last span is never dropped.
    func testUnterminatedSegmentUsesLastTimestamp() {
        let gen = [ts(5), 100, 200]
        let lp = [-9.0, -1.0, -1.0]
        let segs = WhisperDecode.parseSegments(generated: gen, logprobs: lp)
        XCTAssertEqual(segs.count, 1)
        XCTAssertEqual(segs[0].start, 5, accuracy: 1e-9)
        XCTAssertEqual(segs[0].end, 5, accuracy: 1e-9)  // lastTs
    }

    /// Logprob tracking is optional: an empty parallel array yields nil
    /// `avgLogprob` but still parses the spans (back-compat).
    func testMissingLogprobsYieldsNilAvg() {
        let gen = [ts(0), 100, 200, ts(2)]
        let segs = WhisperDecode.parseSegments(generated: gen, logprobs: [])
        XCTAssertEqual(segs.count, 1)
        XCTAssertNil(segs[0].avgLogprob)
    }
}
