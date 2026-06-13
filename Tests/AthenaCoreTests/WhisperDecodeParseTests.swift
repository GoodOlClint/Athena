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

    /// D9: content arriving after a CLOSING timestamp (no new opening ts)
    /// must start at the last timestamp, not 0 — else the trailing span is
    /// emitted out of order at the head of the window.
    func testContentAfterClosingTimestampStartsAtLastTs() {
        let gen = [ts(0), 100, ts(2), 200]  // '200' after the closing <|2|>
        let lp = [-9.0, -1.0, -9.0, -1.0]
        let segs = WhisperDecode.parseSegments(generated: gen, logprobs: lp)
        XCTAssertEqual(segs.count, 2)
        XCTAssertEqual(segs[1].tokens, [200])
        XCTAssertEqual(
            segs[1].start, 2, accuracy: 1e-9,
            "trailing span starts at lastTs (2.0), not 0")
    }

    /// D5 + ND3: the full Whisper language table round-trips a language past
    /// the old 16-entry truncation in BOTH directions (forced + reported),
    /// instead of silently collapsing to English.
    func testLanguageTableCoversBeyondSixteen() {
        let hi = WhisperDecode.languageToken("hi")  // index 17
        XCTAssertEqual(WhisperDecode.languageCode(forToken: hi), "hi")
        XCTAssertNotEqual(hi, WhisperDecode.languageToken("en"))
        let yue = WhisperDecode.languageToken("yue")  // large-v3 addition
        XCTAssertEqual(WhisperDecode.languageCode(forToken: yue), "yue")
    }
}
