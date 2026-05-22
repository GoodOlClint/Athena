import Foundation
import XCTest

@testable import AthenaTranscription

/// M26.2 — the word-alignment math (DTW, median filter, weight
/// reduction, subword→word merge) is MLX-free and runs in CI. The model
/// forward that feeds it is exercised by the gated real-audio test.
final class WordAlignMathTests: XCTestCase {

    func testDTWFollowsDiagonal() {
        // Cheapest path is the diagonal: token i ↔ frame i.
        let cost = [
            [0.0, 1, 1],
            [1, 0.0, 1],
            [1, 1, 0.0],
        ]
        let (text, time) = WordAlignMath.dtw(cost)
        XCTAssertEqual(text, [0, 1, 2])
        XCTAssertEqual(time, [0, 1, 2])
        let frames = WordAlignMath.firstFrames(
            textIdx: text, timeIdx: time, n: 3)
        XCTAssertEqual(frames, [0, 1, 2])
    }

    func testFirstFramesAreMonotonic() {
        // A path that lingers on frames: token 0 spans frames 0..1,
        // token 1 frame 2, token 2 frames 3..4.
        let text = [0, 0, 1, 2, 2]
        let time = [0, 1, 2, 3, 4]
        let frames = WordAlignMath.firstFrames(
            textIdx: text, timeIdx: time, n: 3)
        XCTAssertEqual(frames, [0, 2, 3])
        for i in 1 ..< frames.count {
            XCTAssertGreaterThanOrEqual(frames[i], frames[i - 1])
        }
    }

    func testMedianFilterRemovesSpike() {
        let row = [0.0, 0, 9, 0, 0]
        let out = WordAlignMath.medianFilter(row, width: 3)
        XCTAssertEqual(out, [0, 0, 0, 0, 0])
    }

    func testMedianFilterWidthOneIsIdentity() {
        let row = [3.0, 1, 4, 1, 5]
        XCTAssertEqual(WordAlignMath.medianFilter(row, width: 1), row)
    }

    func testReduceWeightsNormalizesAcrossTokens() {
        // One head, two tokens, three frames; token 0 peaks on frame 0,
        // token 1 on frame 2. width=1 isolates softmax+normalize+mean.
        let weights = [
            [
                [1.0, 0, 0],
                [0.0, 0, 1],
            ]
        ]
        let m = WordAlignMath.reduceWeights(weights, medfiltWidth: 1)
        XCTAssertEqual(m.count, 2)
        XCTAssertEqual(m[0].count, 3)
        // Frame 0: token 0 above mean (+), token 1 below (−); symmetric.
        XCTAssertGreaterThan(m[0][0], 0)
        XCTAssertLessThan(m[1][0], 0)
        XCTAssertEqual(m[0][0], -m[1][0], accuracy: 1e-9)
        // Frame 2 mirrors frame 0.
        XCTAssertLessThan(m[0][2], 0)
        XCTAssertGreaterThan(m[1][2], 0)
        // Frame 1 is constant across tokens ⇒ normalizes to ~0.
        XCTAssertEqual(m[0][1], 0, accuracy: 1e-6)
        XCTAssertEqual(m[1][1], 0, accuracy: 1e-6)
    }

    func testMergeWordsByLeadingSpaceAndPunctuation() {
        // " qui" + "ck" → one word; "," and " world" start new words.
        let subwords: [(text: String, range: Range<Int>)] = [
            (" qui", 0 ..< 1),
            ("ck", 1 ..< 2),
            (",", 2 ..< 3),
            (" world", 3 ..< 4),
        ]
        let words = WordAlignMath.mergeWords(subwords)
        XCTAssertEqual(words.map(\.word), [" quick", ",", " world"])
        XCTAssertEqual(words[0].range, 0 ..< 2)
        XCTAssertEqual(words[1].range, 2 ..< 3)
        XCTAssertEqual(words[2].range, 3 ..< 4)
    }

    func testMergeWordsFirstSubwordWithoutSpaceStillStartsWord() {
        let subwords: [(text: String, range: Range<Int>)] = [
            ("Hel", 0 ..< 1), ("lo", 1 ..< 2),
        ]
        let words = WordAlignMath.mergeWords(subwords)
        XCTAssertEqual(words.map(\.word), ["Hello"])
        XCTAssertEqual(words[0].range, 0 ..< 2)
    }
}
