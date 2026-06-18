import Foundation
import XCTest

@testable import AthenaTranscription

/// Pure (MLX-free) TDT timestamp grouping (ADR 020 S3). Always runs in CI
/// (ADR 008/009). Pins word/sentence grouping, monotonic timing, per-segment
/// word attachment, and the sentence-end rule.
final class ParakeetAlignmentTests: XCTestCase {
    private typealias Token = ParakeetAlignment.Token

    // "Hello world. Bye" — leading space = SentencePiece ▁ word boundary.
    private let sample: [Token] = [
        Token(id: 1, text: " Hello", start: 0.0, duration: 0.16, confidence: 0.9),
        Token(id: 2, text: " world", start: 0.2, duration: 0.16, confidence: 0.8),
        Token(id: 3, text: ".", start: 0.4, duration: 0.08, confidence: 0.95),
        Token(id: 4, text: " Bye", start: 0.6, duration: 0.16, confidence: 0.7),
    ]

    func testWordsGroupedAtBoundaries() {
        let w = ParakeetAlignment.words(from: sample)
        XCTAssertEqual(w.map(\.word), ["Hello", "world.", "Bye"])
        XCTAssertEqual(w[0].start, 0.0, accuracy: 1e-9)
        XCTAssertEqual(w[0].end, 0.16, accuracy: 1e-9)
        XCTAssertEqual(w[1].start, 0.2, accuracy: 1e-9)
        XCTAssertEqual(w[1].end, 0.48, accuracy: 1e-9)  // world(0.36) + "."(0.48)
        XCTAssertEqual(w[2].start, 0.6, accuracy: 1e-9)
        // probability = mean token confidence over the word's pieces.
        XCTAssertEqual(w[1].probability, (0.8 + 0.95) / 2, accuracy: 1e-9)
    }

    func testSentenceSegments() {
        let segs = ParakeetAlignment.segments(from: sample, attachWords: true)
        XCTAssertEqual(segs.count, 2)
        XCTAssertEqual(segs[0].text, "Hello world.")
        XCTAssertEqual(segs[1].text, "Bye")
        XCTAssertEqual(segs[0].start, 0.0, accuracy: 1e-9)
        XCTAssertEqual(segs[0].end, 0.48, accuracy: 1e-9)
        XCTAssertEqual(segs[1].start, 0.6, accuracy: 1e-9)
    }

    func testTimestampsMonotonicAndOrdered() {
        let segs = ParakeetAlignment.segments(from: sample, attachWords: false)
        for s in segs { XCTAssertLessThanOrEqual(s.start, s.end) }
        for i in 1..<segs.count {
            XCTAssertLessThanOrEqual(segs[i - 1].start, segs[i].start)
        }
    }

    func testPerSegmentWordsWithinBounds() {
        let segs = ParakeetAlignment.segments(from: sample, attachWords: true)
        XCTAssertEqual(segs[0].words?.count, 2)  // Hello, world.
        XCTAssertEqual(segs[1].words?.count, 1)  // Bye
        for s in segs {
            for w in s.words ?? [] {
                XCTAssertGreaterThanOrEqual(w.start, s.start)
                XCTAssertLessThanOrEqual(w.start, s.end)
            }
        }
    }

    func testAvgLogprobIsNegativeLogOfConfidence() {
        let segs = ParakeetAlignment.segments(from: sample, attachWords: false)
        let lp = try? XCTUnwrap(segs[0].avgLogprob)
        XCTAssertNotNil(lp)
        // mean(ln 0.9, ln 0.8, ln 0.95) < 0
        XCTAssertLessThan(segs[0].avgLogprob!, 0)
        let expected =
            (log(0.9) + log(0.8) + log(0.95)) / 3
        XCTAssertEqual(segs[0].avgLogprob!, expected, accuracy: 1e-9)
    }

    func testSentenceEndRule() {
        XCTAssertTrue(ParakeetAlignment.isSentenceEnd("?", nextStartsWord: false))
        XCTAssertTrue(ParakeetAlignment.isSentenceEnd("!", nextStartsWord: false))
        XCTAssertTrue(ParakeetAlignment.isSentenceEnd("。", nextStartsWord: false))
        // Period only ends a sentence at a word boundary (or end of stream).
        XCTAssertTrue(ParakeetAlignment.isSentenceEnd(".", nextStartsWord: true))
        XCTAssertTrue(ParakeetAlignment.isSentenceEnd(".", nextStartsWord: nil))
        XCTAssertFalse(
            ParakeetAlignment.isSentenceEnd(".", nextStartsWord: false))
        XCTAssertFalse(
            ParakeetAlignment.isSentenceEnd("word", nextStartsWord: true))
    }

    func testEmptyTokens() {
        XCTAssertEqual(ParakeetAlignment.words(from: []).count, 0)
        XCTAssertEqual(
            ParakeetAlignment.segments(from: [], attachWords: true).count, 0)
    }
}
