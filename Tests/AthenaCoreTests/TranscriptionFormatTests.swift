import Foundation
import XCTest

@testable import AthenaTranscription

/// SRT/VTT formatting — pure, CI-safe. Timecode precision and structure
/// must be exact (players are strict).
final class TranscriptionFormatTests: XCTestCase {

    private let segs = [
        TranscriptionSegment(start: 0, end: 2.5, text: " hello "),
        TranscriptionSegment(
            start: 2.5, end: 3661.004, text: "world"),
    ]

    func testSRT() {
        let s = TranscriptionFormat.srt(segs)
        XCTAssertEqual(
            s,
            "1\n00:00:00,000 --> 00:00:02,500\nhello\n\n"
                + "2\n00:00:02,500 --> 01:01:01,004\nworld\n")
    }

    func testVTT() {
        let v = TranscriptionFormat.vtt(segs)
        XCTAssertTrue(v.hasPrefix("WEBVTT\n\n"))
        XCTAssertTrue(
            v.contains("00:00:00.000 --> 00:00:02.500\nhello\n"))
        XCTAssertTrue(
            v.contains("00:00:02.500 --> 01:01:01.004\nworld\n"))
    }

    func testEmptySegments() {
        XCTAssertEqual(TranscriptionFormat.srt([]), "")
        XCTAssertEqual(TranscriptionFormat.vtt([]), "WEBVTT\n\n")
    }

    /// D12: an inverted span (start > end) must not produce a backwards cue;
    /// the end is clamped up to the start in both subtitle formats.
    func testInvertedSegmentClampsEnd() {
        let seg = [TranscriptionSegment(start: 3.0, end: 1.0, text: "x")]
        XCTAssertTrue(
            TranscriptionFormat.srt(seg).contains(
                "00:00:03,000 --> 00:00:03,000"),
            TranscriptionFormat.srt(seg))
        XCTAssertTrue(
            TranscriptionFormat.vtt(seg).contains(
                "00:00:03.000 --> 00:00:03.000"),
            TranscriptionFormat.vtt(seg))
    }
}
