import XCTest

@testable import AthenaLLM

/// M31.3 — the OpenAI `stop` truncation helper. Covers one-shot (sync
/// path) and streaming (SSE path, with a match split across chunks).
final class StopStreamFilterTests: XCTestCase {

    func testTruncateOneShotCutsAtFirstStop() {
        let r = StopStreamFilter.truncate(
            "hello world STOP and more", stops: ["STOP"])
        XCTAssertTrue(r.stopped)
        XCTAssertEqual(r.text, "hello world ")
    }

    func testTruncateOneShotEarliestOfManyWins() {
        let r = StopStreamFilter.truncate(
            "abcXdefYghi", stops: ["Y", "X"])
        XCTAssertTrue(r.stopped)
        XCTAssertEqual(r.text, "abc", "earliest match (X) cuts first")
    }

    func testTruncateOneShotNoMatchPassesThrough() {
        let r = StopStreamFilter.truncate("nothing here", stops: ["ZZ"])
        XCTAssertFalse(r.stopped)
        XCTAssertEqual(r.text, "nothing here")
    }

    func testStreamingMatchWithinOneChunk() {
        var f = StopStreamFilter(stops: ["END"])
        var out = ""
        out += f.push("keep this END drop this")
        XCTAssertTrue(f.stopped)
        out += f.push(" also dropped")
        out += f.flush()
        XCTAssertEqual(out, "keep this ")
    }

    /// A stop sequence split across two chunks must still be caught — the
    /// filter holds back a tail that could be a stop prefix.
    func testStreamingMatchSplitAcrossChunks() {
        var f = StopStreamFilter(stops: ["STOP"])
        var out = ""
        out += f.push("alpha ST")      // "ST" held back (prefix of STOP)
        out += f.push("OP omega")      // completes STOP ⇒ cut
        out += f.flush()
        XCTAssertTrue(f.stopped)
        XCTAssertEqual(out, "alpha ")
    }

    func testStreamingNoStopEmitsEverything() {
        var f = StopStreamFilter(stops: ["END"])
        var out = ""
        out += f.push("one ")
        out += f.push("two ")
        out += f.push("three")
        out += f.flush()
        XCTAssertFalse(f.stopped)
        XCTAssertEqual(out, "one two three")
    }

    /// No active stops ⇒ a transparent pass-through (the common case).
    func testInactiveFilterIsPassThrough() {
        var f = StopStreamFilter(stops: [])
        XCTAssertFalse(f.isActive)
        XCTAssertEqual(f.push("anything goes"), "anything goes")
        XCTAssertEqual(f.flush(), "")
    }

    /// C13: a ZWJ family emoji is ONE grapheme but FIVE unicode scalars.
    /// A grapheme-count hold-back (maxLen=1 ⇒ keep 0) would surface the
    /// first chunk and miss the stop; the scalar-count hold-back retains
    /// every partial scalar until the full stop can be matched.
    func testMultiScalarStopSplitAcrossChunks() {
        let stop = "👨‍👩‍👧"
        var f = StopStreamFilter(stops: [stop])
        var out = ""
        for scalar in stop.unicodeScalars {
            out += f.push(String(scalar))
        }
        out += f.flush()
        XCTAssertTrue(f.stopped, "multi-scalar stop matched across chunks")
        XCTAssertEqual(out, "", "stop suppressed; nothing surfaced")
    }
}
