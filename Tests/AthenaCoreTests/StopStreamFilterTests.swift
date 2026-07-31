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
        out += f.push("alpha ST")  // "ST" held back (prefix of STOP)
        out += f.push("OP omega")  // completes STOP ⇒ cut
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

    // MARK: - M70.3 L6 — false-prefix release, streaming multi-stop, 1-char split

    /// A held-back tail that turns out NOT to complete a stop must be RELEASED,
    /// not swallowed: when no stop ever matches, the concatenated output equals
    /// the full input exactly (the " ST" held after the first chunk flows out
    /// once "STX" proves it can't become "STOP").
    func testStreamingFalsePrefixIsReleased() {
        var f = StopStreamFilter(stops: ["STOP"])
        var out = ""
        out += f.push("alpha ST")  // " ST" looks like a STOP prefix → held back
        out += f.push("X more")  // "STX" can't be STOP → the prefix releases
        out += f.flush()
        XCTAssertFalse(f.stopped, "no stop ever completed")
        XCTAssertEqual(
            out, "alpha STX more", "a false prefix is emitted, never dropped")
    }

    /// Streaming with several stops: the EARLIEST match in the reassembled
    /// stream wins, even when a later stop also appears.
    func testStreamingMultipleStopsEarliestWins() {
        var f = StopStreamFilter(stops: ["END", "X"])
        var out = ""
        out += f.push("aa")  // shorter than the hold-back ⇒ nothing yet
        out += f.push("bb X cc END")  // both X and END present; X is earlier
        out += f.flush()
        XCTAssertTrue(f.stopped)
        XCTAssertEqual(out, "aabb ", "cut at the earliest stop (X), not END")
    }

    /// WP7 — `matchedStop` names the sequence that actually latched (the
    /// earliest-position one), so Anthropic `stop_sequence` reporting is truthful
    /// rather than a `stops.first` guess.
    func testMatchedStopReportsTheActualSequence() {
        var f = StopStreamFilter(stops: ["END", "X"])
        XCTAssertNil(f.matchedStop, "no match yet")
        _ = f.push("aa")
        _ = f.push("bb X cc END")  // X matches first by position
        XCTAssertTrue(f.stopped)
        XCTAssertEqual(
            f.matchedStop, "X",
            "must report the matched sequence, not stops.first (END)")
    }

    /// Fed one character at a time, a multi-char stop straddling many pushes is
    /// still caught and the prefix before it surfaces exactly once.
    func testStreamingOneCharAtATime() {
        var f = StopStreamFilter(stops: ["STOP"])
        var out = ""
        for ch in "abSTOPcd" {
            out += f.push(String(ch))
        }
        out += f.flush()
        XCTAssertTrue(f.stopped)
        XCTAssertEqual(out, "ab", "everything before STOP, nothing after")
    }
}
