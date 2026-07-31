import AthenaDeploy
import Foundation
import XCTest

/// NB4 (M70.1b) — `parseTTLSeconds` and `isValidLabel` were unreachable by the
/// test suite (free functions in the `athena` executable). Relocated to the
/// MLX-free `AthenaDeploy` and pinned here.
final class CLIParseTests: XCTestCase {

    // MARK: - parseTTLSeconds

    func testTTLUnits() {
        XCTAssertEqual(parseTTLSeconds("30s"), 30)
        XCTAssertEqual(parseTTLSeconds("90m"), 90 * 60)
        XCTAssertEqual(parseTTLSeconds("12h"), 12 * 3600)
        XCTAssertEqual(parseTTLSeconds("3d"), 3 * 86400)
    }

    func testTTLBareIntegerIsSeconds() {
        XCTAssertEqual(parseTTLSeconds("3600"), 3600)
    }

    func testTTLTrimsWhitespace() {
        XCTAssertEqual(parseTTLSeconds("  7d "), 7 * 86400)
    }

    func testTTLRejectsMalformedAndNonPositive() {
        XCTAssertNil(parseTTLSeconds(""))
        XCTAssertNil(parseTTLSeconds("   "))
        XCTAssertNil(parseTTLSeconds("abc"))
        XCTAssertNil(parseTTLSeconds("10x"))
        XCTAssertNil(parseTTLSeconds("0"), "non-positive rejected")
        XCTAssertNil(parseTTLSeconds("-5"), "negative rejected")
        XCTAssertNil(parseTTLSeconds("0d"))
    }

    func testTTLRejectsOverflow() {
        // A day-count that overflows when multiplied by 86400.
        XCTAssertNil(parseTTLSeconds("\(Int.max)d"))
    }

    // MARK: - isValidLabel

    func testValidLabels() {
        XCTAssertTrue(isValidLabel("com.athena.daemon"))
        XCTAssertTrue(isValidLabel("athena-load_1"))
        XCTAssertTrue(isValidLabel("A"))
    }

    func testInvalidLabels() {
        XCTAssertFalse(isValidLabel(""), "empty rejected")
        XCTAssertFalse(isValidLabel("has space"))
        XCTAssertFalse(isValidLabel("semi;colon"))
        XCTAssertFalse(isValidLabel("slash/here"))
        XCTAssertFalse(isValidLabel("new\nline"))
        XCTAssertFalse(
            isValidLabel(String(repeating: "a", count: 256)),
            "over 255 chars rejected")
        XCTAssertTrue(
            isValidLabel(String(repeating: "a", count: 255)),
            "exactly 255 allowed")
    }
}
