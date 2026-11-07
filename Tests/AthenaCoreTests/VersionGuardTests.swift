import AthenaDeploy
import XCTest

final class VersionGuardTests: XCTestCase {
    func testCompareIsNumericNotLexical() {
        XCTAssertEqual(VersionGuard.compare("0.10.23", "0.10.23"), 0)
        // 9 < 10 numerically (lexical "9" > "10" would be wrong).
        XCTAssertLessThan(VersionGuard.compare("0.9.99", "0.10.0"), 0)
        XCTAssertGreaterThan(
            VersionGuard.compare("0.10.24", "0.10.23"), 0)
        // Missing trailing component counts as 0.
        XCTAssertEqual(VersionGuard.compare("1.0", "1.0.0"), 0)
        XCTAssertLessThan(VersionGuard.compare("1.0", "1.0.1"), 0)
    }

    func testPrereleaseChainOrdersBelowFinal() {
        let chain = ["0.12.0-dev.1", "0.12.0-dev.2", "0.12.0-rc.1", "0.12.0"]
        for (lo, hi) in zip(chain, chain.dropFirst()) {
            XCTAssertLessThan(VersionGuard.compare(lo, hi), 0, "\(lo) < \(hi)")
            XCTAssertGreaterThan(VersionGuard.compare(hi, lo), 0, "\(hi) > \(lo)")
        }
    }

    func testGluedPrereleaseIdentifiers() {
        XCTAssertLessThan(VersionGuard.compare("0.12.0-dev1", "0.12.0"), 0)
        XCTAssertLessThan(VersionGuard.compare("0.12.0-dev1", "0.12.0-dev2"), 0)
        XCTAssertGreaterThan(VersionGuard.compare("0.12.0-dev1", "0.12.0-dev.9"), 0)
    }

    func testPrereleaseOfNextCoreAboveOlderRelease() {
        XCTAssertLessThan(VersionGuard.compare("0.11.0", "0.12.0-dev.1"), 0)
    }

    func testPrereleaseNumericPartsCompareAsNumbers() {
        XCTAssertLessThan(VersionGuard.compare("0.12.0-rc.2", "0.12.0-rc.10"), 0)
        XCTAssertLessThan(VersionGuard.compare("0.12.0-rc", "0.12.0-rc.1"), 0)
    }

    func testEqualStringsReinstall() {
        XCTAssertEqual(VersionGuard.compare("0.12.0-dev.1", "0.12.0-dev.1"), 0)
        XCTAssertEqual(
            VersionGuard.classify(from: "0.12.0-dev.1", to: "0.12.0-dev.1"),
            .reinstall)
        XCTAssertEqual(
            VersionGuard.classify(from: "0.12.0-dev.2", to: "0.12.0"), .upgrade)
    }

    func testClassifyTransitions() {
        XCTAssertEqual(
            VersionGuard.classify(from: nil, to: "0.10.24"), .fresh)
        XCTAssertEqual(
            VersionGuard.classify(from: "", to: "0.10.24"), .fresh)
        XCTAssertEqual(
            VersionGuard.classify(from: "0.10.24", to: "0.10.24"),
            .reinstall)
        XCTAssertEqual(
            VersionGuard.classify(from: "0.10.23", to: "0.10.24"),
            .upgrade)
        XCTAssertEqual(
            VersionGuard.classify(from: "0.10.24", to: "0.10.23"),
            .downgrade)
        // 0.9.100 -> 0.10.0 is an UPGRADE (numeric compare).
        XCTAssertEqual(
            VersionGuard.classify(from: "0.9.100", to: "0.10.0"),
            .upgrade)
    }

    func testSummaryStrings() {
        XCTAssertTrue(
            VersionGuard.summary(from: nil, to: "0.10.24")
                .contains("fresh install"))
        XCTAssertTrue(
            VersionGuard.summary(from: "0.10.23", to: "0.10.24")
                .contains("upgrading"))
        XCTAssertTrue(
            VersionGuard.summary(from: "0.10.25", to: "0.10.24")
                .contains("DOWNGRADING"))
    }
}
