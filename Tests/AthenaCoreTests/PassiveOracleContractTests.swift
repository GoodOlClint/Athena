import Foundation
import XCTest

@testable import AthenaCore

/// The passive-oracle contract is Athena's security model, so it is
/// asserted in code rather than left to documentation. These guard the
/// architectural defaults; later milestones extend the suite (no on-disk
/// persistence of request inputs, no Crete-initiated connections except
/// one-way Clio log shipping).
final class PassiveOracleContractTests: XCTestCase {

    func testOwnPortNotOllama() {
        XCTAssertEqual(GovernorConfig.defaultPort, 7447)
        XCTAssertEqual(GovernorConfig().listenPort, 7447)
        XCTAssertNotEqual(GovernorConfig().listenPort, 11434)
    }

    func testLoopbackByDefault() {
        XCTAssertEqual(GovernorConfig().listenHost, "127.0.0.1")
    }

    func testSingleBoundedBudget() {
        let cfg = GovernorConfig()
        let physical = Int(ProcessInfo.processInfo.physicalMemory)
        XCTAssertGreaterThan(cfg.totalBudgetBytes, 0)
        XCTAssertLessThan(cfg.totalBudgetBytes, physical)
    }

    func testExplicitBudgetHonored() {
        XCTAssertEqual(
            GovernorConfig(totalBudgetBytes: 12_345).totalBudgetBytes, 12_345)
    }
}
