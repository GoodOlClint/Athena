import XCTest

@testable import AthenaCore

/// G1 (ADR 023) — the serve-path MLX cache-limit resolution is MLX-free, so it
/// runs in CI (ADR 009). Pins the default-fraction / explicit-override /
/// 0-means-unbounded rules.
final class GovernorMemoryTests: XCTestCase {
    private let budget = 96 * 1024 * 1024 * 1024  // 96 GiB

    func testDefaultIsBudgetFraction() {
        XCTAssertEqual(
            GovernorMemory.resolveCacheLimit(configured: nil, budgetBytes: budget),
            budget / 3)
    }

    func testCustomFractionDenominator() {
        XCTAssertEqual(
            GovernorMemory.resolveCacheLimit(
                configured: nil, budgetBytes: budget, fractionDenominator: 4),
            budget / 4)
    }

    func testConfiguredValueWins() {
        let lim = 8 * 1024 * 1024 * 1024
        XCTAssertEqual(
            GovernorMemory.resolveCacheLimit(
                configured: lim, budgetBytes: budget),
            lim)
    }

    func testZeroMeansUnbounded() {
        XCTAssertNil(
            GovernorMemory.resolveCacheLimit(configured: 0, budgetBytes: budget))
    }

    func testNegativeMeansUnbounded() {
        XCTAssertNil(
            GovernorMemory.resolveCacheLimit(configured: -1, budgetBytes: budget))
    }

    func testNoBudgetNoConfigLeavesDefault() {
        XCTAssertNil(
            GovernorMemory.resolveCacheLimit(configured: nil, budgetBytes: 0))
    }

    func testConfiguredWinsEvenWithZeroBudget() {
        XCTAssertEqual(
            GovernorMemory.resolveCacheLimit(configured: 1234, budgetBytes: 0),
            1234)
    }
}
