import Foundation
import XCTest

@testable import AthenaCore

/// ADR 041 A1 — the pure budget-period arithmetic. Every case pins an explicit
/// calendar/time zone so the boundaries are deterministic in CI regardless of
/// the machine's locale (the production default is `.current`, i.e. LOCAL
/// boundaries, which is the point — an operator reads their own clock).
final class QuotaWindowTests: XCTestCase {

    /// A DST-observing zone with a well-known transition history.
    private func cal(_ tz: String = "America/Chicago") -> Calendar {
        var c = Calendar(identifier: .gregorian)
        c.timeZone = TimeZone(identifier: tz)!
        return c
    }

    private func epoch(
        _ y: Int, _ mo: Int, _ d: Int, _ h: Int = 0, _ mi: Int = 0,
        tz: String = "America/Chicago"
    ) -> Double {
        var c = DateComponents()
        c.year = y
        c.month = mo
        c.day = d
        c.hour = h
        c.minute = mi
        return cal(tz).date(from: c)!.timeIntervalSince1970
    }

    // MARK: parse

    func testParseDefaultsToMonth() {
        XCTAssertEqual(QuotaWindow.parse(nil), .month)
        XCTAssertEqual(QuotaWindow.parse(""), .month)
        XCTAssertEqual(QuotaWindow.parse("   "), .month)
    }

    func testParseAcceptsBothWindowsCaseInsensitively() {
        XCTAssertEqual(QuotaWindow.parse("day"), .day)
        XCTAssertEqual(QuotaWindow.parse("DAY"), .day)
        XCTAssertEqual(QuotaWindow.parse(" Month "), .month)
    }

    /// An unrecognized value must NOT silently become a default — config
    /// parsing fails loudly on nil (ADR 041 §2).
    func testParseRejectsUnknownWindow() {
        XCTAssertNil(QuotaWindow.parse("hour"))
        XCTAssertNil(QuotaWindow.parse("week"))
        XCTAssertNil(QuotaWindow.parse("30d"))
    }

    // MARK: periodStart

    func testDayPeriodStartIsLocalMidnight() {
        let t = epoch(2026, 7, 25, 14, 37)
        XCTAssertEqual(
            QuotaWindow.day.periodStart(containing: t, calendar: cal()),
            epoch(2026, 7, 25, 0, 0))
    }

    func testMonthPeriodStartIsFirstOfLocalMonth() {
        let t = epoch(2026, 7, 25, 14, 37)
        XCTAssertEqual(
            QuotaWindow.month.periodStart(containing: t, calendar: cal()),
            epoch(2026, 7, 1, 0, 0))
    }

    /// Midnight itself belongs to the period it starts (a half-open interval),
    /// so a request at exactly the boundary counts against the NEW period.
    func testBoundaryInstantBelongsToNewPeriod() {
        let midnight = epoch(2026, 7, 25, 0, 0)
        XCTAssertEqual(
            QuotaWindow.day.periodStart(containing: midnight, calendar: cal()),
            midnight)
        let monthStart = epoch(2026, 7, 1, 0, 0)
        XCTAssertEqual(
            QuotaWindow.month.periodStart(
                containing: monthStart, calendar: cal()),
            monthStart)
    }

    // MARK: nextRoll / secondsUntilRoll

    /// The spring-forward day is 23 hours long. Calendar arithmetic must give
    /// the next local midnight, not "+86400" (which would land at 1am and
    /// double-count an hour of budget).
    func testDayRollAcrossSpringForwardIs23Hours() {
        let t = epoch(2026, 3, 8, 1, 0)  // US DST begins 2026-03-08
        let start = QuotaWindow.day.periodStart(containing: t, calendar: cal())
        let roll = QuotaWindow.day.nextRoll(after: t, calendar: cal())
        XCTAssertEqual(roll, epoch(2026, 3, 9, 0, 0))
        XCTAssertEqual(roll - start, 23 * 3600)
    }

    /// The fall-back day is 25 hours long — the same arithmetic, the other
    /// direction.
    func testDayRollAcrossFallBackIs25Hours() {
        let t = epoch(2026, 11, 1, 5, 0)  // US DST ends 2026-11-01
        let start = QuotaWindow.day.periodStart(containing: t, calendar: cal())
        let roll = QuotaWindow.day.nextRoll(after: t, calendar: cal())
        XCTAssertEqual(roll, epoch(2026, 11, 2, 0, 0))
        XCTAssertEqual(roll - start, 25 * 3600)
    }

    /// Month-end must not be "+30 days": February is 28, and a request on the
    /// 31st must roll to the 1st of the next month.
    func testMonthRollHandlesShortAndLongMonths() {
        let feb = epoch(2026, 2, 14, 12, 0)
        XCTAssertEqual(
            QuotaWindow.month.nextRoll(after: feb, calendar: cal()),
            epoch(2026, 3, 1, 0, 0))
        let jan31 = epoch(2026, 1, 31, 23, 59)
        XCTAssertEqual(
            QuotaWindow.month.nextRoll(after: jan31, calendar: cal()),
            epoch(2026, 2, 1, 0, 0))
        let dec = epoch(2026, 12, 20, 8, 0)
        XCTAssertEqual(
            QuotaWindow.month.nextRoll(after: dec, calendar: cal()),
            epoch(2027, 1, 1, 0, 0))
    }

    func testSecondsUntilRollIsPositiveAndRoundsUp() {
        let t = epoch(2026, 7, 25, 23, 59) + 0.5
        let secs = QuotaWindow.day.secondsUntilRoll(from: t, calendar: cal())
        XCTAssertEqual(secs, 60)
        // Even standing exactly on the boundary, Retry-After is never 0.
        let atBoundary = QuotaWindow.day.secondsUntilRoll(
            from: epoch(2026, 7, 25, 0, 0), calendar: cal())
        XCTAssertGreaterThan(atBoundary, 0)
    }

    /// A UTC-only box must behave the same way — the boundaries just move.
    func testWindowsWorkInAFixedOffsetZone() {
        let c = cal("UTC")
        let t = epoch(2026, 7, 25, 14, 0, tz: "UTC")
        XCTAssertEqual(
            QuotaWindow.day.periodStart(containing: t, calendar: c),
            epoch(2026, 7, 25, 0, 0, tz: "UTC"))
        XCTAssertEqual(
            QuotaWindow.day.nextRoll(after: t, calendar: c) - t, 10 * 3600)
    }

    // MARK: periodTokens — the lazy reset

    func testPeriodTokensSumWithinTheCurrentPeriod() {
        let start = epoch(2026, 7, 1)
        XCTAssertEqual(
            QuotaWindow.periodTokens(
                storedPeriodStart: start, promptTokens: 900,
                completionTokens: 100, currentPeriodStart: start),
            1000)
    }

    /// The whole reset mechanism: a row from a period that has rolled reads as
    /// zero, with no job having run and no row having been written.
    func testPeriodTokensZeroWhenTheStoredPeriodHasRolled() {
        XCTAssertEqual(
            QuotaWindow.periodTokens(
                storedPeriodStart: epoch(2026, 6, 1), promptTokens: 5_000_000,
                completionTokens: 1_000_000,
                currentPeriodStart: epoch(2026, 7, 1)),
            0)
    }

    /// A pre-ADR-041 row (period_start defaults to 0) is a rolled period, so it
    /// starts every principal at zero rather than inheriting lifetime totals as
    /// period usage.
    func testLegacyZeroPeriodStartReadsAsZero() {
        XCTAssertEqual(
            QuotaWindow.periodTokens(
                storedPeriodStart: 0, promptTokens: 12_345,
                completionTokens: 678, currentPeriodStart: epoch(2026, 7, 1)),
            0)
    }

    /// Clock skew backwards (a stored start NEWER than the current period)
    /// must not read as zero and silently hand out a fresh budget.
    func testFuturePeriodStartStillCounts() {
        XCTAssertEqual(
            QuotaWindow.periodTokens(
                storedPeriodStart: epoch(2026, 8, 1), promptTokens: 10,
                completionTokens: 5, currentPeriodStart: epoch(2026, 7, 1)),
            15)
    }
}
