import Foundation

// ADR 041 — per-principal token budgets over a rolling period. This file is
// the pure arithmetic half (ADR 009: MLX-free, unit-pinned): mapping a
// timestamp to its period, and reading a stored period counter that may belong
// to a period which has already rolled.
//
// There is no timer and no background job. A period "resets" because a stored
// `period_start` older than the current period is read as zero and overwritten
// on the next `addUsage` — the arithmetic IS the state machine.

/// The budget period. Boundaries are LOCAL (the operator reads their own
/// clock, not UTC), computed through `Calendar` so DST transitions and
/// short/long months are the calendar's problem, not ours.
public enum QuotaWindow: String, Sendable, CaseIterable, Codable {
    case day
    case month

    /// Parse the `token_budget_window` config value. nil/empty ⇒ the default
    /// (`month`). An unrecognized value returns nil so config parsing can fail
    /// loudly — silently falling back would hand the operator a budget with a
    /// window they did not ask for.
    public static func parse(_ raw: String?) -> QuotaWindow? {
        guard let raw, !raw.trimmingCharacters(in: .whitespaces).isEmpty
        else { return .month }
        return QuotaWindow(
            rawValue: raw.trimmingCharacters(in: .whitespaces).lowercased())
    }

    /// Start of the period containing `time` (epoch seconds): local midnight
    /// for `.day`, the first instant of the local month for `.month`.
    public func periodStart(
        containing time: Double, calendar: Calendar = .current
    ) -> Double {
        let date = Date(timeIntervalSince1970: time)
        switch self {
        case .day:
            return calendar.startOfDay(for: date).timeIntervalSince1970
        case .month:
            let comps = calendar.dateComponents([.year, .month], from: date)
            // A missing month start is not reachable for a real date; fall back
            // to the day boundary rather than inventing an epoch.
            return (calendar.date(from: comps)
                ?? calendar.startOfDay(for: date)).timeIntervalSince1970
        }
    }

    /// When the period containing `time` ends — i.e. the next period's start,
    /// which is what a client's `Retry-After` / `x-athena-tokens-reset` must
    /// report. Calendar arithmetic, so a 23- or 25-hour DST day and a 28- or
    /// 31-day month are handled by the calendar.
    public func nextRoll(after time: Double, calendar: Calendar = .current)
        -> Double
    {
        let start = Date(
            timeIntervalSince1970: periodStart(
                containing: time, calendar: calendar))
        let component: Calendar.Component = self == .day ? .day : .month
        guard
            let next = calendar.date(
                byAdding: component, value: 1, to: start)
        else {
            // Unreachable for a real date; a fixed 24h nudge beats returning a
            // roll in the past (which would make Retry-After zero forever).
            return start.timeIntervalSince1970 + 86_400
        }
        return next.timeIntervalSince1970
    }

    /// Seconds until the current period rolls, rounded up, floor 1 — the
    /// `Retry-After` value for an exhausted principal.
    public func secondsUntilRoll(
        from time: Double, calendar: Calendar = .current
    ) -> Int {
        let secs = nextRoll(after: time, calendar: calendar) - time
        return max(1, Int(secs.rounded(.up)))
    }

    /// Tokens a principal has spent in the CURRENT period, given the stored
    /// row's `period_start` and counters. A row whose period has already
    /// rolled reads as **zero** — the stored counters describe a period that
    /// is over, and `addUsage` overwrites them on the next request. This is
    /// the lazy reset; there is nothing to run at the boundary.
    public static func periodTokens(
        storedPeriodStart: Double, promptTokens: Int, completionTokens: Int,
        currentPeriodStart: Double
    ) -> Int {
        guard storedPeriodStart >= currentPeriodStart else { return 0 }
        return promptTokens + completionTokens
    }
}
