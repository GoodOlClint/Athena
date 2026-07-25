import Foundation
import HTTPTypes
import Hummingbird
import XCTest

@testable import AthenaServerKit

/// ADR 041 A3 — the pure budget verdict, the enforcement scope, and the
/// advisory-header contract (ADR 009: everything decidable without MLX or a
/// live daemon is pinned here; the 429-under-load path is the e2e's job).
final class QuotaDecisionTests: XCTestCase {

    // MARK: evaluate

    func testWithinBudgetReportsRemaining() {
        XCTAssertEqual(
            QuotaDecision.evaluate(used: 400, budget: 1000),
            .within(limit: 1000, remaining: 600))
    }

    /// Reaching the budget exactly is exhausted — the next request must spend
    /// at least one more token, so "remaining 0" cannot be admitted.
    func testExactlyAtBudgetIsExhausted() {
        XCTAssertTrue(
            QuotaDecision.evaluate(used: 1000, budget: 1000).isExhausted)
    }

    /// The documented overshoot (ADR 041 §3): one token left is ADMITTED, and
    /// the request that follows may exceed the budget. Pinning it so nobody
    /// "fixes" it into a pre-charge later without reading the ADR.
    func testOneTokenLeftIsAdmittedOvershootIsByDesign() {
        XCTAssertEqual(
            QuotaDecision.evaluate(used: 999, budget: 1000),
            .within(limit: 1000, remaining: 1))
    }

    func testOvershotUsageStaysExhaustedNotNegative() {
        XCTAssertEqual(
            QuotaDecision.evaluate(used: 5000, budget: 1000),
            .exhausted(limit: 1000))
    }

    /// nil (no budget configured, no override) and 0 (explicit unlimited) are
    /// both unlimited — and unlimited means NO headers, so an absent header can
    /// only ever mean "no cap".
    func testNilAndZeroBudgetAreUnlimited() {
        XCTAssertEqual(QuotaDecision.evaluate(used: 0, budget: nil), .unlimited)
        XCTAssertEqual(
            QuotaDecision.evaluate(used: 10_000_000, budget: 0), .unlimited)
        XCTAssertEqual(
            QuotaDecision.evaluate(used: 10, budget: -5), .unlimited)
    }

    func testNegativeUsageClampsRatherThanInflatingRemaining() {
        XCTAssertEqual(
            QuotaDecision.evaluate(used: -50, budget: 100),
            .within(limit: 100, remaining: 100))
    }

    // MARK: enforcement scope

    /// Token-bearing routes only (ADR 041 §3). The load-bearing exclusions:
    /// `/api/usage` (an exhausted principal must still be able to diagnose)
    /// and `count_tokens` (it exists to keep a client UNDER budget, ADR 042).
    func testMeteredScopeIsTokenBearingRoutesOnly() {
        typealias M = QuotaMiddleware<BasicRequestContext>
        XCTAssertTrue(M.metered("/v1/chat/completions"))
        XCTAssertTrue(M.metered("/v1/embeddings"))
        XCTAssertTrue(M.metered("/v1/messages"))

        XCTAssertFalse(M.metered("/v1/chat/completions/count_tokens"))
        XCTAssertFalse(M.metered("/api/usage"))
        XCTAssertFalse(M.metered("/api/models"))
        XCTAssertFalse(M.metered("/healthz"))
        XCTAssertFalse(M.metered("/metrics"))
        XCTAssertFalse(M.metered("/v1/models"))
        XCTAssertFalse(M.metered("/v1/audio/transcriptions"))
        XCTAssertFalse(M.metered("/ui"))
    }

    // MARK: wire shapes

    func testExhaustedResponseCarriesEnvelopeAndRetryAfter() {
        let resets = 1_754_006_400.0  // 2025-08-01T00:00:00Z
        let r = QuotaHeaders.exhaustedResponse(
            retryAfter: 3600, resets: resets)
        XCTAssertEqual(r.status, .tooManyRequests)
        XCTAssertEqual(r.headers[.retryAfter], "3600")
        XCTAssertEqual(
            r.headers[QuotaHeaders.reset],
            QuotaHeaders.iso8601(resets))
        // The canonical error envelope, with the quota-specific type/code so an
        // operator can tell exhaustion from the limiter's rate_limited 429.
        let body = QuotaHeaders.exhaustedBody(resets: resets)
        XCTAssertTrue(body.contains("\"type\":\"insufficient_quota\""))
        XCTAssertTrue(body.contains("\"code\":\"quota_exceeded\""))
        XCTAssertTrue(body.contains("resets"))
    }

    func testAdvisoryHeadersAttachedWhenBudgeted() {
        var r = Response(status: .ok)
        QuotaHeaders.attach(
            &r, decision: .within(limit: 1000, remaining: 250),
            resets: 1_754_006_400.0)
        XCTAssertEqual(r.headers[QuotaHeaders.limit], "1000")
        XCTAssertEqual(r.headers[QuotaHeaders.remaining], "250")
        XCTAssertEqual(
            r.headers[QuotaHeaders.reset], "2025-08-01T00:00:00Z")
    }

    /// Unlimited ⇒ NOT "remaining: 0", but no headers at all.
    func testNoAdvisoryHeadersWhenUnlimited() {
        var r = Response(status: .ok)
        QuotaHeaders.attach(
            &r, decision: .unlimited, resets: 1_754_006_400.0)
        XCTAssertNil(r.headers[QuotaHeaders.limit])
        XCTAssertNil(r.headers[QuotaHeaders.remaining])
        XCTAssertNil(r.headers[QuotaHeaders.reset])
    }

    func testExhaustedAdvisoryReportsZeroRemaining() {
        var r = Response(status: .ok)
        QuotaHeaders.attach(
            &r, decision: .exhausted(limit: 100),
            resets: 1_754_006_400.0)
        XCTAssertEqual(r.headers[QuotaHeaders.limit], "100")
        XCTAssertEqual(r.headers[QuotaHeaders.remaining], "0")
    }

    func testResetIsUTCISO8601() {
        XCTAssertEqual(
            QuotaHeaders.iso8601(1_754_006_400.0), "2025-08-01T00:00:00Z")
    }
}
