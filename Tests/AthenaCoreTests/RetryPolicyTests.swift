import AthenaClient
import Foundation
import XCTest

#if canImport(FoundationNetworking)
    import FoundationNetworking
#endif

/// M19 — the retry policy is the safety boundary (must not silently
/// double-execute a non-idempotent request), so the adversarial
/// idempotency cases are explicit.
final class RetryPolicyTests: XCTestCase {
    private let p = RetryPolicy(maxRetries: 3)

    func testRejectedStatusesRetryForAnyMethod() {
        // 429 / 503 = rejected without executing ⇒ safe for POST too.
        for m in ["GET", "POST", "DELETE", "PUT"] {
            XCTAssertNotNil(
                p.delay(
                    attempt: 0, method: m, outcome: .status(429)))
            XCTAssertNotNil(
                p.delay(
                    attempt: 0, method: m, outcome: .status(503)))
        }
    }

    func testGatewayStatusesIdempotentOnly() {
        for s in [502, 504] {
            XCTAssertNotNil(
                p.delay(
                    attempt: 0, method: "GET",
                    outcome: .status(s)))
            // POST may have executed ⇒ MUST NOT retry.
            XCTAssertNil(
                p.delay(
                    attempt: 0, method: "POST",
                    outcome: .status(s)))
        }
    }

    func testNonRetryableStatuses() {
        for s in [200, 201, 301, 400, 401, 403, 404, 409, 500] {
            XCTAssertNil(
                p.delay(
                    attempt: 0, method: "GET",
                    outcome: .status(s)))
        }
    }

    func testTransportNeverReachedRetriesAnyMethod() {
        for c: URLError.Code in [
            .cannotConnectToHost, .cannotFindHost,
            .dnsLookupFailed, .notConnectedToInternet,
        ] {
            XCTAssertNotNil(
                p.delay(
                    attempt: 0, method: "POST",
                    outcome: .transport(c)))
        }
    }

    func testTransportAmbiguousIdempotentOnly() {
        for c: URLError.Code in [.timedOut, .networkConnectionLost] {
            XCTAssertNotNil(
                p.delay(
                    attempt: 0, method: "GET",
                    outcome: .transport(c)))
            XCTAssertNil(
                p.delay(
                    attempt: 0, method: "POST",
                    outcome: .transport(c)))
        }
    }

    func testMaxRetriesCapAndDisable() {
        // attempt index >= maxRetries ⇒ give up even if retryable.
        XCTAssertNil(
            p.delay(
                attempt: 3, method: "GET", outcome: .status(503)))
        XCTAssertNotNil(
            p.delay(
                attempt: 2, method: "GET", outcome: .status(503)))
        // disabled
        let off = RetryPolicy(maxRetries: 0)
        XCTAssertNil(
            off.delay(
                attempt: 0, method: "GET", outcome: .status(503)))
        // clamp upper bound
        XCTAssertEqual(RetryPolicy(maxRetries: 99).maxRetries, 10)
    }

    func testBackoffScheduleAndRetryAfter() {
        XCTAssertEqual(
            p.delay(
                attempt: 0, method: "GET", outcome: .status(503)),
            0.5)
        XCTAssertEqual(
            p.delay(
                attempt: 1, method: "GET", outcome: .status(503)),
            1.5)
        XCTAssertEqual(
            p.delay(
                attempt: 2, method: "GET", outcome: .status(503)),
            3.0)
        // Retry-After larger than base wins, but is capped.
        XCTAssertEqual(
            p.delay(
                attempt: 0, method: "GET", outcome: .status(503),
                retryAfter: 7),
            7)
        XCTAssertEqual(
            p.delay(
                attempt: 0, method: "GET", outcome: .status(503),
                retryAfter: 999),
            10)  // retryAfterCap
        // Smaller Retry-After ignored — keep the base backoff.
        XCTAssertEqual(
            p.delay(
                attempt: 1, method: "GET", outcome: .status(503),
                retryAfter: 0.1),
            1.5)
    }

    func testFromEnvironment() {
        XCTAssertEqual(
            RetryPolicy.fromEnvironment(
                ["ATHENA_HTTP_MAX_RETRIES": "0"]).maxRetries, 0)
        XCTAssertEqual(
            RetryPolicy.fromEnvironment(
                ["ATHENA_HTTP_MAX_RETRIES": "5"]).maxRetries, 5)
        XCTAssertEqual(
            RetryPolicy.fromEnvironment([:]).maxRetries, 3)
    }
}
