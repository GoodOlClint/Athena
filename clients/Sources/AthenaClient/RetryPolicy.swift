import Foundation

#if canImport(FoundationNetworking)
    import FoundationNetworking
#endif

/// Bounded exponential-backoff retry policy for the thin client's
/// HTTP calls (M19, adapted from the the consuming application LLM broker's
/// request-retry pattern). Pure + `Sendable` so the safety rules are
/// unit-tested in isolation.
///
/// Idempotency is the safety invariant: a non-idempotent request
/// (POST/PUT/DELETE — e.g. queue submit, vector upsert) is retried
/// ONLY when it provably did not execute. `429`/`503` mean the
/// governed daemon REJECTED the request without doing work (M5
/// Metal-OOM/backpressure → 503; rate-limit → 429), so those are
/// safe to retry for any method. `502`/`504` and ambiguous transport
/// failures (timed out / connection lost mid-flight) might have
/// executed, so they are retried for idempotent methods only.
public struct RetryPolicy: Sendable {
    public let maxRetries: Int

    /// Backoff seconds before attempt N+1 (clamped to the last entry).
    static let backoff: [TimeInterval] = [0.5, 1.5, 3.0]
    /// Never honor an absurd `Retry-After` — a CLI must not hang.
    static let retryAfterCap: TimeInterval = 10
    /// Safe to retry even if the request may have executed.
    static let idempotent: Set<String> = ["GET", "HEAD"]

    public init(maxRetries: Int = 3) {
        self.maxRetries = max(0, min(maxRetries, 10))
    }

    /// `ATHENA_HTTP_MAX_RETRIES` overrides the default (0 disables).
    public static func fromEnvironment(
        _ env: [String: String] =
            ProcessInfo.processInfo.environment
    ) -> RetryPolicy {
        if let v = env["ATHENA_HTTP_MAX_RETRIES"], let n = Int(v) {
            return RetryPolicy(maxRetries: n)
        }
        return RetryPolicy()
    }

    public enum Outcome: Sendable {
        case status(Int)
        case transport(URLError.Code)
    }

    /// Seconds to wait before the next attempt, or nil = give up
    /// (exhausted or not retryable). `attempt` is 0-based: 0 = the
    /// decision after the 1st try, before a 2nd.
    public func delay(
        attempt: Int, method: String, outcome: Outcome,
        retryAfter: TimeInterval? = nil
    ) -> TimeInterval? {
        guard attempt < maxRetries else { return nil }
        let safe = Self.idempotent.contains(method.uppercased())
        let retryable: Bool
        switch outcome {
        case .status(let s):
            switch s {
            case 429, 503: retryable = true  // rejected, not run
            case 502, 504: retryable = safe  // maybe ran ⇒ GET only
            default: retryable = false  // 2xx/3xx/4xx/500: never
            }
        case .transport(let c):
            switch c {
            case .cannotConnectToHost, .cannotFindHost,
                .dnsLookupFailed, .notConnectedToInternet:
                retryable = true  // provably never reached server
            case .timedOut, .networkConnectionLost:
                retryable = safe  // ambiguous ⇒ idempotent only
            default:
                retryable = false
            }
        }
        guard retryable else { return nil }
        let base = Self.backoff[
            min(attempt, Self.backoff.count - 1)]
        if let ra = retryAfter, ra > base {
            return min(ra, Self.retryAfterCap)
        }
        return base
    }
}
