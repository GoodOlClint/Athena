import Foundation
import HTTPTypes
import Hummingbird
import NIOCore

// Inbound abuse protection (M29.1). A per-principal token-bucket rate
// limiter that COMPOSES WITH — but is distinct from — the governor's
// Metal-OOM 503 backpressure: the governor caps *memory*, this caps
// *request rate* per caller, so one key can't flood the sync inference
// path. Inbound-only ⇒ passive-oracle thesis intact. Opt-in: a
// non-positive rate disables it (the daemon builds no limiter, the
// middleware isn't installed). Rejection is a 429 + Retry-After using
// the standard {error:{message,type,code}} body — visibly distinct from
// the governor's 503.

/// Per-principal token bucket. Each principal gets a bucket of `burst`
/// tokens that refills at `rate` tokens/sec; a request consumes one. An
/// empty bucket is rejected with the whole seconds until the next token
/// (the Retry-After hint). Buckets are created lazily and bounded by the
/// distinct-principal count (finite: the seeded users + bootstrap keys).
actor RateLimiter {
    private struct Bucket {
        var tokens: Double
        var last: Double
    }
    /// Sustained admitted rate (tokens refilled per second).
    let rate: Double
    /// Bucket capacity — the largest instantaneous burst above `rate`.
    let burst: Double
    private var buckets: [String: Bucket] = [:]

    init(rate: Double, burst: Double) {
        self.rate = rate
        self.burst = max(1, burst)
    }

    /// Consume one token for `principal`. Returns nil when admitted, or
    /// the Retry-After (whole seconds, ≥1) when the bucket is empty.
    func take(_ principal: String, now: Double) -> Int? {
        var b = buckets[principal] ?? Bucket(tokens: burst, last: now)
        let elapsed = max(0, now - b.last)
        b.tokens = min(burst, b.tokens + elapsed * rate)
        b.last = now
        defer { buckets[principal] = b }
        if b.tokens >= 1 {
            b.tokens -= 1
            return nil
        }
        let secs = (1 - b.tokens) / rate
        return max(1, Int(secs.rounded(.up)))
    }
}

struct RateLimitMiddleware<Context: RequestContext>: RouterMiddleware {
    let limiter: RateLimiter
    let auth: AuthConfig

    func handle(
        _ request: Request, context: Context,
        next: (Request, Context) async throws -> Response
    ) async throws -> Response {
        // Auth-off loopback (dev) bypasses entirely: there are no
        // principals to key on, and the fail-safe already confines
        // open mode to a loopback bind.
        guard auth.isEnabled else {
            return try await next(request, context)
        }
        // Only throttle the work/API surface — never health probes, the
        // monitoring console, or /metrics, so a launchd health check or
        // a busy dashboard can't trip the limit.
        guard Self.throttled(request.uri.path) else {
            return try await next(request, context)
        }
        // Key by the resolved principal (per-USER — shared across that
        // user's tokens — using the same identity as queue ownership and
        // usage metering). A throttled path reaching here has already
        // passed AuthMiddleware, so a bearer resolves; a request that
        // doesn't carry one (e.g. a cookie-authed /ui asset, already
        // excluded above) falls through untouched.
        guard
            let header = request.headers[.authorization],
            header.hasPrefix("Bearer "),
            case let token = String(header.dropFirst(7)), !token.isEmpty,
            let subject = await auth.resolve(bearer: token)
        else { return try await next(request, context) }

        if let retryAfter = await limiter.take(
            subject.principal, now: Date().timeIntervalSince1970)
        {
            return Self.tooMany(retryAfter)
        }
        return try await next(request, context)
    }

    /// Paths subject to rate limiting: the inference + admin API
    /// surface. `/healthz`, `/metrics`, and the WebUI (`/ui*`) are exempt
    /// — operator/dashboard traffic, not billable work.
    static func throttled(_ path: String) -> Bool {
        if path == "/healthz" || path == "/metrics" { return false }
        if path == "/ui" || path.hasPrefix("/ui/") { return false }
        return true
    }

    private static func tooMany(_ retryAfter: Int) -> Response {
        let body =
            #"{"error":{"message":"rate limit exceeded; retry after "#
            + "\(retryAfter)s"
            + #"","type":"rate_limit_error","code":"rate_limited"}}"#
        var buf = ByteBuffer()
        buf.writeBytes(Data(body.utf8))
        var headers = HTTPFields()
        headers[.contentType] = "application/json"
        headers[.retryAfter] = String(retryAfter)
        return Response(
            status: .tooManyRequests, headers: headers,
            body: ResponseBody(byteBuffer: buf))
    }
}
