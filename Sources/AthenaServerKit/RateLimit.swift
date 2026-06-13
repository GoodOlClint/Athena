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
public actor RateLimiter {
    private struct Bucket {
        var tokens: Double
        var last: Double
    }
    /// Sustained admitted rate (tokens refilled per second).
    public let rate: Double
    /// Bucket capacity — the largest instantaneous burst above `rate`.
    public let burst: Double
    private var buckets: [String: Bucket] = [:]

    public init(rate: Double, burst: Double) {
        self.rate = rate
        self.burst = max(1, burst)
    }

    /// Consume one token for `principal`. Returns nil when admitted, or
    /// the Retry-After (whole seconds, ≥1) when the bucket is empty.
    public func take(_ principal: String, now: Double) -> Int? {
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

public struct RateLimitMiddleware<Context: RequestContext>: RouterMiddleware {
    let limiter: RateLimiter
    let auth: AuthConfig

    public init(limiter: RateLimiter, auth: AuthConfig) {
        self.limiter = limiter
        self.auth = auth
    }

    public func handle(
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
    public static func throttled(_ path: String) -> Bool {
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

// Concurrency caps (M29.2). A second, orthogonal abuse control: instead
// of capping the request *rate* over time, this bounds the number of
// requests a caller (and the whole daemon) can have IN FLIGHT at once,
// so a handful of slow inference calls from one key can't tie up the
// box. Distinct from the governor's memory 503 (this is admission
// control on count, not memory) — rejection is a 429 with a distinct
// `concurrency_limit` code. Opt-in: both caps non-positive ⇒ no
// middleware installed.

/// In-flight request accounting against a global cap and a per-principal
/// cap. A cap of 0 means "unlimited" for that dimension. `acquire`
/// reserves a slot in BOTH dimensions atomically (so a per-principal
/// rejection never burns a global slot); the caller MUST `release`
/// exactly once for every successful `acquire`.
public actor ConcurrencyLimiter {
    /// Max simultaneous in-flight requests across all principals
    /// (0 = unlimited).
    public let global: Int
    /// Max simultaneous in-flight requests for any single principal
    /// (0 = unlimited).
    public let perPrincipal: Int
    private var globalInFlight = 0
    private var perPrincipalInFlight: [String: Int] = [:]

    public init(global: Int, perPrincipal: Int) {
        self.global = global
        self.perPrincipal = perPrincipal
    }

    /// Reserve one slot for `principal`. Returns true if admitted (the
    /// caller owns a slot it must release); false if either cap is full.
    public func acquire(_ principal: String) -> Bool {
        if global > 0, globalInFlight >= global { return false }
        if perPrincipal > 0,
            (perPrincipalInFlight[principal] ?? 0) >= perPrincipal
        {
            return false
        }
        globalInFlight += 1
        perPrincipalInFlight[principal, default: 0] += 1
        return true
    }

    public func release(_ principal: String) {
        globalInFlight = max(0, globalInFlight - 1)
        if let n = perPrincipalInFlight[principal] {
            if n <= 1 { perPrincipalInFlight[principal] = nil } else {
                perPrincipalInFlight[principal] = n - 1
            }
        }
    }
}

public struct ConcurrencyMiddleware<Context: RequestContext>: RouterMiddleware {
    let limiter: ConcurrencyLimiter
    let auth: AuthConfig

    public init(limiter: ConcurrencyLimiter, auth: AuthConfig) {
        self.limiter = limiter
        self.auth = auth
    }

    public func handle(
        _ request: Request, context: Context,
        next: (Request, Context) async throws -> Response
    ) async throws -> Response {
        // Same scope as the rate limiter: auth-off loopback (dev) and
        // the exempt non-work paths bypass entirely.
        guard auth.isEnabled,
            RateLimitMiddleware<Context>.throttled(request.uri.path)
        else { return try await next(request, context) }

        guard
            let header = request.headers[.authorization],
            header.hasPrefix("Bearer "),
            case let token = String(header.dropFirst(7)), !token.isEmpty,
            let subject = await auth.resolve(bearer: token)
        else { return try await next(request, context) }

        guard await limiter.acquire(subject.principal) else {
            return Self.tooBusy()
        }
        // `defer` can't await, so release explicitly on BOTH the success
        // and the throwing path — a slot must never leak.
        do {
            var response = try await next(request, context)
            // NA3 — hold the slot until the (possibly streamed) body has been
            // fully written, not just until the handler returns its lazy
            // AsyncStream body. Otherwise the concurrency caps don't bound
            // streamed generations at all (one key could open unbounded
            // concurrent streams).
            let principal = subject.principal
            response.body = response.body.onBodyComplete {
                await limiter.release(principal)
            }
            return response
        } catch {
            await limiter.release(subject.principal)
            throw error
        }
    }

    /// Retry-After hint for a concurrency rejection: a slot frees when an
    /// in-flight request finishes, so "try again shortly" (1s) — there's
    /// no time-based refill to compute, unlike the rate limiter.
    private static func tooBusy() -> Response {
        let body =
            #"{"error":{"message":"too many concurrent requests","#
            + #""type":"concurrency_limit_error","#
            + #""code":"concurrency_limit"}}"#
        var buf = ByteBuffer()
        buf.writeBytes(Data(body.utf8))
        var headers = HTTPFields()
        headers[.contentType] = "application/json"
        headers[.retryAfter] = "1"
        return Response(
            status: .tooManyRequests, headers: headers,
            body: ResponseBody(byteBuffer: buf))
    }
}

extension ResponseBody {
    /// NA3 (M69.2) — run `postWrite` once this body has been FULLY written to
    /// the channel (success OR throw), not when the handler returns. A
    /// streaming inference handler returns a lazy `AsyncStream` body almost
    /// immediately; the GPU decode that fills it happens later, while the body
    /// is drained. Replicates Hummingbird's `package` `withPostWriteClosure`
    /// using only public `ResponseBody` API so the concurrency slot + the
    /// inflight/latency metrics can be held for the streamed body's whole
    /// lifetime. Idempotent at the call site: `postWrite` fires exactly once.
    func onBodyComplete(
        _ postWrite: @escaping @Sendable () async -> Void
    ) -> ResponseBody {
        let body = self
        return ResponseBody(contentLength: self.contentLength) { writer in
            do {
                try await body.write(writer)
                await postWrite()
            } catch {
                await postWrite()
                throw error
            }
        }
    }
}
