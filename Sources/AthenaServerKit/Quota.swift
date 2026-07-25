import AthenaCore
import AthenaStore
import Foundation
import HTTPTypes
import Hummingbird
import NIOCore

// ADR 041 §3/§4 — per-principal token-budget enforcement. Same shape and slot
// as `RateLimitMiddleware` (auth-gated, principal-keyed, 429) but a different
// unit and a different refusal code: the limiter caps request RATE, this caps
// TOKENS per period, and the governor's 503 caps memory. Three orthogonal
// reasons to refuse, three distinct codes.
//
// Honesty boundary (ADR 041 §3, binding): a request's token cost is unknown
// before it runs, so enforcement is pre-request against ACCRUED usage. A
// principal with one token left is admitted and may overshoot by that one
// request — bounded by the prefill ceiling plus `max_completion_tokens`, not
// unbounded. Mid-stream cancellation is deliberately not built.

/// The pure budget verdict (ADR 009: MLX-free, unit-pinned).
public enum QuotaDecision: Sendable, Equatable {
    /// No budget applies — no enforcement and NO advisory headers (an absent
    /// header means "no cap", never "zero remaining").
    case unlimited
    case within(limit: Int, remaining: Int)
    case exhausted(limit: Int)

    /// `budget` nil or non-positive ⇒ unlimited (`0` is the documented
    /// "unlimited" value both in config and as a per-user override). Reaching
    /// the budget exactly is exhausted: the next request would have to spend at
    /// least one more token.
    public static func evaluate(used: Int, budget: Int?) -> QuotaDecision {
        guard let budget, budget > 0 else { return .unlimited }
        let remaining = budget - max(0, used)
        return remaining > 0
            ? .within(limit: budget, remaining: remaining)
            : .exhausted(limit: budget)
    }

    public var isExhausted: Bool {
        if case .exhausted = self { return true }
        return false
    }
}

public struct QuotaMiddleware<Context: RequestContext>: RouterMiddleware {
    let store: AthenaStore
    let auth: AuthConfig
    /// Global default budget per period (nil / non-positive ⇒ unlimited unless
    /// a user carries an override).
    let defaultBudget: Int?
    let window: QuotaWindow

    public init(
        store: AthenaStore, auth: AuthConfig, defaultBudget: Int?,
        window: QuotaWindow
    ) {
        self.store = store
        self.auth = auth
        self.defaultBudget = defaultBudget
        self.window = window
    }

    /// Routes that spend model tokens — the ONLY ones enforced. Deliberately
    /// not `RateLimitMiddleware.throttled()`: an exhausted principal must keep
    /// being able to read `GET /api/usage` to see why it was refused, and the
    /// control plane spends no tokens so gating it would add no protection.
    /// `POST /v1/chat/completions/count_tokens` is excluded by exact match —
    /// it exists so a client can stay UNDER its budget (ADR 042).
    public static func metered(_ path: String) -> Bool {
        path == "/v1/chat/completions" || path == "/v1/embeddings"
            || path == "/v1/messages"
    }

    public func handle(
        _ request: Request, context: Context,
        next: (Request, Context) async throws -> Response
    ) async throws -> Response {
        // Auth-off loopback (dev) bypasses entirely — ADR 025: no store, no
        // principal, nothing to charge. Documented in `athena doctor`, because
        // an operator expecting their dev box to be bounded would be wrong.
        guard auth.isEnabled, Self.metered(request.uri.path) else {
            return try await next(request, context)
        }
        guard
            let header = request.headers[.authorization],
            header.hasPrefix("Bearer "),
            case let token = String(header.dropFirst(7)), !token.isEmpty,
            let subject = await auth.resolve(bearer: token)
        else { return try await next(request, context) }

        let now = Date().timeIntervalSince1970
        let periodStart = window.periodStart(containing: now)
        let budget = await resolvedBudget(principal: subject.principal)
        // Unlimited: no read of the counters, no headers, byte-identical to
        // the pre-quota response.
        guard let budget, budget > 0 else {
            return try await next(request, context)
        }

        let used = await periodTokens(
            principal: subject.principal, periodStart: periodStart)
        if QuotaDecision.evaluate(used: used, budget: budget).isExhausted {
            return Self.exhausted(
                retryAfter: window.secondsUntilRoll(from: now),
                resets: window.nextRoll(after: now))
        }

        var response = try await next(request, context)
        // Advisory headers are read AFTER the handler has metered, so
        // `remaining` reflects the request the client just made. One extra
        // primary-key read per metered request — negligible beside inference,
        // and only on the paths that actually carry a budget.
        let after = await periodTokens(
            principal: subject.principal, periodStart: periodStart)
        Self.attachAdvisory(
            &response,
            decision: QuotaDecision.evaluate(used: after, budget: budget),
            resets: window.nextRoll(after: now))
        return response
    }

    /// Per-user override (`auth_users.token_budget`) when the principal is a DB
    /// user, else the configured default. A bootstrap-key principal (`t:<hash>`)
    /// has no user row, so it can only inherit the default.
    private func resolvedBudget(principal: String) async -> Int? {
        guard principal.hasPrefix("u:") else { return defaultBudget }
        let username = String(principal.dropFirst(2))
        return await store.userBudget(username: username) ?? defaultBudget
    }

    private func periodTokens(principal: String, periodStart: Double) async
        -> Int
    {
        guard let row = await store.usage(principal: principal) else {
            return 0
        }
        return QuotaWindow.periodTokens(
            storedPeriodStart: row.periodStart,
            promptTokens: row.periodPromptTokens,
            completionTokens: row.periodCompletionTokens,
            currentPeriodStart: periodStart)
    }

    private static func exhausted(retryAfter: Int, resets: Double) -> Response {
        QuotaHeaders.exhaustedResponse(
            retryAfter: retryAfter, resets: resets)
    }

    private static func attachAdvisory(
        _ response: inout Response, decision: QuotaDecision, resets: Double
    ) {
        QuotaHeaders.attach(&response, decision: decision, resets: resets)
    }
}

/// The wire half, non-generic so it is reachable from tests without
/// specializing the middleware over a request context.
public enum QuotaHeaders {
    public static let limit = HTTPField.Name("x-athena-tokens-limit")!
    public static let remaining = HTTPField.Name("x-athena-tokens-remaining")!
    public static let reset = HTTPField.Name("x-athena-tokens-reset")!

    /// 429 in the canonical error envelope, with `Retry-After` in seconds and
    /// the reset instant. Distinct code from the rate limiter's `rate_limited`
    /// so an operator can tell throttling from exhaustion.
    /// The 429 body, as a string — separated out so the envelope shape is
    /// assertable without collecting a `ResponseBody`.
    public static func exhaustedBody(resets: Double) -> String {
        #"{"error":{"message":"token budget exhausted; resets "#
            + iso8601(resets)
            + #"","type":"insufficient_quota","code":"quota_exceeded"}}"#
    }

    public static func exhaustedResponse(retryAfter: Int, resets: Double)
        -> Response
    {
        let iso = iso8601(resets)
        let body = exhaustedBody(resets: resets)
        var buf = ByteBuffer()
        buf.writeString(body)
        var headers: HTTPFields = [
            .contentType: "application/json",
            .retryAfter: String(retryAfter),
        ]
        headers[reset] = iso
        return Response(
            status: .tooManyRequests, headers: headers,
            body: .init(byteBuffer: buf))
    }

    /// Attach the advisory triple — or nothing at all when unlimited, so an
    /// absent header can only mean "no cap".
    public static func attach(
        _ response: inout Response, decision: QuotaDecision, resets: Double
    ) {
        let lim: Int
        let rem: Int
        switch decision {
        case .unlimited: return
        case .within(let l, let r):
            lim = l
            rem = r
        case .exhausted(let l):
            lim = l
            rem = 0
        }
        response.headers[limit] = String(lim)
        response.headers[remaining] = String(rem)
        response.headers[reset] = iso8601(resets)
    }

    /// UTC ISO8601 (`2026-08-01T00:00:00Z`) — the period roll instant. The
    /// boundary itself is local; the wire format is UTC so a client needs no
    /// knowledge of the daemon's time zone.
    public static func iso8601(_ epoch: Double) -> String {
        let f = ISO8601DateFormatter()
        f.timeZone = TimeZone(identifier: "UTC")
        return f.string(from: Date(timeIntervalSince1970: epoch))
    }
}
