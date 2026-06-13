import AthenaCore
import Hummingbird
import NIOCore

// M65.6 (audit A5/A3) — the daemon's single request context.
//
// The router was `Router()` (default `BasicRequestContext`), which
// carries no client address, so the `/ui/login` limiter (A3) had no peer
// IP to key on and the caller/permission resolution was reimplemented at
// every site that needed it (A5). This context fixes both:
//
//   * It conforms to `RemoteAddressRequestContext`, so `remoteAddress`
//     (the TCP peer, straight off the channel) is available to handlers —
//     specifically the login limiter, which keys on the peer address ONLY
//     and never trusts `X-Forwarded-For` (ADR 004; XFF is spoofable
//     without an enforced trusted proxy).
//   * `AuthMiddleware` resolves the caller ONCE and publishes the result
//     via `ResolvedCaller` (below), so the downstream helpers
//     (`callerPermissions`, `uiCaller`, `auditPrincipal`, `queuePrincipal`)
//     read a single authoritative resolution instead of each re-deriving
//     it from the headers (the A5 drift: bearer-vs-cookie order, sentinels,
//     loopback-trust all diverged across the copies).

/// The custom request context for the whole router. Adds the connected
/// client's address to the core context so per-IP controls (the A3 login
/// limiter) have a trustworthy key.
public struct AppRequestContext: RequestContext, RemoteAddressRequestContext {
    public var coreContext: CoreRequestContextStorage
    /// The TCP peer address (nil for non-IP transports). This is the
    /// channel's real remote address — NOT a forwarded header — so it
    /// cannot be spoofed by a client that merely sets `X-Forwarded-For`.
    public let remoteAddress: SocketAddress?

    public init(source: ApplicationRequestContextSource) {
        self.coreContext = .init(source: source)
        self.remoteAddress = source.channel.remoteAddress
    }
}

/// The caller `AuthMiddleware` resolved for the current request, published
/// as a task-local so the request-scoped helpers read the SINGLE
/// resolution rather than re-deriving it (A5). Bound exactly where the
/// existing `LogScope` is bound — the same per-request task hierarchy — so
/// every handler reached from `next()` observes it. Only bound on the
/// enabled-auth paths (bearer + cookie); the auth-off (open) fast-paths in
/// the readers don't consult it, so open mode leaves it nil by design.
public struct ResolvedCaller: Sendable {
    /// Stable principal id: `u:<user>` (managed token or cookie session),
    /// `t:<hash>` (bootstrap key). Drives queue ownership + audit records.
    public let principal: String
    /// The caller's effective permission set (role ∩ token-scope, already
    /// computed by the resolver).
    public let permissions: Set<Permission>
    /// The logged-in WebUI username when the caller authed via a session
    /// cookie; nil for a bearer caller. Lets `uiCaller` render RBAC-aware
    /// pages without a second session lookup.
    public let uiUser: String?

    public init(
        principal: String, permissions: Set<Permission>, uiUser: String?
    ) {
        self.principal = principal
        self.permissions = permissions
        self.uiUser = uiUser
    }

    @TaskLocal public static var current: ResolvedCaller?
}
