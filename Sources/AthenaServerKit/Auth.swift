import AthenaCore
import AthenaStore
import Crypto
import Foundation
import HTTPTypes
import Hummingbird
import Logging
import NIOCore

// Inbound bearer-token auth (M12 → M15.2 RBAC). Passive-oracle
// intact — this only gates inbound requests. Subjects are unified: a
// bearer token resolves to an owning user; the user holds roles; a
// token may further *narrow* (never widen) to a scoped subset. Each
// route requires a `Permission`; the caller's effective permission
// set must contain it. Keys are never stored; only their SHA-256.
// Constant-time compare. Fail-safe: no credentials on a non-loopback
// bind ⇒ the daemon refuses to start.

/// A resolved caller: a stable principal id (for usage metering / audit)
/// plus the effective permission set.
public struct AuthSubject: Sendable {
    public let principal: String
    public let permissions: Set<Permission>

    public init(principal: String, permissions: Set<Permission>) {
        self.principal = principal
        self.permissions = permissions
    }
}

public struct AuthConfig: Sendable {
    /// Bootstrap token hashes from env/file → the role names they
    /// confer (admin key ⇒ `admin`, inference key ⇒ `member`). These
    /// are synthetic principals with NO DB user (no scoped
    /// narrowing). The DB (`auth_tokens`) is the managed store,
    /// queried per request.
    private let hashes: [[UInt8]: [String]]
    /// SQLite auth store (managed tokens + users + role grants).
    /// nil = bootstrap only.
    private let store: AthenaStore?
    /// Precomputed at startup: any bootstrap hash, OR any DB token,
    /// OR any DB user. Adding the FIRST credential to an already-
    /// running open daemon needs a restart to begin enforcing.
    private let enabled: Bool
    public var isEnabled: Bool { enabled }
    /// Global upper bound on a managed token's age in days (M36.1), 0 ⇒
    /// no cap. Enforced at validation relative to the token's `created`,
    /// so lowering it retroactively shortens every token's lifetime.
    /// Bootstrap (env/file) hashes are unaffected — they have no row.
    private let tokenMaxAgeDays: Int

    public init(
        hashes: [[UInt8]: [String]] = [:],
        store: AthenaStore? = nil,
        enabled: Bool? = nil,
        tokenMaxAgeDays: Int = 0
    ) {
        self.hashes = hashes
        self.store = store
        self.enabled = enabled ?? !hashes.isEmpty
        self.tokenMaxAgeDays = tokenMaxAgeDays
    }

    /// Bind the DB and (re)compute `enabled` including DB rows.
    public func bound(
        to store: AthenaStore, dbHasCredentials: Bool,
        tokenMaxAgeDays: Int = 0
    ) -> AuthConfig {
        AuthConfig(
            hashes: hashes, store: store,
            enabled: !hashes.isEmpty || dbHasCredentials,
            tokenMaxAgeDays: tokenMaxAgeDays)
    }

    public static func sha(_ s: String) -> [UInt8] {
        Array(SHA256.hash(data: Data(s.utf8)))
    }

    /// Mint a fresh 256-bit bearer key (`sk-athena-<b64url>`, shown
    /// ONCE) plus its at-rest SHA-256. The single key-generation path,
    /// shared by `athena auth token add` (offline CLI) and
    /// `POST /api/tokens` (M16.4) — no secret is ever persisted.
    public static func mintToken() -> (key: String, hash: Data) {
        let raw = SymmetricKey(size: .bits256).withUnsafeBytes {
            Data($0)
        }
        let key =
            "sk-athena-"
            + raw.base64EncodedString()
            .replacingOccurrences(of: "+", with: "-")
            .replacingOccurrences(of: "/", with: "_")
            .replacingOccurrences(of: "=", with: "")
        return (key, Data(sha(key)))
    }

    public static func hex(_ bytes: [UInt8]) -> String {
        bytes.map { String(format: "%02x", $0) }.joined()
    }

    /// Decode a `sha256:<64-hex>` entry to its 32 bytes; nil if it
    /// isn't that form (⇒ caller treats the token as a raw key).
    public static func hashEntry(_ token: String) -> [UInt8]? {
        let p = "sha256:"
        guard token.hasPrefix(p) else { return nil }
        let hexStr = token.dropFirst(p.count)
        guard hexStr.count == 64 else { return nil }
        var out: [UInt8] = []
        out.reserveCapacity(32)
        var i = hexStr.startIndex
        while i < hexStr.endIndex {
            let j = hexStr.index(i, offsetBy: 2)
            guard let b = UInt8(hexStr[i ..< j], radix: 16) else {
                return nil
            }
            out.append(b)
            i = j
        }
        return out
    }

    /// Load bootstrap keys from the file (lines `admin <key>` /
    /// `inference <key>`, `#` comments) and the env
    /// (`ATHENA_ADMIN_KEYS` / `ATHENA_INFERENCE_KEYS`,
    /// comma-separated). Env augments the file. `admin` ⇒ the
    /// `admin` role, `inference` ⇒ the `member` role; a key listed
    /// twice gets the union of its roles.
    public static func load(
        file: String?, env: [String: String],
        log: Logger
    ) -> AuthConfig {
        var map: [[UInt8]: Set<String>] = [:]
        func add(_ key: String, _ roles: [String]) {
            let k = key.trimmingCharacters(in: .whitespaces)
            guard !k.isEmpty else { return }
            // `sha256:<64-hex>` ⇒ a pre-hashed entry (recommended;
            // written by `athena auth …` — no secret at rest).
            // Anything else ⇒ a raw key, hashed here.
            let h: [UInt8]
            if let bytes = Self.hashEntry(k) {
                h = bytes
            } else {
                h = sha(k)
            }
            map[h, default: []].formUnion(roles)
        }
        if let file, !file.isEmpty {
            let url = URL(
                fileURLWithPath: (file as NSString).expandingTildeInPath)
            let perms =
                (try? FileManager.default
                    .attributesOfItem(atPath: url.path)[.posixPermissions]
                    as? Int)
            if let perms, perms & 0o077 != 0 {
                // A22: fail closed. A group/other-accessible keys file may
                // have been read — or planted — by another local user, so
                // we refuse to honor its keys until 0600 is restored
                // (SSH-style). Env-supplied keys still load; only the
                // file's entries are skipped, and the operator sees an
                // error, not a warning they can ignore.
                log.error(
                    """
                    auth_keys_file \(url.path) is group/other-accessible \
                    (mode \(String(perms & 0o777, radix: 8))); refusing to \
                    load it — chmod 600 to enable
                    """)
            } else if let text = try? String(
                contentsOf: url, encoding: .utf8)
            {
                for raw in text.split(
                    separator: "\n", omittingEmptySubsequences: true)
                {
                    let line = raw.trimmingCharacters(
                        in: .whitespaces)
                    if line.isEmpty || line.hasPrefix("#") { continue }
                    let parts = line.split(
                        separator: " ", maxSplits: 1,
                        omittingEmptySubsequences: true)
                    guard parts.count == 2 else { continue }
                    let tok = parts[0]
                        .trimmingCharacters(
                            in: CharacterSet(
                                charactersIn: ": \t")
                        )
                        .lowercased()
                    add(
                        String(parts[1]),
                        tok == "admin" ? ["admin"] : ["member"])
                }
            } else {
                log.warning(
                    "auth_keys_file unreadable: \(url.path)")
            }
        }
        for (envKey, roles) in [
            ("ATHENA_ADMIN_KEYS", ["admin"]),
            ("ATHENA_INFERENCE_KEYS", ["member"]),
        ] {
            for k in (env[envKey] ?? "").split(separator: ",") {
                add(String(k), roles)
            }
        }
        return AuthConfig(hashes: map.mapValues(Array.init))
    }

    /// Resolve a presented bearer token to a subject, or nil.
    /// Bootstrap hashes (env/file) are checked in-memory with a
    /// constant-time compare over the fixed 32-byte digest (no early
    /// return — no timing/count leak). On no bootstrap match, the DB
    /// `auth_tokens` table is consulted by exact hash (indexed PK;
    /// the lookup key is already a SHA-256, so byte-probing is
    /// infeasible) → owning user → user roles ∩ token scope. Unknown
    /// role names contribute nothing (fail-closed).
    public func resolve(bearer token: String) async -> AuthSubject? {
        let presented = Self.sha(token)
        var bootRoles: Set<String> = []
        var matched = false
        for (stored, roles) in hashes
        where Self.constantTimeEqual(presented, stored) {
            bootRoles.formUnion(roles)
            matched = true
        }
        if matched {
            // Synthetic, stable principal (no DB user) — the digest
            // itself, so per-key usage metering/audit still works.
            return AuthSubject(
                principal: "t:" + Self.hex(presented),
                permissions: RBAC.permissions(forRoles: bootRoles))
        }
        if let store,
            let tok = await store.tokenPrincipal(
                hash: Data(presented))
        {
            // Expiry (M36.1): a past per-token TTL, or — when a global
            // cap is set — a token older than the cap, fails closed.
            // An expired token resolves to nil exactly like an unknown
            // one (no oracle: same 401, no "expired" disclosure).
            let now = Date().timeIntervalSince1970
            if let exp = tok.expires, exp <= now { return nil }
            if tokenMaxAgeDays > 0,
                tok.created + Double(tokenMaxAgeDays) * 86400 <= now
            {
                return nil
            }
            let userRoles = await store.rolesForUser(
                username: tok.username)
            let perms = RBAC.effectivePermissions(
                userRoles: userRoles,
                tokenScopedRoles: tok.scopedRoles)
            return AuthSubject(
                principal: "u:" + tok.username, permissions: perms)
        }
        return nil
    }

    /// Effective permissions for a logged-in WebUI user (session
    /// cookie path). Roles drive access — no token scoping applies.
    public func permissions(forUser username: String) async
        -> Set<Permission>
    {
        guard let store else { return [] }
        let roles = await store.rolesForUser(username: username)
        return RBAC.permissions(forRoles: roles)
    }

    public static func constantTimeEqual(_ a: [UInt8], _ b: [UInt8]) -> Bool {
        guard a.count == b.count else { return false }
        var diff: UInt8 = 0
        for i in 0 ..< a.count { diff |= a[i] ^ b[i] }
        return diff == 0
    }

    /// Fail-safe: refuse to run wide-open on a non-loopback bind.
    public func validateStartup(listenHost: String) throws {
        let loopback: Set<String> = [
            "127.0.0.1", "::1", "localhost",
        ]
        if !isEnabled, !loopback.contains(listenHost) {
            throw AuthStartupError.openOnNonLoopback(listenHost)
        }
    }
}

public enum AuthStartupError: Error, CustomStringConvertible {
    case openOnNonLoopback(String)
    public var description: String {
        switch self {
        case .openOnNonLoopback(let h):
            return
                "refusing to start: listening on \(h) (non-loopback) "
                + "with NO auth credentials. Seed a user/token, set "
                + "auth_keys_file / ATHENA_ADMIN_KEYS, or bind "
                + "127.0.0.1."
        }
    }
}

/// The permission a route requires, or nil = open (no auth).
/// `/healthz` + `/openapi.json` + `/ui/login` + `/ui/logout` are always
/// open (discovery + login). Every
/// other route maps to exactly one `Permission`; an unlisted route
/// fails closed to `.inference` (the minimum authenticated
/// capability).
public enum AuthPolicy {
    public static func required(method: String, path: String)
        -> Permission?
    {
        if path == "/healthz" || path == "/openapi.json"
            || path == "/ui/login" || path == "/ui/logout"
        {
            return nil
        }
        let mutating =
            method == "POST" || method == "DELETE"
            || method == "PUT" || method == "PATCH"
        if path == "/metrics" { return .metricsRead }
        if path == "/ui" || path.hasPrefix("/ui/") {
            return .daemonAdmin
        }
        // OpenAI model discovery is a read-only store projection (M31.1),
        // gated like the native `/api/models` reads — model.read, never
        // the inference catch-all.
        if path == "/v1/models" || path.hasPrefix("/v1/models/") {
            return .modelRead
        }
        if path == "/api/admin" || path.hasPrefix("/api/admin/") {
            return .daemonAdmin
        }
        // The audit trail is a privileged oversight view — admin-only
        // (daemon.admin), no owner-scoping (M30.2).
        if path == "/api/audit" { return .daemonAdmin }
        // M45.5: daemon-log oversight. The unified-log entries carry
        // req=/principal= across all users + reveal internal call
        // sites; admin-only (daemon.admin) matches the
        // sensitivity profile of /api/audit.
        if path == "/api/logs" || path == "/api/logs/stream" {
            return .daemonAdmin
        }
        // ADR 037 — daemon-mediated config (GET current, PUT a scalar). A
        // privileged control-plane surface; admin-only (daemon.admin). The
        // security deny-list (auth/TLS/encryption/data-dir/debugger keys) is
        // enforced in the handler on top of this gate.
        if path == "/api/config" { return .daemonAdmin }
        // Usage is inference-tier: any authenticated caller sees its OWN
        // counters (handler owner-scopes); an admin sees all. Billing-
        // sensitive, so NOT exposed to the read-only role (M27.3).
        if path == "/api/usage" { return .inference }
        if path == "/api/models" || path.hasPrefix("/api/models/") {
            return mutating ? .modelWrite : .modelRead
        }
        if path == "/api/roles" || path.hasPrefix("/api/roles/") {
            return .usersRead  // read-only RBAC catalog
        }
        if path == "/api/users" || path.hasPrefix("/api/users/") {
            return mutating ? .usersAdmin : .usersRead
        }
        if path == "/api/tokens" || path.hasPrefix("/api/tokens/") {
            // Listing exposes ownership/scope; minting/removing are
            // privileged — tokens.admin for the whole surface.
            return .tokensAdmin
        }
        // Inference surface (/v1/chat, /v1/messages (ADR 036 Anthropic),
        // /v1/embeddings, /v1/audio/*, /v1/video/* (ADR 022)) and any
        // unlisted route — all inference-tier.
        return .inference
    }
}

public struct AuthMiddleware<Context: RequestContext>: RouterMiddleware {
    let config: AuthConfig
    let session: Session

    public init(config: AuthConfig, session: Session) {
        self.config = config
        self.session = session
    }

    public func handle(
        _ request: Request, context: Context,
        next: (Request, Context) async throws -> Response
    ) async throws -> Response {
        // M45.3: bind a per-request LogScope (req-id + resolved
        // principal) so every Logger emission within the downstream
        // task hierarchy auto-includes `req=` / `principal=`
        // metadata. Operator filter `category == "daemon" AND
        // eventMessage CONTAINS "req=<uuid>"` correlates all log
        // lines for a single request across handlers.
        //
        // The principal is filled when:
        //   - auth-off (open mode): nil — req-id alone correlates.
        //   - cookie session (/ui): the validated user.
        //   - bearer-authorized: the resolved subject's principal.
        // Auth-denial paths return BEFORE the next() call, so no
        // scope is bound for those (the audit_log entry is the
        // forensic record for denied requests).
        guard config.isEnabled else {
            let scope = LogScope(principal: nil)
            return try await LogScope.$current.withValue(scope) {
                try await next(request, context)  // open mode
            }
        }
        let path = request.uri.path
        guard
            let required = AuthPolicy.required(
                method: request.method.rawValue, path: path)
        else {
            let scope = LogScope(principal: nil)
            return try await LogScope.$current.withValue(scope) {
                try await next(request, context)
            }
        }

        let isUI = path == "/ui" || path.hasPrefix("/ui/")

        // /ui* via a valid signed session cookie: the logged-in
        // user's own roles drive access (not a flat admin tier).
        if isUI,
            let tok = Session.token(
                fromCookieHeader: request.headers[.cookie]),
            let user = session.validate(tok)
        {
            let perms = await config.permissions(forUser: user)
            if perms.contains(required) {
                // A5: publish the single resolved caller so /ui handlers
                // (uiCaller, callerPermissions, audit) read THIS resolution
                // instead of re-validating the cookie themselves.
                let caller = ResolvedCaller(
                    principal: "u:\(user)", permissions: perms,
                    uiUser: user)
                let scope = LogScope(principal: "u:\(user)")
                return try await ResolvedCaller.$current.withValue(caller) {
                    try await LogScope.$current.withValue(scope) {
                        try await next(request, context)
                    }
                }
            }
            return Self.redirect("/ui/login")
        }

        guard
            let header = request.headers[.authorization],
            header.hasPrefix("Bearer "),
            case let token = String(header.dropFirst(7)),
            !token.isEmpty,
            let subject = await config.resolve(bearer: token)
        else {
            // Browsers can't send bearer on navigation — send them
            // to the login page instead of a JSON 401.
            if isUI {
                return Self.redirect("/ui/login")
            }
            return Self.deny(
                .unauthorized, "missing or invalid bearer token",
                "unauthorized",
                hint:
                    "Set ATHENA_KEY, pass --key <secret>, run "
                    + "`athena auth login --host <h> --port <p>` to "
                    + "cache it in your Keychain, or sign in at "
                    + "/ui/login. Mint a token with `athena auth "
                    + "token add --user <name>` (admin) or offline "
                    + "via `--data-dir`.")
        }
        guard subject.permissions.contains(required) else {
            if isUI { return Self.redirect("/ui/login") }
            return Self.deny(
                .forbidden, "insufficient permissions", "forbidden",
                hint:
                    "This token's role lacks the permission this route "
                    + "requires. Use a higher-privilege token, or "
                    + "grant the role via `athena auth role grant "
                    + "<user> <role>`.")
        }
        // A5: publish the single resolved caller (bearer path). Downstream
        // helpers (callerPermissions, audit) read this
        // instead of calling config.resolve(bearer:) a second time.
        let caller = ResolvedCaller(
            principal: subject.principal,
            permissions: subject.permissions, uiUser: nil)
        let scope = LogScope(principal: subject.principal)
        return try await ResolvedCaller.$current.withValue(caller) {
            try await LogScope.$current.withValue(scope) {
                try await next(request, context)
            }
        }
    }

    private static func redirect(_ location: String) -> Response {
        var headers = HTTPFields()
        headers[.location] = location
        return Response(status: .seeOther, headers: headers)
    }

    private static func deny(
        _ status: HTTPResponse.Status, _ msg: String, _ code: String,
        hint: String? = nil
    ) -> Response {
        // JSON-encoded so the message + code can carry any character
        // safely. The hand-formatted predecessor concatenated two raw
        // string literals around a `","` seam, which produced
        // `"type":"auth_error",""code":` — three quotes — and broke
        // JSON parsing for clients that strict-parse the error body.
        // M43.4 #5 — `hint` is the operator-facing remediation
        // (`ATHENA_KEY`, `athena auth login`, role-grant guidance).
        // Clients render it via the CLI fail() helper; non-CLI
        // consumers ignore the field.
        var err: [String: Any] = [
            "message": msg,
            "type": "auth_error",
            "code": code,
        ]
        if let hint, !hint.isEmpty {
            err["hint"] = hint
        }
        let envelope: [String: Any] = ["error": err]
        let bodyData =
            (try? JSONSerialization.data(withJSONObject: envelope))
            ?? Data(#"{"error":{"message":"auth error"}}"#.utf8)
        var buf = ByteBuffer()
        buf.writeBytes(bodyData)
        var headers = HTTPFields()
        headers[.contentType] = "application/json"
        if status == .unauthorized {
            headers[.wwwAuthenticate] = "Bearer"
        }
        return Response(
            status: status, headers: headers,
            body: ResponseBody(byteBuffer: buf))
    }
}
