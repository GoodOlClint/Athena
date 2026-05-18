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

/// A resolved caller: a stable principal id (for queue ownership)
/// plus the effective permission set.
struct AuthSubject: Sendable {
    let principal: String
    let permissions: Set<Permission>
}

struct AuthConfig: Sendable {
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
    var isEnabled: Bool { enabled }

    init(
        hashes: [[UInt8]: [String]] = [:],
        store: AthenaStore? = nil,
        enabled: Bool? = nil
    ) {
        self.hashes = hashes
        self.store = store
        self.enabled = enabled ?? !hashes.isEmpty
    }

    /// Bind the DB and (re)compute `enabled` including DB rows.
    func bound(to store: AthenaStore, dbHasCredentials: Bool)
        -> AuthConfig
    {
        AuthConfig(
            hashes: hashes, store: store,
            enabled: !hashes.isEmpty || dbHasCredentials)
    }

    static func sha(_ s: String) -> [UInt8] {
        Array(SHA256.hash(data: Data(s.utf8)))
    }

    static func hex(_ bytes: [UInt8]) -> String {
        bytes.map { String(format: "%02x", $0) }.joined()
    }

    /// Decode a `sha256:<64-hex>` entry to its 32 bytes; nil if it
    /// isn't that form (⇒ caller treats the token as a raw key).
    static func hashEntry(_ token: String) -> [UInt8]? {
        let p = "sha256:"
        guard token.hasPrefix(p) else { return nil }
        let hexStr = token.dropFirst(p.count)
        guard hexStr.count == 64 else { return nil }
        var out: [UInt8] = []
        out.reserveCapacity(32)
        var i = hexStr.startIndex
        while i < hexStr.endIndex {
            let j = hexStr.index(i, offsetBy: 2)
            guard let b = UInt8(hexStr[i..<j], radix: 16) else {
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
    static func load(
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
                fileURLWithPath:
                    (file as NSString).expandingTildeInPath)
            if let perms = (try? FileManager.default
                .attributesOfItem(atPath: url.path)[.posixPermissions]
                as? Int), perms & 0o077 != 0 {
                log.warning(
                    "auth_keys_file is group/other-accessible — chmod 600 \(url.path)"
                )
            }
            if let text = try? String(
                contentsOf: url, encoding: .utf8) {
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
                        .trimmingCharacters(in: CharacterSet(
                            charactersIn: ": \t"))
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
    func resolve(bearer token: String) async -> AuthSubject? {
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
            // itself, so per-key queue ownership still works.
            return AuthSubject(
                principal: "t:" + Self.hex(presented),
                permissions: RBAC.permissions(forRoles: bootRoles))
        }
        if let store,
            let tok = await store.tokenPrincipal(
                hash: Data(presented))
        {
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
    func permissions(forUser username: String) async
        -> Set<Permission>
    {
        guard let store else { return [] }
        let roles = await store.rolesForUser(username: username)
        return RBAC.permissions(forRoles: roles)
    }

    static func constantTimeEqual(_ a: [UInt8], _ b: [UInt8]) -> Bool {
        guard a.count == b.count else { return false }
        var diff: UInt8 = 0
        for i in 0..<a.count { diff |= a[i] ^ b[i] }
        return diff == 0
    }

    /// Fail-safe: refuse to run wide-open on a non-loopback bind.
    func validateStartup(listenHost: String) throws {
        let loopback: Set<String> = [
            "127.0.0.1", "::1", "localhost",
        ]
        if !isEnabled, !loopback.contains(listenHost) {
            throw AuthStartupError.openOnNonLoopback(listenHost)
        }
    }
}

enum AuthStartupError: Error, CustomStringConvertible {
    case openOnNonLoopback(String)
    var description: String {
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
/// `/healthz` + `/ui/login` + `/ui/logout` are always open. Every
/// other route maps to exactly one `Permission`; an unlisted route
/// fails closed to `.inference` (the minimum authenticated
/// capability). Per-owner queue isolation is still enforced in the
/// handlers (M12.6) on top of the `.queueSubmit` gate.
enum AuthPolicy {
    static func required(method: String, path: String)
        -> Permission?
    {
        if path == "/healthz" || path == "/ui/login"
            || path == "/ui/logout"
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
        if path.hasPrefix("/v1/store") { return .storeAdmin }
        if path == "/api/stop" { return .daemonAdmin }
        if path.hasPrefix("/v1/vectors") {
            if path == "/v1/vectors/query" { return .vectorsRead }
            return mutating ? .vectorsWrite : .vectorsRead
        }
        if path.hasPrefix("/v1/queue") { return .queueSubmit }
        // Inference surface (/v1/chat, /v1/embeddings, /v1/audio/*,
        // the /api/* shim) and any unlisted route.
        return .inference
    }
}

struct AuthMiddleware<Context: RequestContext>: RouterMiddleware {
    let config: AuthConfig
    let session: Session

    func handle(
        _ request: Request, context: Context,
        next: (Request, Context) async throws -> Response
    ) async throws -> Response {
        guard config.isEnabled else {
            return try await next(request, context)  // open mode
        }
        let path = request.uri.path
        guard
            let required = AuthPolicy.required(
                method: request.method.rawValue, path: path)
        else { return try await next(request, context) }

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
                return try await next(request, context)
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
                "unauthorized")
        }
        guard subject.permissions.contains(required) else {
            if isUI { return Self.redirect("/ui/login") }
            return Self.deny(
                .forbidden, "insufficient permissions", "forbidden")
        }
        return try await next(request, context)
    }

    private static func redirect(_ location: String) -> Response {
        var headers = HTTPFields()
        headers[.location] = location
        return Response(status: .seeOther, headers: headers)
    }

    private static func deny(
        _ status: HTTPResponse.Status, _ msg: String, _ code: String
    ) -> Response {
        let body =
            #"{"error":{"message":"\#(msg)","type":"auth_error",""#
            + #""code":"\#(code)"}}"#
        var buf = ByteBuffer()
        buf.writeBytes(Data(body.utf8))
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
