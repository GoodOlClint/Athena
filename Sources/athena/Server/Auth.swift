import AthenaStore
import Crypto
import Foundation
import HTTPTypes
import Hummingbird
import Logging
import NIOCore

// Inbound bearer-token auth (M12). Passive-oracle intact — this only
// gates inbound requests. Two tiers: `admin ⊇ inference`. Keys are
// never stored; only their SHA-256. Constant-time compare. Fail-safe:
// no keys on a non-loopback bind ⇒ the daemon refuses to start.

enum AuthTier: Int, Sendable, Comparable {
    case inference = 0
    case admin = 1
    static func < (a: AuthTier, b: AuthTier) -> Bool {
        a.rawValue < b.rawValue
    }
}

struct AuthConfig: Sendable {
    /// Bootstrap token hashes from env/file (in-memory; checked
    /// first). The DB (`auth_tokens`) is the managed store, queried
    /// per request.
    private let hashes: [[UInt8]: AuthTier]
    /// SQLite auth store (managed tokens + users). nil = bootstrap
    /// only.
    private let store: AthenaStore?
    /// Precomputed at startup: any bootstrap hash, OR any DB token,
    /// OR any DB user. Adding the FIRST credential to an already-
    /// running open daemon needs a restart to begin enforcing.
    private let enabled: Bool
    var isEnabled: Bool { enabled }

    init(
        hashes: [[UInt8]: AuthTier] = [:],
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

    /// SHA-256 of a raw key, as the `sha256:<hex>` entry persisted by
    /// `athena auth add` (no secret stored at rest).
    static func hashEntry(forRawKey key: String) -> String {
        "sha256:" + hex(sha(key))
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

    /// Load keys from the file (lines `tier key`, `#` comments) and
    /// the env (`ATHENA_ADMIN_KEYS` / `ATHENA_INFERENCE_KEYS`,
    /// comma-separated). Env augments the file. A given key keeps its
    /// highest tier if listed twice.
    static func load(
        file: String?, env: [String: String],
        log: Logger
    ) -> AuthConfig {
        var map: [[UInt8]: AuthTier] = [:]
        func add(_ key: String, _ tier: AuthTier) {
            let k = key.trimmingCharacters(in: .whitespaces)
            guard !k.isEmpty else { return }
            // `sha256:<64-hex>` ⇒ a pre-hashed entry (recommended;
            // written by `athena auth add` — no secret at rest).
            // Anything else ⇒ a raw key, hashed here.
            let h: [UInt8]
            if let bytes = Self.hashEntry(k) {
                h = bytes
            } else {
                h = sha(k)
            }
            if let cur = map[h] { map[h] = max(cur, tier) } else {
                map[h] = tier
            }
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
                    let tierTok = parts[0]
                        .trimmingCharacters(in: CharacterSet(
                            charactersIn: ": \t"))
                        .lowercased()
                    let tier: AuthTier =
                        tierTok == "admin" ? .admin : .inference
                    add(String(parts[1]), tier)
                }
            } else {
                log.warning(
                    "auth_keys_file unreadable: \(url.path)")
            }
        }
        for (envKey, tier) in [
            ("ATHENA_ADMIN_KEYS", AuthTier.admin),
            ("ATHENA_INFERENCE_KEYS", AuthTier.inference),
        ] {
            for k in (env[envKey] ?? "").split(separator: ",") {
                add(String(k), tier)
            }
        }
        return AuthConfig(hashes: map)
    }

    /// Tier granted to a presented bearer token, or nil. Bootstrap
    /// hashes (env/file) are checked in-memory with a constant-time
    /// compare over the fixed 32-byte digest (no early return — no
    /// timing/count leak). On no match, the DB `auth_tokens` table
    /// is consulted by exact hash (indexed PK; the lookup key is
    /// already a SHA-256, so byte-probing is infeasible).
    func tier(forBearer token: String) async -> AuthTier? {
        let presented = Self.sha(token)
        var granted: AuthTier?
        for (stored, tier) in hashes
        where Self.constantTimeEqual(presented, stored) {
            granted = granted.map { max($0, tier) } ?? tier
        }
        if granted == nil, let store {
            if let t = await store.tokenTier(
                hash: Data(presented))
            {
                granted = t == "admin" ? .admin : .inference
            }
        }
        return granted
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
                + "with NO auth keys. Set auth_keys_file / "
                + "ATHENA_ADMIN_KEYS, or bind 127.0.0.1."
        }
    }
}

/// The tier a route requires, or nil = open (no auth). Admin covers
/// the WebUI, metrics, the shared store, and mutating queue/vector
/// ops; everything else is the inference surface; `/healthz` and
/// `/ui/login` are always open.
enum AuthPolicy {
    static func required(method: String, path: String) -> AuthTier? {
        if path == "/healthz" || path == "/ui/login"
            || path == "/ui/logout"
        {
            return nil
        }
        let mutating =
            method == "POST" || method == "DELETE"
            || method == "PUT" || method == "PATCH"
        if path == "/metrics" { return .admin }
        if path == "/ui" || path.hasPrefix("/ui/") { return .admin }
        if path.hasPrefix("/v1/store") { return .admin }
        if path == "/api/stop" { return .admin }
        // Mutating store/queue/vector admin; reads stay inference.
        if path.hasPrefix("/v1/queue"), mutating { return .admin }
        if path.hasPrefix("/v1/vectors"), mutating,
            path != "/v1/vectors/query"
        {
            return .admin
        }
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

        // /ui* also accepts a valid signed session cookie (admin).
        if isUI,
            let tok = Session.token(
                fromCookieHeader: request.headers[.cookie]),
            session.validate(tok) != nil
        {
            return try await next(request, context)
        }

        guard
            let header = request.headers[.authorization],
            header.hasPrefix("Bearer "),
            case let token = String(header.dropFirst(7)),
            !token.isEmpty,
            let granted = await config.tier(forBearer: token)
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
        if granted < required {
            if isUI { return Self.redirect("/ui/login") }
            return Self.deny(
                .forbidden, "admin privilege required", "forbidden")
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
