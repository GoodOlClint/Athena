import Crypto
import Foundation

/// WebUI session cookies (M12.2). Stateless: a signed
/// `user:expiry` token, HMAC-SHA256 with a per-process random
/// secret (sessions invalidate on daemon restart — acceptable for
/// an appliance, and a security plus). No server-side session store.
struct Session: Sendable {
    static let cookieName = "athena_session"
    /// 12h default lifetime.
    static let ttl: TimeInterval = 12 * 3600

    private let secret: SymmetricKey

    init() { secret = SymmetricKey(size: .bits256) }

    private static func b64url(_ d: Data) -> String {
        d.base64EncodedString()
            .replacingOccurrences(of: "+", with: "-")
            .replacingOccurrences(of: "/", with: "_")
            .replacingOccurrences(of: "=", with: "")
    }
    private static func unb64url(_ s: String) -> Data? {
        var t = s.replacingOccurrences(of: "-", with: "+")
            .replacingOccurrences(of: "_", with: "/")
        while t.count % 4 != 0 { t += "=" }
        return Data(base64Encoded: t)
    }

    /// `<b64url(payload)>.<b64url(hmac)>`, payload = `user:expiry`.
    func mint(user: String, now: Date = Date()) -> String {
        let exp = Int(now.addingTimeInterval(Self.ttl)
            .timeIntervalSince1970)
        let payload = Data("\(user):\(exp)".utf8)
        let mac = HMAC<SHA256>.authenticationCode(
            for: payload, using: secret)
        return Self.b64url(payload) + "." + Self.b64url(Data(mac))
    }

    /// Constant-time verify (HMAC + unexpired). Returns the user.
    func validate(_ token: String, now: Date = Date()) -> String? {
        let parts = token.split(
            separator: ".", maxSplits: 1,
            omittingEmptySubsequences: false)
        guard parts.count == 2,
            let payload = Self.unb64url(String(parts[0])),
            let mac = Self.unb64url(String(parts[1]))
        else { return nil }
        guard
            HMAC<SHA256>.isValidAuthenticationCode(
                mac, authenticating: payload, using: secret)
        else { return nil }
        guard let s = String(data: payload, encoding: .utf8) else {
            return nil
        }
        let f = s.split(separator: ":")
        guard f.count == 2, let exp = TimeInterval(f[1]),
            exp > now.timeIntervalSince1970
        else { return nil }
        return String(f[0])
    }

    // MARK: - CSRF (M18.1)

    /// Per-session CSRF token = `HMAC(secret, "csrf:" + user)`,
    /// b64url. Defense-in-depth ON TOP of the HttpOnly +
    /// SameSite=Strict cookie: a destructive /ui/api/* form/fetch
    /// must echo this; an off-site page cannot read it (no cookie
    /// read, no CORS) so it cannot forge the field. Bound to the
    /// per-process secret (rotates on restart, like the session) and
    /// to the user (a different account's token won't validate).
    func csrf(user: String) -> String {
        let mac = HMAC<SHA256>.authenticationCode(
            for: Data("csrf:\(user)".utf8), using: secret)
        return Self.b64url(Data(mac))
    }

    /// Constant-time CSRF check (HMAC verify on the fixed digest).
    func validateCSRF(_ token: String?, user: String) -> Bool {
        guard let token, let mac = Self.unb64url(token) else {
            return false
        }
        return HMAC<SHA256>.isValidAuthenticationCode(
            mac, authenticating: Data("csrf:\(user)".utf8),
            using: secret)
    }

    /// Read our cookie out of a `Cookie:` header value.
    static func token(fromCookieHeader header: String?) -> String? {
        guard let header else { return nil }
        for pair in header.split(separator: ";") {
            let kv = pair.split(
                separator: "=", maxSplits: 1)
            guard kv.count == 2 else { continue }
            if kv[0].trimmingCharacters(in: .whitespaces)
                == cookieName
            {
                return kv[1].trimmingCharacters(in: .whitespaces)
            }
        }
        return nil
    }

    static func setCookie(_ value: String) -> String {
        "\(cookieName)=\(value); HttpOnly; SameSite=Strict; "
            + "Path=/; Max-Age=\(Int(ttl))"
    }
    static let clearCookie =
        "\(cookieName)=; HttpOnly; SameSite=Strict; Path=/; "
        + "Max-Age=0"
}
