import Foundation

#if canImport(CFNetwork)
    import CFNetwork
#endif

/// Egress-proxy support for the only outbound the appliance makes —
/// model-weight fetches from the HF Hub (M13.2). Substrate-agnostic
/// (Foundation only); the `athena` target resolves config/Keychain
/// into the env *before* a download, this just consumes the standard
/// `*_PROXY` env vars and builds a proxied `URLSession`.
///
/// `URLSession(configuration: .default)` honors only *system* proxy
/// settings on macOS — it ignores `HTTPS_PROXY` etc. So the HF client
/// must be handed a session we configured. With no proxy env set,
/// behavior is unchanged (system settings still apply).
///
/// Loopback (`127.0.0.1`, `::1`, `localhost`) is always added to the
/// bypass list defensively, so local traffic is never proxied.
public enum AthenaProxy {
    /// Case-insensitive env lookup (`HTTPS_PROXY` / `https_proxy`).
    private static func env(_ name: String) -> String? {
        let e = ProcessInfo.processInfo.environment
        if let v = e[name], !v.isEmpty { return v }
        if let v = e[name.lowercased()], !v.isEmpty { return v }
        return nil
    }

    struct ParsedProxy {
        let isSOCKS: Bool
        let host: String
        let port: Int
        let user: String?
        let pass: String?
    }

    /// Validate + decompose a proxy URL. Returns nil for anything
    /// malformed (⇒ caller proceeds proxy-less rather than trapping).
    static func parse(_ raw: String) -> ParsedProxy? {
        let s = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        // Tolerate a bare `host:port` (no scheme) → assume http.
        let withScheme =
            s.contains("://") ? s : "http://" + s
        guard let c = URLComponents(string: withScheme),
            let host = c.host, !host.isEmpty
        else { return nil }
        let scheme = (c.scheme ?? "http").lowercased()
        let isSOCKS = scheme.hasPrefix("socks")
        let port =
            c.port
            ?? (scheme == "https"
                ? 443 : (isSOCKS ? 1080 : 8080))
        guard port > 0, port <= 65535 else { return nil }
        let user = c.user.flatMap {
            $0.removingPercentEncoding ?? $0
        }
        let pass = c.password.flatMap {
            $0.removingPercentEncoding ?? $0
        }
        return ParsedProxy(
            isSOCKS: isSOCKS, host: host, port: port,
            user: (user?.isEmpty ?? true) ? nil : user,
            pass: (pass?.isEmpty ?? true) ? nil : pass)
    }

    /// Hosts that must never be proxied: explicit `NO_PROXY`
    /// (comma/space-separated) plus loopback, always.
    public static func bypassList() -> [String] {
        var out = ["127.0.0.1", "::1", "localhost"]
        if let np = env("NO_PROXY") {
            for tok in np.split(whereSeparator: {
                $0 == "," || $0 == " "
            }) {
                let h = tok.trimmingCharacters(in: .whitespaces)
                if !h.isEmpty { out.append(h) }
            }
        }
        return out
    }

    /// The effective proxy: `HTTPS_PROXY` > `ALL_PROXY` >
    /// `HTTP_PROXY` (HF Hub is HTTPS, so the secure var wins).
    static func effective() -> ParsedProxy? {
        for name in ["HTTPS_PROXY", "ALL_PROXY", "HTTP_PROXY"] {
            if let v = env(name), let p = parse(v) { return p }
        }
        return nil
    }

    /// A `URLSession` honoring the resolved proxy (incl. Basic auth
    /// from the URL userinfo) and the bypass list. No proxy env ⇒ a
    /// plain `.default` session (system settings still apply).
    public static func proxiedURLSession() -> URLSession {
        let cfg = URLSessionConfiguration.default
        guard let p = effective() else {
            return URLSession(configuration: cfg)
        }
        var dict: [AnyHashable: Any] = [
            kCFNetworkProxiesExceptionsList as String:
                bypassList()
        ]
        if p.isSOCKS {
            dict[kCFNetworkProxiesSOCKSEnable as String] = 1
            dict[kCFNetworkProxiesSOCKSProxy as String] = p.host
            dict[kCFNetworkProxiesSOCKSPort as String] = p.port
        } else {
            dict[kCFNetworkProxiesHTTPEnable as String] = 1
            dict[kCFNetworkProxiesHTTPProxy as String] = p.host
            dict[kCFNetworkProxiesHTTPPort as String] = p.port
            #if os(macOS)
                dict[kCFNetworkProxiesHTTPSEnable as String] = 1
                dict[kCFNetworkProxiesHTTPSProxy as String] = p.host
                dict[kCFNetworkProxiesHTTPSPort as String] = p.port
            #endif
        }
        if let u = p.user, let pw = p.pass {
            dict[kCFProxyUsernameKey as String] = u
            dict[kCFProxyPasswordKey as String] = pw
        }
        cfg.connectionProxyDictionary = dict
        return URLSession(configuration: cfg)
    }

    /// Redacted one-line description for `proxy status` / `doctor`
    /// (NEVER prints the password). nil ⇒ no proxy configured.
    public static func describe() -> String? {
        guard let p = effective() else { return nil }
        let scheme = p.isSOCKS ? "socks" : "http(s)"
        let auth = (p.user != nil) ? " (auth: \(p.user!):***)" : ""
        return "\(scheme)://\(p.host):\(p.port)\(auth)"
    }
}
