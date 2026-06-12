import ArgumentParser
import AthenaClient
import AthenaCore
import AthenaDeploy
import Darwin
import Foundation

/// Resolves the egress proxy into the standard `*_PROXY` env vars
/// before any model download, so `AthenaProxy` (AthenaCore) can build
/// a proxied `URLSession`. Precedence: an operator-set env var always
/// wins; otherwise the `athena.toml` `[network]` keys fill it in.
/// Keychain proxy creds (`athena proxy login`) are spliced in last
/// and take precedence over any inline `user:pass@` in the URL. M13.2.
enum ProxyEnv {
    private static func setIfAbsent(_ name: String, _ val: String?) {
        guard let val, !val.isEmpty else { return }
        let e = ProcessInfo.processInfo.environment
        if e[name] != nil || e[name.lowercased()] != nil { return }
        setenv(name, val, 1)
    }

    private static func withCreds(
        _ urlStr: String, _ u: String, _ p: String
    ) -> String? {
        let s =
            urlStr.contains("://") ? urlStr : "http://" + urlStr
        guard var c = URLComponents(string: s) else { return nil }
        c.user = u
        c.password = p
        return c.string
    }

    /// Idempotent; safe to call from `serve`/`pull`/`convert` and
    /// `proxy status`. Mutates only this process's env.
    static func applyConfigAndAuth() {
        let cfg = try? AthenaConfig.parse(
            file: ConfigEditor.resolvePath(nil))
        setIfAbsent("HTTPS_PROXY", cfg?.httpsProxy)
        setIfAbsent("HTTP_PROXY", cfg?.httpProxy)
        setIfAbsent("ALL_PROXY", cfg?.allProxy)
        setIfAbsent("NO_PROXY", cfg?.noProxy)

        guard let (u, p) = ProxyAuth.read() else { return }
        let e = ProcessInfo.processInfo.environment
        for name in ["HTTPS_PROXY", "ALL_PROXY", "HTTP_PROXY"] {
            guard let cur = e[name] ?? e[name.lowercased()],
                !cur.isEmpty
            else { continue }
            if let merged = withCreds(cur, u, p) {
                setenv(name, merged, 1)
            }
        }
    }
}

/// `athena proxy` — manage egress-proxy auth (M13.2). The proxy
/// *endpoint* is configured via the standard `HTTPS_PROXY` /
/// `HTTP_PROXY` / `ALL_PROXY` / `NO_PROXY` env vars or the
/// `athena.toml` `[network]` section (env wins). This command manages
/// only the proxy *credentials*, stored in the macOS Keychain.
struct Proxy: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "proxy",
        abstract: "Manage egress-proxy credentials (Keychain).",
        subcommands: [
            ProxyLogin.self, ProxyLogout.self, ProxyStatus.self,
        ])
}

struct ProxyLogin: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "login",
        abstract:
            "Store egress-proxy credentials (macOS Keychain).")
    @Option(help: "Proxy username.") var user: String?
    @Flag(
        name: .customLong("password-stdin"),
        help: "Read the proxy password from stdin (one line) instead of prompting. Else set ATHENA_PASSWORD. (ADR 005 — never on argv.)")
    var passwordStdin = false

    func run() async throws {
        let u: String
        if let user, !user.isEmpty {
            u = user
        } else {
            u = String(cString: getpass("proxy username: "))
        }
        guard !u.isEmpty else {
            FailableExit.die("error: empty username")
        }
        let p = PasswordInput.resolve(
            stdin: passwordStdin, confirmNew: false,
            prompt: "proxy password: ")
        do {
            try ProxyAuth.store(user: u, pass: p)
        } catch {
            FailableExit.die("error: \(error)")
        }
        print("stored egress-proxy credentials (not echoed)")
    }
}

struct ProxyLogout: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "logout",
        abstract: "Remove stored egress-proxy credentials.")
    func run() async throws {
        print(
            ProxyAuth.remove()
                ? "removed stored egress-proxy credentials"
                : "no stored egress-proxy credentials")
    }
}

struct ProxyStatus: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "status",
        abstract:
            "Report the resolved proxy (no secret printed).")
    func run() async throws {
        ProxyEnv.applyConfigAndAuth()
        let proxy = AthenaProxy.describe() ?? "none (direct)"
        let bypass = AthenaProxy.bypassList().joined(
            separator: ", ")
        print("egress proxy: \(proxy)")
        print("bypass:       \(bypass)")
        print("credentials:  \(ProxyAuth.source())")
    }
}
