import ArgumentParser
import Foundation

#if canImport(Darwin)
    import Darwin
#elseif canImport(Glibc)
    import Glibc
#endif

// Client-side credential store (login/logout/status) — manages THIS
// host's stored bearer key for talking to a (local or remote) daemon.
// Portable client surface (M14.2a). Server-side `auth add/list/rm` and
// `auth user …` stay in the daemon binary.

/// `athena auth …` on the portable client: this host's stored bearer
/// key (login/logout/status) PLUS remote RBAC administration of a
/// daemon (M17.2 — user/role/token over `/api/*`; the server enforces
/// RBAC). On macOS the same `auth user/role/token` are the LOCAL
/// offline verbs and route to the remote ones when `--host` is
/// off-box.
public struct AuthClient: AsyncParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "auth",
        abstract: "Stored daemon key + remote RBAC administration.",
        subcommands: [
            AuthLogin.self, AuthLogout.self, AuthStatus.self,
            CAuthUser.self, CAuthRole.self, CAuthToken.self,
            CAuthList.self, CAuthRemove.self,
        ])
    public init() {}
}

public struct AuthLogin: AsyncParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "login",
        abstract: "Store a bearer key for a daemon (macOS Keychain).")
    @Option(help: "Daemon host.") public var host: String =
        "127.0.0.1"
    @Option(help: "Daemon port.")
    public var port: Int = athenaDefaultPort
    @Option(help: "Key (omit to read stdin / prompt, no echo).")
    public var key: String?
    public init() {}

    public func run() async throws {
        let k: String
        if let key, !key.isEmpty {
            k = key
        } else if isatty(0) == 0 {
            k = (readLine() ?? "").trimmingCharacters(
                in: .whitespacesAndNewlines)
        } else {
            k = String(cString: getpass("daemon bearer key: "))
        }
        guard !k.isEmpty else {
            FailableExit.die("error: empty key")
        }
        do {
            try Credentials.store(k, host: host, port: port)
        } catch {
            FailableExit.die("error: \(error)")
        }
        print("stored key for \(host):\(port) (not echoed)")
    }
}

public struct AuthLogout: AsyncParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "logout",
        abstract: "Remove this host's stored key for a daemon.")
    @Option(help: "Daemon host.") public var host: String =
        "127.0.0.1"
    @Option(help: "Daemon port.")
    public var port: Int = athenaDefaultPort
    public init() {}
    public func run() async throws {
        print(
            Credentials.remove(host: host, port: port)
                ? "removed stored key for \(host):\(port)"
                : "no stored key for \(host):\(port)")
    }
}

public struct AuthStatus: AsyncParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "status",
        abstract: "Report how the daemon key resolves (no secret).")
    @Option(help: "Daemon host.") public var host: String =
        "127.0.0.1"
    @Option(help: "Daemon port.")
    public var port: Int = athenaDefaultPort
    public init() {}
    public func run() async throws {
        let env = ProcessInfo.processInfo.environment["ATHENA_KEY"]
        let source: String
        if let env, !env.isEmpty {
            source = "ATHENA_KEY env"
        } else if Credentials.resolve(
            explicit: nil, host: host, port: port) != nil
        {
            source = "Keychain"
        } else {
            source = "none (set --key / ATHENA_KEY / auth login)"
        }
        print("\(host):\(port) — credential source: \(source)")
    }
}
