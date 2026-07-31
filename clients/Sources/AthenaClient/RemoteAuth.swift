import ArgumentParser
import Foundation

#if canImport(FoundationNetworking)
    import FoundationNetworking
#endif
#if canImport(Darwin)
    import Darwin
#elseif canImport(Glibc)
    import Glibc
#endif

// M17.2 — remote RBAC administration. The portable `athena auth …`
// drives the M16.4 `/api/users` `/api/tokens` `/api/roles` surface
// over HTTP. The SERVER is authoritative for RBAC (users.read /
// users.admin / tokens.admin gating, `RBAC.canGrant` vs the caller,
// last-admin protection) — the client adds NO trust and only
// surfaces the server's 400/403/404 as exit status. On macOS the
// SAME `auth user/role/token` verbs run LOCALLY (offline SQLite) and
// route here when `--host` is off-box (`DaemonOptions.isRemote`).
// NOTE: `user add` sends the password in the request body; this
// rides the standing no-TLS caveat (front a TLS proxy for untrusted
// links) — the server hashes; nothing secret is logged.

public enum RemoteAuth {
    private struct UserDTO: Decodable {
        let username: String
        let roles: [String]
    }
    private struct UsersResp: Decodable { let users: [UserDTO] }
    private struct RoleEntry: Decodable {
        let role: String
        let permissions: [String]
    }
    private struct RolesResp: Decodable { let roles: [RoleEntry] }
    private struct TokenDTO: Decodable {
        let username: String
        let scope: [String]?
        let hash_prefix: String
        let label: String?
        let expires: Double?
    }

    /// Human-readable expiry status for a token's `expires` epoch
    /// (M36.2). Mirrors the offline `auth list` rendering.
    private static func expiryNote(_ expires: Double?) -> String {
        guard let expires else { return "no-expiry" }
        let now = Date().timeIntervalSince1970
        if expires <= now { return "EXPIRED" }
        let days = Int((expires - now) / 86400)
        return days >= 1 ? "exp \(days)d" : "exp <1d"
    }
    private struct TokensResp: Decodable { let tokens: [TokenDTO] }
    private struct MintResp: Decodable {
        let user: String
        let scope: [String]?
        let token: String
        let hash_prefix: String
    }

    private static func fail(_ code: Int, _ data: Data) throws {
        HTTPClient.printJSON(data)
        if code >= 400 { throw ExitCode.failure }
    }

    /// Percent-escape a path segment (server validates names; this
    /// just keeps an odd char from breaking the URL).
    private static func enc(_ s: String) -> String {
        s.addingPercentEncoding(
            withAllowedCharacters: .urlPathAllowed) ?? s
    }

    // MARK: users

    public static func usersList(_ d: DaemonOptions) async throws {
        let (code, data): (Int, Data)
        do {
            (code, data) = try await HTTPClient.send(
                "GET", d.base + "/api/users", key: d.authKey)
        } catch { HTTPClient.noDaemon(d, error) }
        guard code < 400,
            let r = try? JSONDecoder().decode(
                UsersResp.self, from: data)
        else { return try fail(code, data) }
        if r.users.isEmpty {
            print("no users")
            return
        }
        for u in r.users {
            let rs =
                u.roles.isEmpty
                ? "(no roles)" : u.roles.joined(separator: ", ")
            print("\(u.username)\t[\(rs)]")
        }
    }

    public static func userCreate(
        _ d: DaemonOptions, username: String, password: String,
        role: String
    ) async throws {
        let body = try JSONSerialization.data(withJSONObject: [
            "username": username, "password": password, "role": role,
        ])
        let (code, data): (Int, Data)
        do {
            (code, data) = try await HTTPClient.send(
                "POST", d.base + "/api/users", body: body,
                key: d.authKey)
        } catch { HTTPClient.noDaemon(d, error) }
        guard code < 400,
            let u = try? JSONDecoder().decode(
                UserDTO.self, from: data)
        else { return try fail(code, data) }
        print(
            "user '\(u.username)' saved with role(s) "
                + "[\(u.roles.joined(separator: ", "))]")
    }

    public static func userDelete(
        _ d: DaemonOptions, username: String
    ) async throws {
        let (code, data): (Int, Data)
        do {
            (code, data) = try await HTTPClient.send(
                "DELETE", d.base + "/api/users/\(enc(username))",
                key: d.authKey)
        } catch { HTTPClient.noDaemon(d, error) }
        try fail(code, data)
    }

    /// ADR 041 — set (`budget`) or clear (`clear`) one user's per-period token
    /// budget. A null `token_budget` is what clears it, so the body carries an
    /// explicit NSNull rather than omitting the key.
    public static func userBudget(
        _ d: DaemonOptions, username: String, budget: Int?, clear: Bool
    ) async throws {
        let value: Any = budget.map { $0 as Any } ?? NSNull()
        let body = try JSONSerialization.data(
            withJSONObject: ["token_budget": value])
        let (code, data): (Int, Data)
        do {
            (code, data) = try await HTTPClient.send(
                "PUT", d.base + "/api/users/\(enc(username))/budget",
                body: body, key: d.authKey)
        } catch { HTTPClient.noDaemon(d, error) }
        guard code < 400 else { return try fail(code, data) }
        if clear {
            print(
                "budget override cleared for '\(username)' "
                    + "(inherits the configured token_budget)")
        } else {
            let n = budget ?? 0
            print(
                "budget for '\(username)' set to \(n) tokens/period"
                    + (n == 0 ? " (unlimited)" : ""))
        }
    }

    // MARK: roles

    public static func roleGrant(
        _ d: DaemonOptions, username: String, role: String
    ) async throws {
        let (code, data): (Int, Data)
        do {
            (code, data) = try await HTTPClient.send(
                "POST",
                d.base
                    + "/api/users/\(enc(username))/roles/\(enc(role))",
                key: d.authKey)
        } catch { HTTPClient.noDaemon(d, error) }
        guard code < 400 else { return try fail(code, data) }
        print("granted '\(role)' to '\(username)'")
    }

    public static func roleRevoke(
        _ d: DaemonOptions, username: String, role: String
    ) async throws {
        let (code, data): (Int, Data)
        do {
            (code, data) = try await HTTPClient.send(
                "DELETE",
                d.base
                    + "/api/users/\(enc(username))/roles/\(enc(role))",
                key: d.authKey)
        } catch { HTTPClient.noDaemon(d, error) }
        guard code < 400 else { return try fail(code, data) }
        print("revoked '\(role)' from '\(username)'")
    }

    public static func rolesList(_ d: DaemonOptions) async throws {
        let (code, data): (Int, Data)
        do {
            (code, data) = try await HTTPClient.send(
                "GET", d.base + "/api/roles", key: d.authKey)
        } catch { HTTPClient.noDaemon(d, error) }
        guard code < 400,
            let r = try? JSONDecoder().decode(
                RolesResp.self, from: data)
        else { return try fail(code, data) }
        for e in r.roles {
            print(
                "\(e.role)\t["
                    + e.permissions.joined(separator: ", ") + "]")
        }
    }

    // MARK: tokens

    public static func tokensList(_ d: DaemonOptions) async throws {
        let (code, data): (Int, Data)
        do {
            (code, data) = try await HTTPClient.send(
                "GET", d.base + "/api/tokens", key: d.authKey)
        } catch { HTTPClient.noDaemon(d, error) }
        guard code < 400,
            let r = try? JSONDecoder().decode(
                TokensResp.self, from: data)
        else { return try fail(code, data) }
        if r.tokens.isEmpty {
            print("no tokens")
            return
        }
        for t in r.tokens {
            let scope =
                t.scope.map { "[\($0.joined(separator: ","))]" }
                ?? "(full)"
            print(
                "\(t.username)\t\(scope)\tsha256:\(t.hash_prefix)…"
                    + "\t\(Self.expiryNote(t.expires))"
                    + (t.label.map { "\t\($0)" } ?? ""))
        }
        print("\(r.tokens.count) token(s)")
    }

    public static func tokenCreate(
        _ d: DaemonOptions, user: String, roles: [String],
        label: String?, ttlSecs: Int? = nil
    ) async throws {
        var body: [String: Any] = ["user": user]
        if !roles.isEmpty { body["role"] = roles }
        if let label { body["label"] = label }
        if let ttlSecs { body["ttl_secs"] = ttlSecs }
        let payload = try JSONSerialization.data(
            withJSONObject: body)
        let (code, data): (Int, Data)
        do {
            (code, data) = try await HTTPClient.send(
                "POST", d.base + "/api/tokens", body: payload,
                key: d.authKey)
        } catch { HTTPClient.noDaemon(d, error) }
        guard code < 400,
            let m = try? JSONDecoder().decode(
                MintResp.self, from: data)
        else { return try fail(code, data) }
        let scopeNote =
            m.scope.map {
                " scoped to [\($0.joined(separator: ", "))]"
            } ?? " (inherits \(m.user)'s roles)"
        print(
            """
            token for '\(m.user)'\(scopeNote) \
            (SAVE NOW — shown once, not stored):

              \(m.token)

            use:  Authorization: Bearer \(m.token)
            """)
    }

    public static func tokenRotate(
        _ d: DaemonOptions, prefix: String, ttlSecs: Int? = nil
    ) async throws {
        var body: [String: Any] = [:]
        if let ttlSecs { body["ttl_secs"] = ttlSecs }
        let payload = try JSONSerialization.data(withJSONObject: body)
        let (code, data): (Int, Data)
        do {
            (code, data) = try await HTTPClient.send(
                "POST",
                d.base + "/api/tokens/\(enc(prefix))/rotate",
                body: payload, key: d.authKey)
        } catch { HTTPClient.noDaemon(d, error) }
        guard code < 400,
            let m = try? JSONDecoder().decode(MintResp.self, from: data)
        else { return try fail(code, data) }
        let scopeNote =
            m.scope.map {
                " scoped to [\($0.joined(separator: ", "))]"
            } ?? " (inherits \(m.user)'s roles)"
        print(
            """
            rotated token for '\(m.user)'\(scopeNote) \
            — old revoked (SAVE NOW — shown once, not stored):

              \(m.token)

            use:  Authorization: Bearer \(m.token)
            """)
    }

    public static func tokenDelete(
        _ d: DaemonOptions, prefix: String
    ) async throws {
        let (code, data): (Int, Data)
        do {
            (code, data) = try await HTTPClient.send(
                "DELETE", d.base + "/api/tokens/\(enc(prefix))",
                key: d.authKey)
        } catch { HTTPClient.noDaemon(d, error) }
        try fail(code, data)
    }

    /// Resolve a password WITHOUT ever accepting it on argv (ADR 005 /
    /// audit B2). Precedence: `--password-stdin` (one line) >
    /// `$ATHENA_PASSWORD` > an interactive no-echo prompt with
    /// confirmation. Shared by the portable and macOS-overload `user add`.
    public static func resolvePassword(stdin: Bool) -> String {
        if stdin {
            var line = readLine(strippingNewline: true) ?? ""
            if line.hasSuffix("\r") { line.removeLast() }
            return line
        }
        if let env = ProcessInfo.processInfo.environment[
            "ATHENA_PASSWORD"], !env.isEmpty
        {
            return env
        }
        let a = String(cString: getpass("password: "))
        let b = String(cString: getpass("confirm:  "))
        guard a == b else {
            FailableExit.die("error: passwords do not match")
        }
        return a
    }
}

// MARK: - Portable command structs (remote-only)

// Distinct Swift type names: the monorepo `athena` target imports
// AthenaClient AND defines its own `AuthUser`/`AuthList`/… (the local
// offline verbs), so a shared `AuthUser` here would make those
// references ambiguous. UX is unchanged — only `commandName` shows.

public struct CAuthUser: AsyncParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "user",
        abstract: "Manage accounts on a daemon (username + roles).",
        subcommands: [
            CAuthUserAdd.self, CAuthUserList.self,
            CAuthUserRemove.self,
        ])
    public init() {}
}

public struct CAuthUserAdd: AsyncParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "add",
        abstract: "Create an account (server hashes the password).")
    @OptionGroup public var daemon: DaemonOptions
    @Argument(help: "Username.") public var username: String
    @Flag(
        name: .customLong("password-stdin"),
        help:
            "Read the password from stdin (one line) instead of prompting. Else set ATHENA_PASSWORD. (ADR 005 — never on argv.)"
    )
    public var passwordStdin = false
    @Option(help: "Initial role (default: member).")
    public var role: String = "member"
    public init() {}
    public func run() async throws {
        let pw = RemoteAuth.resolvePassword(stdin: passwordStdin)
        try await RemoteAuth.userCreate(
            daemon, username: username, password: pw, role: role)
    }
}

public struct CAuthUserList: AsyncParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "list", abstract: "List accounts and roles.")
    @OptionGroup public var daemon: DaemonOptions
    public init() {}
    public func run() async throws {
        try await RemoteAuth.usersList(daemon)
    }
}

public struct CAuthUserRemove: AsyncParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "rm",
        abstract: "Delete an account (cascades roles + tokens).")
    @OptionGroup public var daemon: DaemonOptions
    @Argument(help: "Username.") public var username: String
    public init() {}
    public func run() async throws {
        try await RemoteAuth.userDelete(daemon, username: username)
    }
}

public struct CAuthRole: AsyncParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "role",
        abstract: "Grant/revoke a role; list the role catalog.",
        subcommands: [
            CAuthRoleGrant.self, CAuthRoleRevoke.self,
            CAuthRoleList.self,
        ])
    public init() {}
}

public struct CAuthRoleGrant: AsyncParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "grant", abstract: "Grant ROLE to USER.")
    @OptionGroup public var daemon: DaemonOptions
    @Argument(help: "Username.") public var user: String
    @Argument(help: "Role to grant.") public var role: String
    public init() {}
    public func run() async throws {
        try await RemoteAuth.roleGrant(
            daemon, username: user, role: role)
    }
}

public struct CAuthRoleRevoke: AsyncParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "revoke", abstract: "Revoke ROLE from USER.")
    @OptionGroup public var daemon: DaemonOptions
    @Argument(help: "Username.") public var user: String
    @Argument(help: "Role to revoke.") public var role: String
    public init() {}
    public func run() async throws {
        try await RemoteAuth.roleRevoke(
            daemon, username: user, role: role)
    }
}

public struct CAuthRoleList: AsyncParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "list",
        abstract: "List the daemon's role → permission catalog.")
    @OptionGroup public var daemon: DaemonOptions
    public init() {}
    public func run() async throws {
        try await RemoteAuth.rolesList(daemon)
    }
}

public struct CAuthToken: AsyncParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "token",
        abstract: "Mint bearer tokens (hash-only at rest).",
        subcommands: [CAuthTokenAdd.self])
    public init() {}
}

public struct CAuthTokenAdd: AsyncParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "add",
        abstract: "Mint a bearer token for a user (shown once).")
    @OptionGroup public var daemon: DaemonOptions
    @Option(help: "Owning username (required).")
    public var user: String
    @Option(
        help:
            "Scope to a role subset (repeatable; default: the user's full set)."
    )
    public var role: [String] = []
    @Option(help: "Optional label (shown by `auth list`).")
    public var label: String?
    public init() {}
    public func run() async throws {
        try await RemoteAuth.tokenCreate(
            daemon, user: user, roles: role, label: label)
    }
}

public struct CAuthList: AsyncParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "list",
        abstract: "List tokens: user, scope, hash prefix.")
    @OptionGroup public var daemon: DaemonOptions
    public init() {}
    public func run() async throws {
        try await RemoteAuth.tokensList(daemon)
    }
}

public struct CAuthRemove: AsyncParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "rm",
        abstract: "Remove tokens whose hash hex starts with PREFIX.")
    @OptionGroup public var daemon: DaemonOptions
    @Argument(help: "Hash hex prefix (>= 6 chars).")
    public var prefix: String
    public init() {}
    public func run() async throws {
        try await RemoteAuth.tokenDelete(daemon, prefix: prefix)
    }
}
