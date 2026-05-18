import ArgumentParser
import AthenaClient
import AthenaCore
import AthenaDeploy
import AthenaStore
import Crypto
import Darwin
import Foundation

/// `athena auth …` — manage RBAC subjects (M15.2). Users hold roles;
/// bearer tokens belong to a user and may scope-narrow to a role
/// subset. API keys are generated here, shown ONCE, and persisted
/// only as `sha256:<hex>` — no secret is ever written to disk.
struct Auth: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "auth",
        abstract: "Manage RBAC users, roles, and bearer tokens.",
        subcommands: [
            AuthUser.self, AuthRole.self, AuthToken.self,
            AuthList.self, AuthRemove.self,
            AuthLogin.self, AuthLogout.self, AuthStatus.self,
        ])
}

// Offline CLI grantor = implicitly admin (holds every permission),
// so the escalation guard always passes for a valid role here; the
// API path (M16) supplies the real caller's permission set. We still
// route grants through `RBAC.canGrant` so an unknown role is refused
// the same way everywhere (fail-closed).
private let cliGrantorPermissions = Set(Permission.allCases)

/// The daemon's SQLite path: configured `data_dir` (or ~/.athena) +
/// athena.sqlite. Works OFFLINE so the first admin can be created
/// before the auth-protected daemon exists.
private func storeDBPath(_ override: String? = nil) -> URL {
    let dir: URL
    if let override, !override.isEmpty {
        dir = URL(
            fileURLWithPath:
                (override as NSString).expandingTildeInPath,
            isDirectory: true)
    } else if let cfg = try? AthenaConfig.parse(
        file: ConfigEditor.resolvePath(nil)),
        let d = cfg.dataDir, !d.isEmpty
    {
        dir = URL(
            fileURLWithPath: (d as NSString).expandingTildeInPath,
            isDirectory: true)
    } else {
        dir = AthenaEnv.userHome()
            .appendingPathComponent(".athena", isDirectory: true)
    }
    return dir.appendingPathComponent("athena.sqlite")
}

private func openStore(_ dataDir: String?) -> AthenaStore {
    do {
        return try AthenaStore(path: storeDBPath(dataDir))
    } catch {
        FailableExit.die("error: cannot open store: \(error)")
    }
}

private func requireValidRole(_ role: String) {
    guard RBAC.isValidRole(role) else {
        FailableExit.die(
            "error: unknown role '\(role)' — valid: "
                + RBAC.roleNames.joined(separator: ", "))
    }
}

/// Refuse an op that would strip the last `admin`-role holder, which
/// would lock the appliance out of its own administration.
private func guardLastAdmin(
    _ db: AthenaStore, losing username: String
) async {
    let admins = await db.usersWithRole("admin")
    if admins == [username] {
        FailableExit.die(
            "error: '\(username)' is the only admin — refusing "
                + "(grant admin to another user first)")
    }
}

// MARK: - Users (username/password account + role grants)

struct AuthUser: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "user",
        abstract: "Manage accounts (username/password + roles).",
        subcommands: [
            AuthUserAdd.self, AuthUserList.self, AuthUserRemove.self,
        ])
}

struct AuthUserAdd: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "add",
        abstract: "Create/replace an account (prompts, no echo).")
    @Argument(help: "Username.") var username: String
    @Option(help: "Password (omit to prompt; avoid in shell history).")
    var password: String?
    @Option(help: "Initial role to grant (default: member).")
    var role: String = "member"
    @Option(help: "Data dir (default: configured / ~/.athena).")
    var dataDir: String?

    func run() async throws {
        requireValidRole(role)
        let pw: String
        if let password, !password.isEmpty {
            pw = password
        } else if isatty(0) == 0 {
            pw = (readLine() ?? "")
        } else {
            let a = String(cString: getpass("password: "))
            let b = String(cString: getpass("confirm:  "))
            guard a == b else {
                FailableExit.die("error: passwords do not match")
            }
            pw = a
        }
        guard pw.count >= 8 else {
            FailableExit.die("error: password must be >= 8 chars")
        }
        let salt = Passwords.randomSalt()
        let hash = Passwords.derive(
            password: pw, salt: salt,
            iters: Passwords.defaultIterations)
        let db = openStore(dataDir)
        do {
            try await db.putUser(
                username: username, salt: salt, hash: hash,
                iters: Passwords.defaultIterations)
            try await db.grantRole(
                username: username, role: role)
        } catch {
            FailableExit.die("error: \(error)")
        }
        print(
            "user '\(username)' saved with role '\(role)' "
                + "(\(storeDBPath(dataDir).path))")
    }
}

struct AuthUserList: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "list", abstract: "List accounts and roles.")
    @Option(help: "Data dir (default: configured / ~/.athena).")
    var dataDir: String?
    func run() async throws {
        guard let db = try? AthenaStore(path: storeDBPath(dataDir))
        else {
            print("no store at \(storeDBPath(dataDir).path)")
            return
        }
        let users = await db.listUsers()
        if users.isEmpty { print("no users"); return }
        for u in users {
            let roles = await db.rolesForUser(username: u)
            let rs =
                roles.isEmpty
                ? "(no roles)" : roles.joined(separator: ", ")
            print("\(u)\t[\(rs)]")
        }
    }
}

struct AuthUserRemove: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "rm",
        abstract: "Delete an account (cascades roles + tokens).")
    @Argument(help: "Username.") var username: String
    @Option(help: "Data dir (default: configured / ~/.athena).")
    var dataDir: String?
    func run() async throws {
        guard let db = try? AthenaStore(path: storeDBPath(dataDir))
        else {
            FailableExit.die(
                "error: no store at \(storeDBPath(dataDir).path)")
        }
        await guardLastAdmin(db, losing: username)
        let ok = await db.deleteUser(username: username)
        print(ok ? "removed '\(username)'" : "no such user")
    }
}

// MARK: - Roles (grant / revoke)

struct AuthRole: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "role",
        abstract: "Grant or revoke a role on a user.",
        subcommands: [AuthRoleGrant.self, AuthRoleRevoke.self])
}

struct AuthRoleGrant: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "grant", abstract: "Grant ROLE to USER.")
    @Argument(help: "Username.") var user: String
    @Argument(help: "Role to grant.") var role: String
    @Option(help: "Data dir (default: configured / ~/.athena).")
    var dataDir: String?
    func run() async throws {
        requireValidRole(role)
        guard
            RBAC.canGrant(
                role: role,
                grantorPermissions: cliGrantorPermissions)
        else {
            FailableExit.die(
                "error: refused — cannot grant a role conferring "
                    + "permissions you do not hold")
        }
        let db = openStore(dataDir)
        guard await db.getUser(username: user) != nil else {
            FailableExit.die("error: no such user '\(user)'")
        }
        do {
            try await db.grantRole(username: user, role: role)
        } catch {
            FailableExit.die("error: \(error)")
        }
        print("granted '\(role)' to '\(user)'")
    }
}

struct AuthRoleRevoke: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "revoke", abstract: "Revoke ROLE from USER.")
    @Argument(help: "Username.") var user: String
    @Argument(help: "Role to revoke.") var role: String
    @Option(help: "Data dir (default: configured / ~/.athena).")
    var dataDir: String?
    func run() async throws {
        let db = openStore(dataDir)
        if role == "admin" {
            await guardLastAdmin(db, losing: user)
        }
        let ok = await db.revokeRole(username: user, role: role)
        print(
            ok
                ? "revoked '\(role)' from '\(user)'"
                : "no such grant ('\(user)' had no '\(role)')")
    }
}

// MARK: - Tokens (bearer keys bound to a user, optional scope)

struct AuthToken: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "token",
        abstract: "Manage bearer tokens (hash-only at rest).",
        subcommands: [AuthTokenAdd.self])
}

struct AuthTokenAdd: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "add",
        abstract: "Mint a bearer token for a user (shown once).")
    @Option(help: "Owning username (required).") var user: String
    @Option(
        help:
            "Scope the token to a role subset (repeatable; default: the user's full role set)."
    )
    var role: [String] = []
    @Option(help: "Optional label (shown by `auth list`).")
    var label: String?
    @Option(help: "Data dir (default: configured / ~/.athena).")
    var dataDir: String?

    func run() async throws {
        for r in role { requireValidRole(r) }
        let db = openStore(dataDir)
        guard await db.getUser(username: user) != nil else {
            FailableExit.die("error: no such user '\(user)'")
        }
        // Single key-generation path, shared with POST /api/tokens.
        let (key, hash) = AuthConfig.mintToken()
        let scoped = role.isEmpty ? nil : role
        do {
            try await db.putToken(
                hash: hash, username: user,
                scopedRoles: scoped, label: label)
        } catch {
            FailableExit.die("error: \(error)")
        }
        let scopeNote =
            scoped.map { " scoped to [\($0.joined(separator: ", "))]" }
            ?? " (inherits \(user)'s roles)"
        print(
            """
            token for '\(user)'\(scopeNote) \
            (SAVE NOW — shown once, not stored):

              \(key)

            stored as a hash in \(storeDBPath(dataDir).path)
            use:  Authorization: Bearer \(key)
            """)
    }
}

// MARK: - Token listing / removal

/// 64-char hex → bytes (for `auth rm <hashprefix>` reconstruction).
private func hexBytes(_ s: Substring) -> [UInt8]? {
    guard s.count == 64 else { return nil }
    var out: [UInt8] = []
    var i = s.startIndex
    while i < s.endIndex {
        let j = s.index(i, offsetBy: 2)
        guard let b = UInt8(s[i..<j], radix: 16) else { return nil }
        out.append(b)
        i = j
    }
    return out
}

struct AuthList: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "list",
        abstract: "List tokens: user, scope, hash prefix (no secrets).")
    @Option(help: "Data dir (default: configured / ~/.athena).")
    var dataDir: String?

    func run() async throws {
        guard let db = try? AthenaStore(path: storeDBPath(dataDir))
        else {
            print("no store at \(storeDBPath(dataDir).path)")
            return
        }
        let toks = await db.listTokens()
        if toks.isEmpty { print("no tokens"); return }
        for t in toks {
            let scope =
                t.scoped.map { "[\($0.joined(separator: ","))]" }
                ?? "(full)"
            print(
                "\(t.username)\t\(scope)\tsha256:\(t.hex.prefix(12))…"
                    + (t.label.map { "\t\($0)" } ?? ""))
        }
        print("\(toks.count) token(s) — \(storeDBPath(dataDir).path)")
    }
}

struct AuthRemove: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "rm",
        abstract: "Remove tokens whose hash hex starts with PREFIX.")
    @Argument(help: "Hash hex prefix (>= 6 chars).") var prefix: String
    @Option(help: "Data dir (default: configured / ~/.athena).")
    var dataDir: String?

    func run() async throws {
        guard prefix.count >= 6 else {
            FailableExit.die("error: prefix must be >= 6 hex chars")
        }
        guard let db = try? AthenaStore(path: storeDBPath(dataDir))
        else {
            FailableExit.die(
                "error: no store at \(storeDBPath(dataDir).path)")
        }
        let matches = await db.listTokens().filter {
            $0.hex.hasPrefix(prefix)
        }
        if matches.isEmpty {
            FailableExit.die("error: no token matched \(prefix)")
        }
        var removed = 0
        for m in matches {
            if let bytes = hexBytes(Substring(m.hex)),
                await db.deleteToken(hash: Data(bytes))
            {
                removed += 1
            }
        }
        print("removed \(removed) token(s)")
    }
}
