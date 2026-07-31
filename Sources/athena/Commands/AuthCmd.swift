import ArgumentParser
import AthenaClient
import AthenaCore
import AthenaDeploy
import AthenaServerKit
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
/// before the auth-protected daemon exists. (Module-internal so the
/// `usage` command's local path reuses it — M27.3.)
func storeDBPath(_ override: String? = nil) -> URL {
    let dir: URL
    if let override, !override.isEmpty {
        dir = URL(
            fileURLWithPath: (override as NSString).expandingTildeInPath,
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
        return try AthenaStore(path: storeDBPath(dataDir), key: StoreKey.resolve())
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
    // NB11 (M66.2): a query failure must NOT be read as "no admins" (which
    // would let the last admin be stripped). usersWithRole now throws on a
    // store error; treat that as refuse (fail closed).
    let admins: [String]
    do {
        admins = try await db.usersWithRole("admin")
    } catch {
        FailableExit.die(
            "error: could not verify the admin set (\(error)) — "
                + "refusing to remove the last admin")
    }
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
            AuthUserAdd.self, AuthUserPasswd.self,
            AuthUserList.self, AuthUserRemove.self,
            AuthUserBudget.self,
        ])
}

/// ADR 041 — per-user token budget. ONE verb, context-keyed: local SQLite when
/// run on the daemon host, the `PUT /api/users/:name/budget` control-plane
/// route when `--host` points off-box.
struct AuthUserBudget: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "budget",
        abstract:
            "Set/clear a user's per-period token budget (0 = unlimited).")
    @Argument(help: "Username.") var username: String
    @Argument(
        help:
            "Tokens allowed per budget period. 0 ⇒ unlimited for this user. Omit with --clear to inherit the configured token_budget default."
    )
    var tokens: Int?
    @Flag(
        name: .long,
        help: "Clear the override so the user inherits the configured default.")
    var clear = false
    @Option(help: "Data dir (default: configured / ~/.athena).")
    var dataDir: String?
    @OptionGroup var daemon: DaemonOptions

    func run() async throws {
        guard clear != (tokens != nil) else {
            FailableExit.die(
                "error: pass either a token count or --clear (not both, "
                    + "not neither)")
        }
        if let tokens, tokens < 0 {
            FailableExit.die("error: budget must be >= 0 (0 = unlimited)")
        }
        let budget: Int? = clear ? nil : tokens
        if daemon.isRemote {
            try await RemoteAuth.userBudget(
                daemon, username: username, budget: budget, clear: clear)
            return
        }
        let db = openStore(dataDir)
        let existed: Bool
        do {
            existed = try await db.setUserBudget(
                username: username, budget: budget)
        } catch {
            FailableExit.die("error: \(error)")
        }
        guard existed else {
            FailableExit.die(
                "error: no such user '\(username)' — create it with "
                    + "`athena auth user add`")
        }
        print(
            budget.map {
                "budget for '\(username)' set to \($0) tokens/period"
                    + ($0 == 0 ? " (unlimited)" : "")
            }
                ?? "budget override cleared for '\(username)' "
                + "(inherits the configured token_budget)")
    }
}

struct AuthUserAdd: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "add",
        abstract: "Create/replace an account (prompts, no echo).")
    @Argument(help: "Username.") var username: String
    @Flag(
        name: .customLong("password-stdin"),
        help:
            "Read the password from stdin (one line) instead of prompting. Else set ATHENA_PASSWORD. (ADR 005 — never on argv.)"
    )
    var passwordStdin = false
    @Option(help: "Initial role to grant (default: member).")
    var role: String = "member"
    @Flag(
        name: .long,
        help:
            "Replace an existing account (resets its password); add refuses to overwrite an existing user without it."
    )
    var force = false
    @Option(help: "Data dir (default: configured / ~/.athena).")
    var dataDir: String?
    @OptionGroup var daemon: DaemonOptions

    func run() async throws {
        if daemon.isRemote {
            let pw = RemoteAuth.resolvePassword(stdin: passwordStdin)
            try await RemoteAuth.userCreate(
                daemon, username: username, password: pw,
                role: role)
            return
        }
        requireValidRole(role)
        let pw = PasswordInput.resolve(
            stdin: passwordStdin, confirmNew: true)
        guard pw.count >= 8 else {
            FailableExit.die("error: password must be >= 8 chars")
        }
        let salt = Passwords.randomSalt()
        let hash = Passwords.derive(
            password: pw, salt: salt,
            iters: Passwords.defaultIterations)
        let db = openStore(dataDir)
        // B5 (M66.2): `putUser` upserts the credential columns, so an `add`
        // against an existing username would silently reset that password.
        // Refuse unless `--force`; point at `passwd` for a roles-preserving
        // reset.
        if await db.getUser(username: username) != nil, !force {
            FailableExit.die(
                "error: user '\(username)' already exists — pass --force "
                    + "to replace it (resets the password), or use "
                    + "`athena auth user passwd \(username)` to reset just "
                    + "the password (roles kept)")
        }
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

/// Offline password recovery: reset an EXISTING account's password
/// without touching its roles/tokens. Local-only by design — the
/// locked-out-admin case has no token; the trust boundary is OS
/// ownership of the data-dir DB.
struct AuthUserPasswd: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "passwd",
        abstract:
            "Reset an existing account's password (roles kept).")
    @Argument(help: "Username.") var username: String
    @Flag(
        name: .customLong("password-stdin"),
        help:
            "Read the new password from stdin (one line) instead of prompting. Else set ATHENA_PASSWORD. (ADR 005 — never on argv.)"
    )
    var passwordStdin = false
    @Option(help: "Data dir (default: configured / ~/.athena).")
    var dataDir: String?
    @OptionGroup var daemon: DaemonOptions

    func run() async throws {
        if daemon.isRemote {
            FailableExit.die(
                "error: `passwd` is offline recovery only — run it "
                    + "on the daemon host. Remotely, an admin can "
                    + "replace the account via `auth user add`.")
        }
        let db = openStore(dataDir)
        guard await db.getUser(username: username) != nil else {
            FailableExit.die(
                "error: no such user '\(username)' — create it "
                    + "with `athena auth user add`")
        }
        let pw = PasswordInput.resolve(
            stdin: passwordStdin, confirmNew: true,
            prompt: "new password: ")
        guard pw.count >= 8 else {
            FailableExit.die("error: password must be >= 8 chars")
        }
        let salt = Passwords.randomSalt()
        let hash = Passwords.derive(
            password: pw, salt: salt,
            iters: Passwords.defaultIterations)
        do {
            try await db.putUser(
                username: username, salt: salt, hash: hash,
                iters: Passwords.defaultIterations)
        } catch {
            FailableExit.die("error: \(error)")
        }
        let roles = await db.rolesForUser(username: username)
        print(
            "password reset for '\(username)' — roles kept: "
                + (roles.isEmpty
                    ? "(none)" : roles.joined(separator: ", "))
                + " (\(storeDBPath(dataDir).path))")
    }
}

struct AuthUserList: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "list", abstract: "List accounts and roles.")
    @Option(help: "Data dir (default: configured / ~/.athena).")
    var dataDir: String?
    @OptionGroup var daemon: DaemonOptions
    func run() async throws {
        if daemon.isRemote {
            try await RemoteAuth.usersList(daemon)
            return
        }
        guard let db = try? AthenaStore(path: storeDBPath(dataDir), key: StoreKey.resolve())
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
    @OptionGroup var daemon: DaemonOptions
    func run() async throws {
        if daemon.isRemote {
            try await RemoteAuth.userDelete(
                daemon, username: username)
            return
        }
        guard let db = try? AthenaStore(path: storeDBPath(dataDir), key: StoreKey.resolve())
        else {
            FailableExit.die(
                "error: no store at \(storeDBPath(dataDir).path)")
        }
        await guardLastAdmin(db, losing: username)
        let ok: Bool
        do {
            ok = try await db.deleteUser(username: username)
        } catch {
            FailableExit.die("error: delete failed: \(error)")
        }
        print(ok ? "removed '\(username)'" : "no such user")
    }
}

// MARK: - Roles (grant / revoke)

struct AuthRole: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "role",
        abstract: "Grant or revoke a role; list the role catalog.",
        subcommands: [
            AuthRoleGrant.self, AuthRoleRevoke.self,
            AuthRoleList.self,
        ])
}

/// `athena auth role list` — the role → permission catalog. Local:
/// the compiled-in `RBAC` catalog (no daemon needed). Off-box: the
/// daemon's `GET /api/roles` (M17.2).
struct AuthRoleList: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "list",
        abstract: "List the role → permission catalog.")
    @OptionGroup var daemon: DaemonOptions
    func run() async throws {
        if daemon.isRemote {
            try await RemoteAuth.rolesList(daemon)
            return
        }
        for r in RBAC.roleNames {
            let perms = (RBAC.catalog[r] ?? [])
                .map(\.rawValue).sorted()
            print("\(r)\t[\(perms.joined(separator: ", "))]")
        }
    }
}

struct AuthRoleGrant: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "grant", abstract: "Grant ROLE to USER.")
    @Argument(help: "Username.") var user: String
    @Argument(help: "Role to grant.") var role: String
    @Option(help: "Data dir (default: configured / ~/.athena).")
    var dataDir: String?
    @OptionGroup var daemon: DaemonOptions
    func run() async throws {
        if daemon.isRemote {
            try await RemoteAuth.roleGrant(
                daemon, username: user, role: role)
            return
        }
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
    @OptionGroup var daemon: DaemonOptions
    func run() async throws {
        if daemon.isRemote {
            try await RemoteAuth.roleRevoke(
                daemon, username: user, role: role)
            return
        }
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
        subcommands: [AuthTokenAdd.self, AuthTokenRotate.self])
}

// `parseTTLSeconds` moved to AthenaDeploy (NB4 / M70.1b) so it is
// unit-testable under `swift test`; the call sites below are unchanged
// (AuthCmd imports AthenaDeploy).

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
    @Option(
        help:
            "Lifetime: 30d / 12h / 90m / 3600s (or bare seconds). Absent ⇒ never expires (subject to token_max_age_days)."
    )
    var ttl: String?
    @Option(help: "Data dir (default: configured / ~/.athena).")
    var dataDir: String?
    @OptionGroup var daemon: DaemonOptions

    func run() async throws {
        let ttlSecs: Int? = try ttl.map {
            guard let s = parseTTLSeconds($0) else {
                FailableExit.die(
                    "error: invalid --ttl '\($0)' (use e.g. 30d, 12h, "
                        + "90m, 3600s, or bare seconds)")
            }
            return s
        }
        if daemon.isRemote {
            try await RemoteAuth.tokenCreate(
                daemon, user: user, roles: role, label: label,
                ttlSecs: ttlSecs)
            return
        }
        for r in role { requireValidRole(r) }
        let db = openStore(dataDir)
        guard await db.getUser(username: user) != nil else {
            FailableExit.die("error: no such user '\(user)'")
        }
        // Single key-generation path, shared with POST /api/tokens.
        let (key, hash) = AuthConfig.mintToken()
        let scoped = role.isEmpty ? nil : role
        let expires = ttlSecs.map {
            Date().timeIntervalSince1970 + Double($0)
        }
        do {
            try await db.putToken(
                hash: hash, username: user,
                scopedRoles: scoped, label: label, expires: expires)
        } catch {
            FailableExit.die("error: \(error)")
        }
        let scopeNote =
            scoped.map { " scoped to [\($0.joined(separator: ", "))]" }
            ?? " (inherits \(user)'s roles)"
        let expiryNote =
            ttlSecs.map { " expires in \($0)s" } ?? " (no expiry)"
        print(
            """
            token for '\(user)'\(scopeNote)\(expiryNote) \
            (SAVE NOW — shown once, not stored):

              \(key)

            stored as a hash in \(storeDBPath(dataDir).path)
            use:  Authorization: Bearer \(key)
            """)
    }
}

struct AuthTokenRotate: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "rotate",
        abstract:
            "Revoke a token (by hash prefix) and reissue a fresh one "
            + "with the same owner/scope/label (shown once).")
    @Argument(help: "Hash hex prefix (>= 6 chars; must match exactly one).")
    var prefix: String
    @Option(
        help:
            "New lifetime: 30d / 12h / 90m / 3600s (or bare seconds). Absent ⇒ the rotated token never expires (the old TTL is not carried)."
    )
    var ttl: String?
    @Option(help: "Data dir (default: configured / ~/.athena).")
    var dataDir: String?
    @OptionGroup var daemon: DaemonOptions

    func run() async throws {
        let ttlSecs: Int? = try ttl.map {
            guard let s = parseTTLSeconds($0) else {
                FailableExit.die(
                    "error: invalid --ttl '\($0)' (use e.g. 30d, 12h, "
                        + "90m, 3600s, or bare seconds)")
            }
            return s
        }
        if daemon.isRemote {
            try await RemoteAuth.tokenRotate(
                daemon, prefix: prefix, ttlSecs: ttlSecs)
            return
        }
        guard prefix.count >= 6 else {
            FailableExit.die("error: prefix must be >= 6 hex chars")
        }
        let db = openStore(dataDir)
        let matches = await db.tokensMatchingHashPrefix(prefix)
        guard matches.count == 1, let m = matches.first,
            let oldBytes = hexBytes(Substring(m.hex))
        else {
            FailableExit.die(
                matches.count > 1
                    ? "error: prefix '\(prefix)' matched "
                        + "\(matches.count) tokens — use a longer prefix"
                    : "error: no token matched '\(prefix)'")
        }
        let (key, hash) = AuthConfig.mintToken()
        let expires = ttlSecs.map {
            Date().timeIntervalSince1970 + Double($0)
        }
        // B4 (M66.2): persist the new token BEFORE revoking the old one,
        // so a putToken failure leaves the holder's existing token working
        // instead of locking them out. Old is revoked only on success.
        do {
            try await db.putToken(
                hash: hash, username: m.username,
                scopedRoles: m.scoped, label: m.label, expires: expires)
        } catch {
            FailableExit.die("error: \(error)")
        }
        _ = await db.deleteToken(hash: Data(oldBytes))
        let scopeNote =
            m.scoped.map { " scoped to [\($0.joined(separator: ", "))]" }
            ?? " (inherits \(m.username)'s roles)"
        let expiryNote =
            ttlSecs.map { " expires in \($0)s" } ?? " (no expiry)"
        print(
            """
            rotated token for '\(m.username)'\(scopeNote)\(expiryNote) \
            — old revoked (SAVE NOW — shown once, not stored):

              \(key)

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
        guard let b = UInt8(s[i ..< j], radix: 16) else { return nil }
        out.append(b)
        i = j
    }
    return out
}

/// Human-readable expiry status for a token's `expires` epoch (M36.2).
/// Shared by the offline `auth list`; the remote path renders server-side.
func tokenExpiryNote(_ expires: Double?) -> String {
    guard let expires else { return "no-expiry" }
    let now = Date().timeIntervalSince1970
    if expires <= now { return "EXPIRED" }
    let days = Int((expires - now) / 86400)
    return days >= 1 ? "exp \(days)d" : "exp <1d"
}

struct AuthList: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "list",
        abstract: "List tokens: user, scope, hash prefix (no secrets).")
    @Option(help: "Data dir (default: configured / ~/.athena).")
    var dataDir: String?
    @OptionGroup var daemon: DaemonOptions

    func run() async throws {
        if daemon.isRemote {
            try await RemoteAuth.tokensList(daemon)
            return
        }
        guard let db = try? AthenaStore(path: storeDBPath(dataDir), key: StoreKey.resolve())
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
                "\(t.username)\t\(scope)\tsha256:\(t.hashPrefix)…"
                    + "\t\(tokenExpiryNote(t.expires))"
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
    @OptionGroup var daemon: DaemonOptions

    func run() async throws {
        if daemon.isRemote {
            try await RemoteAuth.tokenDelete(daemon, prefix: prefix)
            return
        }
        guard prefix.count >= 6 else {
            FailableExit.die("error: prefix must be >= 6 hex chars")
        }
        guard let db = try? AthenaStore(path: storeDBPath(dataDir), key: StoreKey.resolve())
        else {
            FailableExit.die(
                "error: no store at \(storeDBPath(dataDir).path)")
        }
        let matches = await db.tokensMatchingHashPrefix(prefix)
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
