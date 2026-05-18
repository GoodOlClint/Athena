import ArgumentParser
import AthenaClient
import AthenaCore
import AthenaDeploy
import AthenaStore
import Crypto
import Darwin
import Foundation

/// `athena auth …` — manage bearer-auth keys (M12.1b). Keys are
/// generated here, shown ONCE, and persisted only as
/// `sha256:<hex>` — no secret is ever written to disk.
struct Auth: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "auth",
        abstract: "Manage bearer-auth keys (hash-only at rest).",
        subcommands: [
            AuthAdd.self, AuthList.self, AuthRemove.self,
            AuthLogin.self, AuthLogout.self, AuthStatus.self,
            AuthUser.self,
        ])
}

// MARK: - WebUI accounts (username/password, PBKDF2 in SQLite)
// Works OFFLINE (opens the daemon's DB directly) so the first
// admin can be created before the auth-protected UI exists.

/// The daemon's SQLite path: configured `data_dir` (or ~/.athena) +
/// athena.sqlite.
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

struct AuthUser: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "user",
        abstract: "Manage WebUI accounts (username/password).",
        subcommands: [
            AuthUserAdd.self, AuthUserList.self, AuthUserRemove.self,
        ])
}

struct AuthUserAdd: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "add",
        abstract: "Create/replace a WebUI account (prompts, no echo).")
    @Argument(help: "Username.") var username: String
    @Option(help: "Password (omit to prompt; avoid in shell history).")
    var password: String?
    @Option(help: "Data dir (default: configured / ~/.athena).")
    var dataDir: String?

    func run() async throws {
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
        let db: AthenaStore
        do {
            db = try AthenaStore(path: storeDBPath(dataDir))
        } catch {
            FailableExit.die("error: cannot open store: \(error)")
        }
        do {
            try await db.putUser(
                username: username, salt: salt, hash: hash,
                iters: Passwords.defaultIterations)
        } catch {
            FailableExit.die("error: \(error)")
        }
        print(
            "user '\(username)' saved "
                + "(\(storeDBPath(dataDir).path))")
    }
}

struct AuthUserList: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "list", abstract: "List WebUI accounts.")
    @Option(help: "Data dir (default: configured / ~/.athena).")
    var dataDir: String?
    func run() async throws {
        guard let db = try? AthenaStore(path: storeDBPath(dataDir))
        else {
            print("no store at \(storeDBPath(dataDir).path)")
            return
        }
        let users = await db.listUsers()
        if users.isEmpty { print("no users") } else {
            users.forEach { print($0) }
        }
    }
}

struct AuthUserRemove: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "rm", abstract: "Delete a WebUI account.")
    @Argument(help: "Username.") var username: String
    @Option(help: "Data dir (default: configured / ~/.athena).")
    var dataDir: String?
    func run() async throws {
        guard let db = try? AthenaStore(path: storeDBPath(dataDir))
        else {
            FailableExit.die(
                "error: no store at \(storeDBPath(dataDir).path)")
        }
        let ok = await db.deleteUser(username: username)
        print(ok ? "removed '\(username)'" : "no such user")
    }
}

// Client-side credential verbs (login/logout/status) live in the
// portable `AthenaClient` module (M14.2a) — the daemon's `Auth`
// command re-exposes them as subcommands.

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

private func validTier(_ s: String) -> String {
    let t = s.lowercased()
    guard t == "admin" || t == "inference" else {
        FailableExit.die("error: tier must be admin or inference")
    }
    return t
}

struct AuthAdd: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "add",
        abstract: "Generate a key for a tier (shown once).")
    @Argument(help: "Tier: admin | inference.") var tier: String
    @Option(help: "Optional label (shown by `auth list`).")
    var label: String?
    @Option(help: "Data dir (default: configured / ~/.athena).")
    var dataDir: String?

    func run() async throws {
        let t = validTier(tier)
        // 32 bytes CSPRNG → sk-athena-<base64url>.
        let raw = SymmetricKey(size: .bits256).withUnsafeBytes {
            Data($0)
        }
        let key =
            "sk-athena-"
            + raw.base64EncodedString()
            .replacingOccurrences(of: "+", with: "-")
            .replacingOccurrences(of: "/", with: "_")
            .replacingOccurrences(of: "=", with: "")
        let db: AthenaStore
        do {
            db = try AthenaStore(path: storeDBPath(dataDir))
        } catch {
            FailableExit.die("error: cannot open store: \(error)")
        }
        do {
            try await db.putToken(
                hash: Data(AuthConfig.sha(key)), tier: t,
                label: label)
        } catch {
            FailableExit.die("error: \(error)")
        }
        print(
            """
            \(t) key (SAVE NOW — shown once, not stored):

              \(key)

            stored as a hash in \(storeDBPath(dataDir).path)
            use:  Authorization: Bearer \(key)
            """)
    }
}

struct AuthList: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "list",
        abstract: "List token tiers + hash prefixes (no secrets).")
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
            print(
                "\(t.tier)\tsha256:\(t.hex.prefix(12))…"
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
