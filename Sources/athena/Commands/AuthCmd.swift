import ArgumentParser
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

// MARK: - Client-side credential store (login/logout/status)
// `add/list/rm` manage the SERVER keyfile; these manage THIS host's
// stored bearer key for talking to a (local or remote) daemon.

struct AuthLogin: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "login",
        abstract: "Store a bearer key for a daemon (macOS Keychain).")
    @Option(help: "Daemon host.") var host: String = "127.0.0.1"
    @Option(help: "Daemon port.")
    var port: Int = GovernorConfig.defaultPort
    @Option(help: "Key (omit to read stdin / prompt, no echo).")
    var key: String?

    func run() async throws {
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

struct AuthLogout: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "logout",
        abstract: "Remove this host's stored key for a daemon.")
    @Option(help: "Daemon host.") var host: String = "127.0.0.1"
    @Option(help: "Daemon port.")
    var port: Int = GovernorConfig.defaultPort
    func run() async throws {
        print(
            Credentials.remove(host: host, port: port)
                ? "removed stored key for \(host):\(port)"
                : "no stored key for \(host):\(port)")
    }
}

struct AuthStatus: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "status",
        abstract: "Report how the daemon key resolves (no secret).")
    @Option(help: "Daemon host.") var host: String = "127.0.0.1"
    @Option(help: "Daemon port.")
    var port: Int = GovernorConfig.defaultPort
    func run() async throws {
        let env = ProcessInfo.processInfo.environment["ATHENA_KEY"]
        let source: String
        if let env, !env.isEmpty {
            source = "ATHENA_KEY env"
        } else if Credentials.resolve(
            explicit: nil, host: host, port: port) != nil {
            source = "Keychain"
        } else {
            source = "none (set --key / ATHENA_KEY / auth login)"
        }
        print("\(host):\(port) — credential source: \(source)")
    }
}

/// Keyfile: `--file` wins; else the configured `auth_keys_file`;
/// else `~/.athena/auth.keys`.
private func keyfile(_ override: String?) -> URL {
    if let override, !override.isEmpty {
        return URL(
            fileURLWithPath:
                (override as NSString).expandingTildeInPath)
    }
    if let cfg = try? AthenaConfig.parse(
        file: ConfigEditor.resolvePath(nil)),
        let f = cfg.authKeysFile, !f.isEmpty
    {
        return URL(
            fileURLWithPath: (f as NSString).expandingTildeInPath)
    }
    return AthenaEnv.userHome()
        .appendingPathComponent(".athena/auth.keys")
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
    @Option(help: "Keys file (default: configured / ~/.athena/auth.keys).")
    var file: String?

    func run() async throws {
        let t = validTier(tier)
        let url = keyfile(file)
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
        let entry = AuthConfig.hashEntry(forRawKey: key)

        let fm = FileManager.default
        try? fm.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true)
        var contents =
            (try? String(contentsOf: url, encoding: .utf8)) ?? ""
        if !contents.isEmpty, !contents.hasSuffix("\n") {
            contents += "\n"
        }
        contents += "\(t) \(entry)\n"
        do {
            try contents.write(
                to: url, atomically: true, encoding: .utf8)
            try fm.setAttributes(
                [.posixPermissions: 0o600],
                ofItemAtPath: url.path)
        } catch {
            FailableExit.die("error: cannot write \(url.path): \(error)")
        }
        print(
            """
            \(t) key (SAVE NOW — shown once, not stored):

              \(key)

            stored as a hash in \(url.path)
            use:  Authorization: Bearer \(key)
            """)
    }
}

struct AuthList: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "list",
        abstract: "List key tiers + hash prefixes (no secrets).")
    @Option(help: "Keys file.") var file: String?

    func run() async throws {
        let url = keyfile(file)
        guard
            let text = try? String(contentsOf: url, encoding: .utf8)
        else {
            print("no keys file at \(url.path)")
            return
        }
        var n = 0
        for raw in text.split(separator: "\n") {
            let line = raw.trimmingCharacters(in: .whitespaces)
            if line.isEmpty || line.hasPrefix("#") { continue }
            let parts = line.split(
                separator: " ", maxSplits: 1,
                omittingEmptySubsequences: true)
            guard parts.count == 2 else { continue }
            n += 1
            let val = String(parts[1])
            let shown =
                val.hasPrefix("sha256:")
                ? String(val.prefix(20)) + "…" : "(raw key)"
            print("\(parts[0])\t\(shown)")
        }
        print(n == 0 ? "no keys" : "\(n) key(s) — \(url.path)")
    }
}

struct AuthRemove: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "rm",
        abstract: "Remove key entries whose hash hex starts with PREFIX.")
    @Argument(help: "Hash hex prefix (>= 6 chars).") var prefix: String
    @Option(help: "Keys file.") var file: String?

    func run() async throws {
        guard prefix.count >= 6 else {
            FailableExit.die("error: prefix must be >= 6 hex chars")
        }
        let url = keyfile(file)
        guard
            let text = try? String(contentsOf: url, encoding: .utf8)
        else { FailableExit.die("error: no keys file at \(url.path)") }
        var kept: [String] = []
        var removed = 0
        for raw in text.split(
            separator: "\n", omittingEmptySubsequences: false)
        {
            let line = raw.trimmingCharacters(in: .whitespaces)
            if line.contains("sha256:\(prefix)") {
                removed += 1
                continue
            }
            kept.append(String(raw))
        }
        if removed == 0 {
            FailableExit.die("error: no entry matched \(prefix)")
        }
        do {
            try kept.joined(separator: "\n").write(
                to: url, atomically: true, encoding: .utf8)
        } catch {
            FailableExit.die("error: cannot write \(url.path): \(error)")
        }
        print("removed \(removed) entr\(removed == 1 ? "y" : "ies")")
    }
}
