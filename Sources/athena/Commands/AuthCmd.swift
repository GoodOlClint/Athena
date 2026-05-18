import ArgumentParser
import AthenaCore
import AthenaDeploy
import Crypto
import Foundation

/// `athena auth …` — manage bearer-auth keys (M12.1b). Keys are
/// generated here, shown ONCE, and persisted only as
/// `sha256:<hex>` — no secret is ever written to disk.
struct Auth: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "auth",
        abstract: "Manage bearer-auth keys (hash-only at rest).",
        subcommands: [AuthAdd.self, AuthList.self, AuthRemove.self])
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
