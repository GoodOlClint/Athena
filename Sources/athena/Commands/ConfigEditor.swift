import AthenaDeploy
import Foundation

/// Shared flat-TOML config editing used by `athena config` (M9.4b) and
/// `athena default` (M9.5d). Keeps the in-place scalar rewrite (which
/// preserves comments/layout) in exactly one place.
enum ConfigEditor {
    /// String-valued keys are quoted; these two are bare ints.
    static let intKeys: Set<String> = ["listen_port", "budget_bytes"]
    static let knownKeys: Set<String> = [
        "listen_host", "listen_port", "budget_bytes", "engine",
        "model", "data_dir", "log_level", "syslog_remote", "log_dir",
    ]

    /// `--config` wins; else the installed file if present; else the
    /// in-repo dev copy.
    static func resolvePath(_ override: String?) -> URL {
        if let override {
            return URL(
                fileURLWithPath:
                    (override as NSString).expandingTildeInPath)
        }
        let installed = URL(
            fileURLWithPath: "/usr/local/etc/athena/athena.toml")
        if FileManager.default.fileExists(atPath: installed.path) {
            return installed
        }
        return URL(fileURLWithPath: "deploy/athena.toml")
    }

    static func read(_ url: URL) -> String {
        guard let s = try? String(contentsOf: url, encoding: .utf8)
        else {
            FailableExit.die("error: no config at \(url.path)")
        }
        return s
    }

    /// The effective value of `key` from a parsed config (nil ⇒ unset
    /// / built-in default).
    static func value(_ key: String, in cfg: AthenaConfig) -> String? {
        switch key {
        case "listen_host": return cfg.listenHost
        case "listen_port": return String(cfg.listenPort)
        case "budget_bytes": return cfg.budgetBytes.map(String.init)
        case "engine": return cfg.engine
        case "model": return cfg.model
        case "data_dir": return cfg.dataDir
        case "log_level": return cfg.logLevel
        case "syslog_remote": return cfg.syslogRemote
        case "log_dir": return cfg.logDir
        default: FailableExit.die("error: unknown key '\(key)'")
        }
    }

    /// An active (uncommented) `key = …` assignment on this line?
    private static func isAssignment(
        _ line: Substring, _ key: String
    ) -> Bool {
        let t = line.drop(while: { $0 == " " || $0 == "\t" })
        guard !t.hasPrefix("#"), t.hasPrefix(key) else { return false }
        let rest = t.dropFirst(key.count)
            .drop(while: { $0 == " " || $0 == "\t" })
        return rest.first == "="
    }

    /// A commented `# key = …` line (so we can uncomment in place)?
    private static func isCommented(
        _ line: Substring, _ key: String
    ) -> Bool {
        var t = line.drop(while: { $0 == " " || $0 == "\t" })
        guard t.first == "#" else { return false }
        t = t.dropFirst().drop(while: { $0 == " " || $0 == "\t" })
        guard t.hasPrefix(key) else { return false }
        let rest = t.dropFirst(key.count)
            .drop(while: { $0 == " " || $0 == "\t" })
        return rest.first == "="
    }

    enum Failure: Error, CustomStringConvertible {
        case unknownKey(String)
        case notAnInteger(String)
        case noConfig(URL)
        case writeFailed(URL, String)
        var description: String {
            switch self {
            case .unknownKey(let k):
                return
                    "unknown key '\(k)' (allowed: "
                    + ConfigEditor.knownKeys.sorted()
                    .joined(separator: ", ") + ")"
            case .notAnInteger(let k):
                return "\(k) must be an integer"
            case .noConfig(let u): return "no config at \(u.path)"
            case .writeFailed(let u, let e):
                return "cannot write \(u.path): \(e)"
            }
        }
    }

    /// Validate + rewrite one scalar in place (replacing an active
    /// line or uncommenting a `# key =` one), then sanity-parse.
    /// THROWS rather than exiting — safe to call from the server
    /// (`/ui/api/config`); a bad request must never kill the daemon.
    static func setScalarThrowing(
        key: String, value: String, in url: URL
    ) throws {
        guard knownKeys.contains(key) else {
            throw Failure.unknownKey(key)
        }
        let formatted: String
        if intKeys.contains(key) {
            guard Int(value) != nil else {
                throw Failure.notAnInteger(key)
            }
            formatted = "\(key) = \(value)"
        } else {
            formatted = "\(key) = \"\(value)\""
        }
        guard
            let contents = try? String(
                contentsOf: url, encoding: .utf8)
        else { throw Failure.noConfig(url) }

        var lines = contents.split(
            separator: "\n", omittingEmptySubsequences: false)
        if let i = lines.firstIndex(where: { isAssignment($0, key) }) {
            lines[i] = Substring(formatted)
        } else if let i = lines.firstIndex(where: {
            isCommented($0, key)
        }) {
            lines[i] = Substring(formatted)
        } else {
            if lines.last?.isEmpty == true { lines.removeLast() }
            lines.append(Substring(formatted))
        }
        let rewritten = lines.joined(separator: "\n") + "\n"
        do {
            try rewritten.write(
                to: url, atomically: true, encoding: .utf8)
        } catch {
            throw Failure.writeFailed(url, "\(error)")
        }
        // Sanity-parse; warn (don't roll back) on failure — usually a
        // pre-existing missing required key, not this edit.
        if (try? AthenaConfig.parse(file: url)) == nil {
            FileHandle.standardError.write(
                Data("warning: config now unparseable\n".utf8))
        }
    }

    /// CLI wrapper: same as before — print + exit(1) on failure.
    static func setScalar(key: String, value: String, in url: URL) {
        do {
            try setScalarThrowing(key: key, value: value, in: url)
        } catch {
            FailableExit.die("error: \(error)")
        }
    }
}
