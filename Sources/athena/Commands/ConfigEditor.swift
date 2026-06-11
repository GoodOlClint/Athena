import AthenaClient
import AthenaDeploy
import Foundation

/// Shared flat-TOML config editing used by `athena config` (M9.4b) and
/// `athena default` (M9.5d). Keeps the in-place scalar rewrite (which
/// preserves comments/layout) in exactly one place.
enum ConfigEditor {
    /// String-valued keys are quoted; these two are bare ints.
    static let intKeys: Set<String> = [
        "listen_port", "budget_bytes", "max_tokens",
        "vector_cap_bytes", "rate_burst",
        "max_concurrency", "max_concurrency_per_principal",
        "audit_retention_days", "token_max_age_days",
        "request_timeout_secs",
        "queue_result_ttl_secs", "queue_max_rows", "vector_ttl_secs",
        "prompt_cache_max_entries", "prompt_cache_max_bytes",
        "prompt_cache_idle_ttl_secs",
    ]
    /// Written bare (unquoted), like ints: floats and bools.
    static let rawKeys: Set<String> = [
        "temperature", "speculative", "rate_limit", "preload",
        "drop_request_content", "encrypt_store", "prompt_cache_enabled",
        "dflash_enabled",
    ]
    static let knownKeys: Set<String> = [
        "listen_host", "listen_port", "budget_bytes", "engine",
        "model", "model_store", "data_dir", "log_level",
        "log_dir", "max_tokens", "temperature",
        "speculative", "vector_cap_bytes", "auth_keys_file",
        "tls_cert", "tls_key", "rate_limit", "rate_burst",
        "max_concurrency", "max_concurrency_per_principal",
        "audit_retention_days", "token_max_age_days",
        "request_timeout_secs", "preload",
        "queue_result_ttl_secs", "queue_max_rows", "vector_ttl_secs",
        "drop_request_content", "encrypt_store",
        "https_proxy", "http_proxy", "all_proxy", "no_proxy",
        "kv_compression",
        "prompt_cache_enabled", "prompt_cache_max_entries",
        "prompt_cache_max_bytes", "prompt_cache_idle_ttl_secs",
        "prompt_cache_scope", "dflash_enabled",
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
        case "model_store": return cfg.modelStore
        case "data_dir": return cfg.dataDir
        case "log_level": return cfg.logLevel
        case "log_dir": return cfg.logDir
        case "max_tokens": return cfg.maxTokens.map(String.init)
        case "temperature": return cfg.temperature
        case "speculative":
            return cfg.speculative.map { $0 ? "true" : "false" }
        case "vector_cap_bytes":
            return cfg.vectorCapBytes.map(String.init)
        case "auth_keys_file": return cfg.authKeysFile
        case "tls_cert": return cfg.tlsCert
        case "tls_key": return cfg.tlsKey
        case "rate_limit": return cfg.rateLimit
        case "rate_burst": return cfg.rateBurst.map(String.init)
        case "max_concurrency":
            return cfg.maxConcurrency.map(String.init)
        case "max_concurrency_per_principal":
            return cfg.maxConcurrencyPerPrincipal.map(String.init)
        case "audit_retention_days":
            return cfg.auditRetentionDays.map(String.init)
        case "token_max_age_days":
            return cfg.tokenMaxAgeDays.map(String.init)
        case "request_timeout_secs":
            return cfg.requestTimeoutSecs.map(String.init)
        case "preload":
            return cfg.preload.map { $0 ? "true" : "false" }
        case "queue_result_ttl_secs":
            return cfg.queueResultTtlSecs.map(String.init)
        case "queue_max_rows":
            return cfg.queueMaxRows.map(String.init)
        case "vector_ttl_secs":
            return cfg.vectorTtlSecs.map(String.init)
        case "drop_request_content":
            return cfg.dropRequestContent.map { $0 ? "true" : "false" }
        case "encrypt_store":
            return cfg.encryptStore.map { $0 ? "true" : "false" }
        case "https_proxy": return cfg.httpsProxy
        case "http_proxy": return cfg.httpProxy
        case "all_proxy": return cfg.allProxy
        case "no_proxy": return cfg.noProxy
        case "kv_compression": return cfg.kvCompression
        case "prompt_cache_enabled":
            return cfg.promptCacheEnabled.map { $0 ? "true" : "false" }
        case "dflash_enabled":
            return cfg.dflashEnabled.map { $0 ? "true" : "false" }
        case "prompt_cache_max_entries":
            return cfg.promptCacheMaxEntries.map(String.init)
        case "prompt_cache_max_bytes":
            return cfg.promptCacheMaxBytes.map(String.init)
        case "prompt_cache_idle_ttl_secs":
            return cfg.promptCacheIdleTtlSecs.map(String.init)
        case "prompt_cache_scope": return cfg.promptCacheScope
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
        case badValue(String, String)
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
            case .badValue(let k, let want):
                return "\(k) must be \(want)"
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
        } else if rawKeys.contains(key) {
            // Bare, unquoted. Validate the two raw keys' shapes.
            if key == "temperature", Double(value) == nil {
                throw Failure.badValue(key, "a number")
            }
            if key == "rate_limit", Double(value) == nil {
                throw Failure.badValue(key, "a number")
            }
            if key == "speculative",
                value != "true", value != "false" {
                throw Failure.badValue(key, "true or false")
            }
            if key == "preload",
                value != "true", value != "false" {
                throw Failure.badValue(key, "true or false")
            }
            if key == "drop_request_content",
                value != "true", value != "false" {
                throw Failure.badValue(key, "true or false")
            }
            if key == "encrypt_store",
                value != "true", value != "false" {
                throw Failure.badValue(key, "true or false")
            }
            if key == "prompt_cache_enabled",
                value != "true", value != "false" {
                throw Failure.badValue(key, "true or false")
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
