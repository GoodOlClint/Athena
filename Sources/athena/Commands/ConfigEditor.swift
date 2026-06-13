import AthenaClient
import AthenaDeploy
import Foundation

/// CLI conveniences over the `ConfigEditor` editing core (which lives in the
/// MLX-free `AthenaDeploy` so its validation is unit-testable — NB4 /
/// M70.1b). These three call `FailableExit.die`, which lives in the
/// Linux-clean `AthenaClient`; keeping them here means `AthenaDeploy` need not
/// depend on `AthenaClient`. The editing/validation logic (`setScalarThrowing`,
/// key sets, `resolvePath`, `Failure`) is in `AthenaDeploy`.
extension ConfigEditor {
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

    /// CLI wrapper over `setScalarThrowing`: print + exit(1) on failure.
    static func setScalar(key: String, value: String, in url: URL) {
        do {
            try setScalarThrowing(key: key, value: value, in: url)
        } catch {
            FailableExit.die("error: \(error)")
        }
    }
}
