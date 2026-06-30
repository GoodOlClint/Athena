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
        case "max_prompt_tokens": return cfg.maxPromptTokens.map(String.init)
        case "temperature": return cfg.temperature
        case "speculative":
            return cfg.speculative.map { $0 ? "true" : "false" }
        case "mtp_drafter": return cfg.mtpDrafter
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
        case "cold_load_wait_secs":
            return cfg.coldLoadWaitSecs.map(String.init)
        case "max_audio_upload_bytes":
            return cfg.maxAudioUploadBytes.map(String.init)
        case "max_video_upload_bytes":
            return cfg.maxVideoUploadBytes.map(String.init)
        case "max_request_body_bytes":
            return cfg.maxRequestBodyBytes.map(String.init)
        case "mlx_cache_limit_bytes":
            return cfg.mlxCacheLimitBytes.map(String.init)
        case "governor_admission_mode":
            return cfg.governorAdmissionMode
        case "preload":
            return cfg.preload.map { $0 ? "true" : "false" }
        case "encrypt_store":
            return cfg.encryptStore.map { $0 ? "true" : "false" }
        case "persist_store":
            return cfg.persistStore.map { $0 ? "true" : "false" }
        case "https_proxy": return cfg.httpsProxy
        case "http_proxy": return cfg.httpProxy
        case "all_proxy": return cfg.allProxy
        case "no_proxy": return cfg.noProxy
        case "kv_compression": return cfg.kvCompression
        case "prompt_cache_enabled":
            return cfg.promptCacheEnabled.map { $0 ? "true" : "false" }
        case "prompt_cache_max_entries":
            return cfg.promptCacheMaxEntries.map(String.init)
        case "prompt_cache_max_bytes":
            return cfg.promptCacheMaxBytes.map(String.init)
        case "prompt_cache_idle_ttl_secs":
            return cfg.promptCacheIdleTtlSecs.map(String.init)
        case "prompt_cache_scope": return cfg.promptCacheScope
        case "prompt_cache_encrypt_idle":
            return cfg.promptCacheEncryptIdle.map { $0 ? "true" : "false" }
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
