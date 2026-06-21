import Foundation

/// Generates the Athena launchd daemon plist directly as a property list —
/// no string template, no `sed` sentinels. Boot-time system daemon
/// (RunAtLoad + KeepAlive); logs to local files for Vector → Clio (Athena
/// initiates nothing to Crete — passive-oracle contract).
public enum LaunchdPlist {

    public static func dictionary(
        label: String,
        executablePath: String,
        user: String,
        workingDirectory: String,
        config: AthenaConfig,
        configPath: String = ""
    ) -> [String: Any] {
        // `executablePath` is the `athena` binary itself; the launchd
        // daemon runs `athena load …` directly. The pre-M42 plist
        // execed the `athenad` shim, which then `execv`-ed `athena`
        // with `argv[0]="athena"` (bare). Under launchd, that argv[0]
        // / actual-binary-path discrepancy made Swift's
        // `Bundle.main.bundleURL` resolve to the wrong directory, so
        // mlx-c's SwiftPM-bundle lookup couldn't find
        // `mlx-swift_Cmlx.bundle` and `MLX.Memory.memoryLimit = …`
        // threw "Failed to load the default metallib library not
        // found …" four times at first call. Foreground `athena load`
        // worked because the shell exec preserved the full path as
        // argv[0]. Skipping `athenad` from the launchd path keeps
        // argv[0] aligned with the kernel's view, so Bundle.main
        // resolves correctly.
        //
        // `--background` (M45.1) tells the daemon it's launchd-spawned
        // so it drops the foreground stdout sink; the macOS unified
        // log is the sole diagnostic surface for the installed
        // daemon. Operators query via `log show / log stream` and
        // tune verbosity at runtime with `sudo log config --mode
        // "level:debug" --subsystem athena`.
        var args: [String] = [
            executablePath,
            "load",
            "--background",
            "--host", config.listenHost,
            "--port", String(config.listenPort),
        ]
        if let budget = config.budgetBytes {
            args += ["--budget-bytes", String(budget)]
        }
        if let engine = config.engine {
            args += ["--engine", engine]
        }
        if let model = config.model {
            args += ["--model", model]
        }
        // ADR 026 — per-module default model ids become the per-module
        // first-boot seed flags so the installed daemon's configured default
        // reaches `athena load` (which uses each flag's first entry as the
        // module's default).
        if let m = config.embeddingModel {
            args += ["--embedding-model", m]
        }
        if let m = config.transcriptionModel {
            args += ["--whisper-model", m]
        }
        if let m = config.diarizationModel {
            args += ["--diarization-model", m]
        }
        if let m = config.speakerEmbeddingModel {
            args += ["--speaker-embedding-model", m]
        }
        if let modelStore = config.modelStore {
            args += ["--model-store", modelStore]
        }
        if let dataDir = config.dataDir {
            args += ["--data-dir", dataDir]
        }
        if let logLevel = config.logLevel {
            args += ["--log-level", logLevel]
        }
        if let maxTokens = config.maxTokens {
            args += ["--max-tokens", String(maxTokens)]
        }
        if let temperature = config.temperature {
            args += ["--temperature", temperature]
        }
        if config.speculative == true {
            args += ["--speculative"]  // a Flag, not an Option
        }
        if let authKeysFile = config.authKeysFile {
            args += ["--auth-keys-file", authKeysFile]
        }
        if let tlsCert = config.tlsCert {
            args += ["--tls-cert", tlsCert]
        }
        if let tlsKey = config.tlsKey {
            args += ["--tls-key", tlsKey]
        }
        if let rateLimit = config.rateLimit {
            args += ["--rate-limit", rateLimit]
        }
        if let rateBurst = config.rateBurst {
            args += ["--rate-burst", String(rateBurst)]
        }
        if let maxConcurrency = config.maxConcurrency {
            args += ["--max-concurrency", String(maxConcurrency)]
        }
        if let maxConcPP = config.maxConcurrencyPerPrincipal {
            args += [
                "--max-concurrency-per-principal", String(maxConcPP),
            ]
        }
        if let auditDays = config.auditRetentionDays {
            args += ["--audit-retention-days", String(auditDays)]
        }
        if let tokenMaxAge = config.tokenMaxAgeDays {
            args += ["--token-max-age-days", String(tokenMaxAge)]
        }
        if let reqTimeout = config.requestTimeoutSecs {
            args += ["--request-timeout-secs", String(reqTimeout)]
        }
        if let coldWait = config.coldLoadWaitSecs {
            args += ["--cold-load-wait-secs", String(coldWait)]
        }
        if let audioCap = config.maxAudioUploadBytes {
            args += ["--max-audio-upload-bytes", String(audioCap)]
        }
        if let videoCap = config.maxVideoUploadBytes {
            args += ["--max-video-upload-bytes", String(videoCap)]
        }
        if let bodyCap = config.maxRequestBodyBytes {
            args += ["--max-request-body-bytes", String(bodyCap)]
        }
        if let cacheLimit = config.mlxCacheLimitBytes {
            args += ["--mlx-cache-limit-bytes", String(cacheLimit)]
        }
        if let admissionMode = config.governorAdmissionMode {
            args += ["--governor-admission-mode", admissionMode]
        }
        if config.preload == true {
            args += ["--preload"]  // a Flag, not an Option
        }
        if config.encryptStore == true {
            args += ["--encrypt-store"]  // a Flag, not an Option
        }
        if config.persistStore == true {
            args += ["--persist-store"]  // a Flag, not an Option
        }

        // Diagnostic surface is the macOS unified log (M45.1). The
        // daemon emits nothing to stdout under launchd, so
        // StandardOutPath is /dev/null. StandardErrorPath stays as a
        // small file under log_dir to catch process-death output
        // (fatalError, NIO precondition failures, MLX/Metal panics)
        // that fires after Logger is torn down — a CRASH DUMP, not a
        // diagnostic log.
        var dict: [String: Any] = [
            "Label": label,
            "ProgramArguments": args,
            "UserName": user,
            "RunAtLoad": true,
            "KeepAlive": true,
            "ProcessType": "Standard",
            "WorkingDirectory": workingDirectory,
            "StandardOutPath": "/dev/null",
            "StandardErrorPath": "\(config.logDir)/athena.err.log",
            "SoftResourceLimits": ["NumberOfFiles": 8192],
        ]
        // NJ2 (M66.4): export the PREFIX-CORRECT installed config path so
        // the daemon's TOML-only re-reads (kv_compression, prompt_cache_*,
        // the egress-proxy keys — none forwarded as `load` args above)
        // resolve to the file this install actually wrote, not the
        // hard-coded `/usr/local`. `ConfigEditor.resolvePath(nil)` reads
        // `ATHENA_CONFIG`.
        if !configPath.isEmpty {
            dict["EnvironmentVariables"] = ["ATHENA_CONFIG": configPath]
        }
        return dict
    }

    public static func xmlData(
        label: String,
        executablePath: String,
        user: String,
        workingDirectory: String,
        config: AthenaConfig,
        configPath: String = ""
    ) throws -> Data {
        let dict = dictionary(
            label: label, executablePath: executablePath, user: user,
            workingDirectory: workingDirectory, config: config,
            configPath: configPath)
        return try PropertyListSerialization.data(
            fromPropertyList: dict, format: .xml, options: 0)
    }
}
