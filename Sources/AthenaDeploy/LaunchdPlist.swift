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
        config: AthenaConfig
    ) -> [String: Any] {
        // `executablePath` is the `athenad` launcher (M14.2d): it
        // execs `athena load` itself, so no "load" arg here.
        var args: [String] = [
            executablePath,
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
        if let modelStore = config.modelStore {
            args += ["--model-store", modelStore]
        }
        if let dataDir = config.dataDir {
            args += ["--data-dir", dataDir]
        }
        if let logLevel = config.logLevel {
            args += ["--log-level", logLevel]
        }
        if let syslogRemote = config.syslogRemote {
            args += ["--syslog-remote", syslogRemote]
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
        if let vectorCapBytes = config.vectorCapBytes {
            args += ["--vector-cap-bytes", String(vectorCapBytes)]
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
        if config.preload == true {
            args += ["--preload"]  // a Flag, not an Option
        }
        if let queueTtl = config.queueResultTtlSecs {
            args += ["--queue-result-ttl-secs", String(queueTtl)]
        }
        if let queueMax = config.queueMaxRows {
            args += ["--queue-max-rows", String(queueMax)]
        }
        if let vectorTtl = config.vectorTtlSecs {
            args += ["--vector-ttl-secs", String(vectorTtl)]
        }
        if config.dropRequestContent == true {
            args += ["--drop-request-content"]  // a Flag, not an Option
        }
        if config.encryptStore == true {
            args += ["--encrypt-store"]  // a Flag, not an Option
        }

        return [
            "Label": label,
            "ProgramArguments": args,
            "UserName": user,
            "RunAtLoad": true,
            "KeepAlive": true,
            "ProcessType": "Standard",
            "WorkingDirectory": workingDirectory,
            "StandardOutPath": "\(config.logDir)/athena.out.log",
            "StandardErrorPath": "\(config.logDir)/athena.err.log",
            "SoftResourceLimits": ["NumberOfFiles": 8192],
        ]
    }

    public static func xmlData(
        label: String,
        executablePath: String,
        user: String,
        workingDirectory: String,
        config: AthenaConfig
    ) throws -> Data {
        let dict = dictionary(
            label: label, executablePath: executablePath, user: user,
            workingDirectory: workingDirectory, config: config)
        return try PropertyListSerialization.data(
            fromPropertyList: dict, format: .xml, options: 0)
    }
}
