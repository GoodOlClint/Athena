import Foundation

/// Athena's daemon configuration, parsed from the flat TOML at
/// `deploy/athena.toml`. The schema is deliberately flat scalars only — no
/// tables/arrays — so parsing stays trivial and dependency-free (this
/// replaces the installer's old `sed` reader). Commented keys are absent.
public struct AthenaConfig: Sendable, Equatable {
    public var listenHost: String
    public var listenPort: Int
    public var budgetBytes: Int?
    public var engine: String?
    public var model: String?
    /// Model-store root directory. Optional — falls back to the
    /// built-in default (`~/.athena/models`) when absent. Set this to
    /// relocate the store (e.g. onto an external SSD).
    public var modelStore: String?
    /// Where the daemon keeps its SQLite store (vectors + queue + jobs).
    /// Optional — the daemon defaults to `~/.athena` when absent.
    public var dataDir: String?
    /// Log verbosity floor (trace|debug|info|notice|warning|error|
    /// critical). Optional — daemon defaults to info when absent.
    public var logLevel: String?
    /// Opt-in remote syslog sink, e.g. "udp://host:514". The single
    /// documented passive-oracle exception (logs only, default off).
    public var syslogRemote: String?
    /// Inference tuning. All optional — daemon defaults apply when
    /// absent (max 1024 tokens, temp 0.7, speculative off,
    /// vector cap = budget/8). Kept as the raw scalar for
    /// temperature so the value passes straight to `--temperature`.
    public var maxTokens: Int?
    public var temperature: String?
    public var speculative: Bool?
    public var vectorCapBytes: Int?
    /// KV-cache compression codec: `none` (default), `turboquant`, or
    /// `triattention`. Optional — daemon defaults to `none` when absent.
    /// The `ATHENA_KV_COMPRESSION` env var overrides this at startup.
    public var kvCompression: String?
    /// Bearer-auth keys file. Optional — no keys + loopback = open;
    /// no keys + non-loopback = the daemon refuses to start.
    public var authKeysFile: String?
    /// `[network]` egress-proxy keys (M13.2). An operator-set
    /// `*_PROXY` env var always wins over these.
    public var httpsProxy: String?
    public var httpProxy: String?
    public var allProxy: String?
    public var noProxy: String?
    public var logDir: String

    public init(
        listenHost: String, listenPort: Int, budgetBytes: Int?,
        engine: String?, model: String?, modelStore: String? = nil,
        dataDir: String? = nil,
        logLevel: String? = nil, syslogRemote: String? = nil,
        maxTokens: Int? = nil, temperature: String? = nil,
        speculative: Bool? = nil, vectorCapBytes: Int? = nil,
        kvCompression: String? = nil,
        authKeysFile: String? = nil,
        httpsProxy: String? = nil, httpProxy: String? = nil,
        allProxy: String? = nil, noProxy: String? = nil,
        logDir: String
    ) {
        self.listenHost = listenHost
        self.listenPort = listenPort
        self.budgetBytes = budgetBytes
        self.engine = engine
        self.model = model
        self.modelStore = modelStore
        self.dataDir = dataDir
        self.logLevel = logLevel
        self.syslogRemote = syslogRemote
        self.maxTokens = maxTokens
        self.temperature = temperature
        self.speculative = speculative
        self.vectorCapBytes = vectorCapBytes
        self.kvCompression = kvCompression
        self.authKeysFile = authKeysFile
        self.httpsProxy = httpsProxy
        self.httpProxy = httpProxy
        self.allProxy = allProxy
        self.noProxy = noProxy
        self.logDir = logDir
    }

    public enum ParseError: Error, Equatable {
        case missingRequiredKey(String)
        case invalidInt(key: String, value: String)
    }

    /// First uncommented `key = value`; strips surrounding quotes, inline
    /// `#` comments, and whitespace. Returns nil for absent/commented keys.
    static func scalar(_ key: String, in toml: String) -> String? {
        for rawLine in toml.split(
            separator: "\n", omittingEmptySubsequences: false)
        {
            let line = rawLine.drop(while: { $0 == " " || $0 == "\t" })
            if line.first == "#" { continue }
            guard line.hasPrefix(key) else { continue }
            let rest = line.dropFirst(key.count)
                .drop(while: { $0 == " " || $0 == "\t" })
            guard rest.first == "=" else { continue }  // exact key, not prefix
            var value = String(rest.dropFirst())
                .trimmingCharacters(in: .whitespaces)
            if let hash = value.firstIndex(of: "#") {
                value = String(value[..<hash])
                    .trimmingCharacters(in: .whitespaces)
            }
            if value.count >= 2, value.hasPrefix("\""), value.hasSuffix("\"") {
                value = String(value.dropFirst().dropLast())
            }
            return value.isEmpty ? nil : value
        }
        return nil
    }

    public static func parse(toml: String) throws -> AthenaConfig {
        func require(_ key: String) throws -> String {
            guard let v = scalar(key, in: toml) else {
                throw ParseError.missingRequiredKey(key)
            }
            return v
        }
        func int(_ key: String, _ raw: String) throws -> Int {
            guard let i = Int(raw) else {
                throw ParseError.invalidInt(key: key, value: raw)
            }
            return i
        }

        let host = try require("listen_host")
        let portRaw = try require("listen_port")
        let logDir = try require("log_dir")

        var budget: Int?
        if let b = scalar("budget_bytes", in: toml) {
            budget = try int("budget_bytes", b)
        }
        var maxTok: Int?
        if let m = scalar("max_tokens", in: toml) {
            maxTok = try int("max_tokens", m)
        }
        var vecCap: Int?
        if let v = scalar("vector_cap_bytes", in: toml) {
            vecCap = try int("vector_cap_bytes", v)
        }
        let spec = scalar("speculative", in: toml).map { $0 == "true" }

        return AthenaConfig(
            listenHost: host,
            listenPort: try int("listen_port", portRaw),
            budgetBytes: budget,
            engine: scalar("engine", in: toml),
            model: scalar("model", in: toml),
            modelStore: scalar("model_store", in: toml),
            dataDir: scalar("data_dir", in: toml),
            logLevel: scalar("log_level", in: toml),
            syslogRemote: scalar("syslog_remote", in: toml),
            maxTokens: maxTok,
            temperature: scalar("temperature", in: toml),
            speculative: spec,
            vectorCapBytes: vecCap,
            kvCompression: scalar("kv_compression", in: toml),
            authKeysFile: scalar("auth_keys_file", in: toml),
            httpsProxy: scalar("https_proxy", in: toml),
            httpProxy: scalar("http_proxy", in: toml),
            allProxy: scalar("all_proxy", in: toml),
            noProxy: scalar("no_proxy", in: toml),
            logDir: logDir)
    }

    public static func parse(file url: URL) throws -> AthenaConfig {
        try parse(toml: String(contentsOf: url, encoding: .utf8))
    }
}
