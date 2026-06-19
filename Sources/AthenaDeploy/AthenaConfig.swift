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
    /// Foreground terminal verbosity floor (trace|debug|info|notice|
    /// warning|error|critical). Optional — daemon defaults to info
    /// when absent. M45.1: gates the foreground stdout handler only;
    /// the macOS unified log captures everything regardless.
    /// `sudo log config --mode "level:debug" --subsystem athena` is
    /// the runtime gate for the unified-log side.
    public var logLevel: String?
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
    /// `[prompt_cache]` — cross-request prompt-prefix KV reuse (M59).
    /// `prompt_cache_enabled` (default false) is the master switch;
    /// `prompt_cache_max_entries` (default 4) bounds the in-memory LRU by
    /// count; `prompt_cache_max_bytes` (default governor-derived =
    /// promptCacheCapBytes) caps the pool's bytes; `prompt_cache_idle_ttl_secs`
    /// (default 600) evicts entries idle longer than that. Absent ⇒ disabled.
    public var promptCacheEnabled: Bool?
    public var promptCacheMaxEntries: Int?
    public var promptCacheMaxBytes: Int?
    public var promptCacheIdleTtlSecs: Int?
    /// Scope mode (M59.3): `principal` (default — never cross callers),
    /// `cache_key` (key by the OpenAI prompt_cache_key hint), or `both`.
    public var promptCacheScope: String?
    /// `dflash_enabled` (M63) — DFlash lossless speculative decoding for
    /// attention-only targets (Gemma4) that MTP can't accelerate. Default
    /// false. When on, a request to a target with a registered drafter
    /// (`DFlashRegistry`) decodes via the block draft/verify engine; the
    /// drafter is operator-pulled from Hugging Face on first use. The
    /// `ATHENA_DFLASH` env var (1/true/0/false) overrides this at startup.
    public var dflashEnabled: Bool?
    /// Bearer-auth keys file. Optional — no keys + loopback = open;
    /// no keys + non-loopback = the daemon refuses to start.
    public var authKeysFile: String?
    /// In-daemon TLS (M28). PEM certificate chain + private key paths.
    /// Both must be set together to serve HTTPS; setting only one is a
    /// hard startup error. Absent ⇒ plaintext HTTP (loopback or behind
    /// a TLS reverse proxy — see docs/reverse-proxy.md).
    public var tlsCert: String?
    public var tlsKey: String?
    /// Inbound per-principal rate limiting (M29.1). `rateLimit` =
    /// sustained requests/sec per caller (kept as the raw scalar so it
    /// passes straight to `--rate-limit`); `rateBurst` = token-bucket
    /// capacity. Both optional — absent / non-positive ⇒ disabled
    /// (opt-in, off by default).
    public var rateLimit: String?
    public var rateBurst: Int?
    /// Inbound concurrency caps (M29.2). `maxConcurrency` = max in-flight
    /// requests daemon-wide; `maxConcurrencyPerPrincipal` = max in-flight
    /// per caller. Both optional — absent / non-positive ⇒ unlimited
    /// (opt-in, off by default).
    public var maxConcurrency: Int?
    public var maxConcurrencyPerPrincipal: Int?
    /// Audit-log retention in days (M30.3). Rows older than this are
    /// pruned as the trail grows. Optional — absent / non-positive ⇒
    /// keep forever (opt-in, off by default; an audit trail should not
    /// silently auto-delete unless the operator asks for it).
    public var auditRetentionDays: Int?
    /// Global cap on a managed bearer token's age in days (M36.1).
    /// Optional — absent / non-positive ⇒ no cap (tokens never expire
    /// unless minted with a per-token TTL). Enforced at validation
    /// relative to mint time, so lowering it shortens existing tokens.
    public var tokenMaxAgeDays: Int?
    /// Per-request inference timeout in seconds (M33.1). A generation
    /// that overruns becomes a 504 (sync) or is truncated (streamed) so a
    /// runaway decode is bounded by wall-clock, not only `max_tokens`.
    /// Optional — absent / non-positive ⇒ no deadline (opt-in, off by
    /// default).
    public var requestTimeoutSecs: Int?
    /// Block-until-ready budget for a request-path cold-load, in seconds
    /// (ADR 015). A request for a non-resident-but-on-disk model waits up to
    /// this long for the local load, then serves; on timeout it falls back to
    /// `503 module_loading` + `Retry-After`. Distinct from
    /// `requestTimeoutSecs` (which bounds generation, after the load). Optional
    /// — absent ⇒ a 120s default; `0` ⇒ legacy immediate-503 (the revert
    /// switch). Downloads (operator pull) are never waited on.
    public var coldLoadWaitSecs: Int?
    /// Inbound upload caps (ADR 017). `maxAudioUploadBytes` bounds the raw
    /// multipart body of the three `/v1/audio/*` routes (file + MIME
    /// framing); `maxRequestBodyBytes` bounds the JSON request bodies
    /// (`/v1/chat/completions`, `/v1/embeddings`, …). Both optional — absent
    /// ⇒ daemon defaults (100 MiB audio / 4 MiB JSON). A configured value of
    /// `0` or negative is a parse error (an unbounded upload is a memory-DoS
    /// the governor does not cover — set a large positive number for
    /// "effectively unlimited"). Over-cap ⇒ a clean `413 payload_too_large`.
    public var maxAudioUploadBytes: Int?
    /// Upload cap for the `/v1/video/*` routes (ADR 022). Video dwarfs audio,
    /// so the daemon default is larger (1 GiB). Optional — absent ⇒ default; a
    /// configured `0`/negative is a parse error, like the audio cap.
    public var maxVideoUploadBytes: Int?
    public var maxRequestBodyBytes: Int?
    /// Serve-path MLX buffer-cache bound (ADR 023 G1). Caps
    /// `MLX.Memory.cacheLimit` so the reclaimable buffer pool can't grow to fill
    /// the whole Metal budget (the field finding: 79 GiB of ungoverned cache).
    /// Absent ⇒ a fraction (~⅓) of `budgetBytes`; `0` ⇒ unbounded (MLX default,
    /// today's behavior — an explicit opt-out, NOT a parse error like the upload
    /// caps).
    public var mlxCacheLimitBytes: Int?
    /// Preload (warm) the LLM at startup instead of lazily on first
    /// request (M33.3). Optional — absent / false ⇒ lazy load (default).
    /// Opt-in: the operator trades a slower start for a warm first
    /// request. Best-effort — a failed warm logs and falls back to lazy.
    public var preload: Bool?
    /// Queue-result retention (M34.1). `queueResultTtlSecs` prunes
    /// terminal (done/error/canceled) results older than the window;
    /// `queueMaxRows` caps total job rows (oldest terminal trimmed
    /// first). Both optional — absent / non-positive ⇒ keep forever
    /// (opt-in, off by default; results hold inference outputs, so
    /// retention is the operator's call). Pending jobs are never pruned.
    public var queueResultTtlSecs: Int?
    public var queueMaxRows: Int?
    /// Vector-store retention (M34.2). Prunes vectors written longer ago
    /// than the window on each upsert. Optional — absent / non-positive ⇒
    /// keep forever (opt-in, off by default). Vectors stored before the
    /// feature (no timestamp) are never auto-pruned.
    public var vectorTtlSecs: Int?
    /// Content opt-out (M34.2). When true, a queued job's prompt
    /// (request) blob is cleared once it finishes so inference inputs
    /// don't persist on disk; the result stays (bounded by the queue
    /// TTL). Optional — absent / false ⇒ prompts retained (default).
    public var dropRequestContent: Bool?
    /// At-rest encryption (M34.3b). When true, the SQLite store is opened
    /// (and a plaintext store migrated) under SQLCipher AES-256, keyed
    /// from `ATHENA_STORE_KEY` env or the Keychain (generated on first
    /// run). Optional — absent / false ⇒ plaintext store on disk relying
    /// on FileVault (default).
    public var encryptStore: Bool?
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
        logLevel: String? = nil,
        maxTokens: Int? = nil, temperature: String? = nil,
        speculative: Bool? = nil, vectorCapBytes: Int? = nil,
        kvCompression: String? = nil,
        promptCacheEnabled: Bool? = nil,
        promptCacheMaxEntries: Int? = nil,
        promptCacheMaxBytes: Int? = nil,
        promptCacheIdleTtlSecs: Int? = nil,
        promptCacheScope: String? = nil,
        dflashEnabled: Bool? = nil,
        authKeysFile: String? = nil,
        tlsCert: String? = nil, tlsKey: String? = nil,
        rateLimit: String? = nil, rateBurst: Int? = nil,
        maxConcurrency: Int? = nil,
        maxConcurrencyPerPrincipal: Int? = nil,
        auditRetentionDays: Int? = nil,
        tokenMaxAgeDays: Int? = nil,
        requestTimeoutSecs: Int? = nil,
        coldLoadWaitSecs: Int? = nil,
        maxAudioUploadBytes: Int? = nil,
        maxVideoUploadBytes: Int? = nil,
        maxRequestBodyBytes: Int? = nil,
        mlxCacheLimitBytes: Int? = nil,
        preload: Bool? = nil,
        queueResultTtlSecs: Int? = nil,
        queueMaxRows: Int? = nil,
        vectorTtlSecs: Int? = nil,
        dropRequestContent: Bool? = nil,
        encryptStore: Bool? = nil,
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
        self.maxTokens = maxTokens
        self.temperature = temperature
        self.speculative = speculative
        self.vectorCapBytes = vectorCapBytes
        self.kvCompression = kvCompression
        self.promptCacheEnabled = promptCacheEnabled
        self.promptCacheMaxEntries = promptCacheMaxEntries
        self.promptCacheMaxBytes = promptCacheMaxBytes
        self.promptCacheIdleTtlSecs = promptCacheIdleTtlSecs
        self.promptCacheScope = promptCacheScope
        self.dflashEnabled = dflashEnabled
        self.authKeysFile = authKeysFile
        self.tlsCert = tlsCert
        self.tlsKey = tlsKey
        self.rateLimit = rateLimit
        self.rateBurst = rateBurst
        self.maxConcurrency = maxConcurrency
        self.maxConcurrencyPerPrincipal = maxConcurrencyPerPrincipal
        self.auditRetentionDays = auditRetentionDays
        self.tokenMaxAgeDays = tokenMaxAgeDays
        self.requestTimeoutSecs = requestTimeoutSecs
        self.coldLoadWaitSecs = coldLoadWaitSecs
        self.maxAudioUploadBytes = maxAudioUploadBytes
        self.maxVideoUploadBytes = maxVideoUploadBytes
        self.maxRequestBodyBytes = maxRequestBodyBytes
        self.mlxCacheLimitBytes = mlxCacheLimitBytes
        self.preload = preload
        self.queueResultTtlSecs = queueResultTtlSecs
        self.queueMaxRows = queueMaxRows
        self.vectorTtlSecs = vectorTtlSecs
        self.dropRequestContent = dropRequestContent
        self.encryptStore = encryptStore
        self.httpsProxy = httpsProxy
        self.httpProxy = httpProxy
        self.allProxy = allProxy
        self.noProxy = noProxy
        self.logDir = logDir
    }

    public enum ParseError: Error, Equatable {
        case missingRequiredKey(String)
        case invalidInt(key: String, value: String)
        case invalidBool(key: String, value: String)
    }

    /// J1 (M66.4): parse a TOML bool truthily but strictly. Accepts
    /// `true/false`, `1/0`, `yes/no`, `on/off` (case-insensitive); ANY
    /// other value is a ParseError, not a silent `false`. The old
    /// `$0 == "true"` coerced `1`/`True`/`yes` — and a CRLF `true\r` — to
    /// `false`, silently disabling `encrypt_store`/`preload`/etc.
    static func parseBool(_ key: String, _ raw: String) throws -> Bool {
        switch raw.lowercased() {
        case "true", "1", "yes", "on": return true
        case "false", "0", "no", "off": return false
        default: throw ParseError.invalidBool(key: key, value: raw)
        }
    }

    /// First uncommented `key = value`; strips surrounding quotes, inline
    /// `#` comments, and whitespace. Returns nil for absent/commented keys.
    static func scalar(_ key: String, in toml: String) -> String? {
        // NJ1 (M66.4): normalize CRLF/CR → LF up front. `.whitespaces`
        // does NOT strip a carriage return, so a Windows-saved config
        // otherwise left a trailing `\r` on every value — breaking int
        // parsing (`Int("7447\r")` ⇒ nil ⇒ abort), the quote-strip
        // (closing `"` no longer last), and bool coercion (`"true\r"`).
        let normalized = toml
            .replacingOccurrences(of: "\r\n", with: "\n")
            .replacingOccurrences(of: "\r", with: "\n")
        for rawLine in normalized.split(
            separator: "\n", omittingEmptySubsequences: false)
        {
            let line = rawLine.drop(while: { $0 == " " || $0 == "\t" })
            if line.first == "#" { continue }
            guard line.hasPrefix(key) else { continue }
            let rest = line.dropFirst(key.count)
                .drop(while: { $0 == " " || $0 == "\t" })
            guard rest.first == "=" else { continue }  // exact key, not prefix
            let value = String(rest.dropFirst())
                .trimmingCharacters(in: .whitespaces)
            // J2 (M66.4): a quoted value is literal — an inline `#` inside
            // the quotes is part of the value, not a comment. Take the text
            // up to the closing quote (anything after it, including a
            // trailing `#` comment, is ignored). Only an UNQUOTED value
            // treats `#` as the start of a comment.
            if value.hasPrefix("\"") {
                let body = value.dropFirst()
                if let close = body.firstIndex(of: "\"") {
                    let inner = String(body[..<close])
                    return inner.isEmpty ? nil : inner
                }
                // Unterminated quote — fall through to unquoted handling.
            }
            var v = value
            if let hash = v.firstIndex(of: "#") {
                v = String(v[..<hash])
                    .trimmingCharacters(in: .whitespaces)
            }
            return v.isEmpty ? nil : v
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
        // ADR 017: an upload cap of `0`/negative is rejected (not treated as
        // "unlimited") — an unbounded buffered upload is a memory-DoS the
        // governor doesn't account for. Reuses `invalidInt` so the ParseError
        // enum stays stable.
        func positiveInt(_ key: String, _ raw: String) throws -> Int {
            let i = try int(key, raw)
            guard i > 0 else {
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
        var rateBurst: Int?
        if let rb = scalar("rate_burst", in: toml) {
            rateBurst = try int("rate_burst", rb)
        }
        var maxConc: Int?
        if let mc = scalar("max_concurrency", in: toml) {
            maxConc = try int("max_concurrency", mc)
        }
        var maxConcPP: Int?
        if let mp = scalar("max_concurrency_per_principal", in: toml) {
            maxConcPP = try int("max_concurrency_per_principal", mp)
        }
        var auditDays: Int?
        if let ad = scalar("audit_retention_days", in: toml) {
            auditDays = try int("audit_retention_days", ad)
        }
        var tokenMaxAge: Int?
        if let tm = scalar("token_max_age_days", in: toml) {
            tokenMaxAge = try int("token_max_age_days", tm)
        }
        var reqTimeout: Int?
        if let rt = scalar("request_timeout_secs", in: toml) {
            reqTimeout = try int("request_timeout_secs", rt)
        }
        var coldLoadWait: Int?
        if let cw = scalar("cold_load_wait_secs", in: toml) {
            coldLoadWait = try int("cold_load_wait_secs", cw)
        }
        var maxAudioUpload: Int?
        if let ma = scalar("max_audio_upload_bytes", in: toml) {
            maxAudioUpload = try positiveInt("max_audio_upload_bytes", ma)
        }
        var maxVideoUpload: Int?
        if let mv = scalar("max_video_upload_bytes", in: toml) {
            maxVideoUpload = try positiveInt("max_video_upload_bytes", mv)
        }
        var maxRequestBody: Int?
        if let mb = scalar("max_request_body_bytes", in: toml) {
            maxRequestBody = try positiveInt("max_request_body_bytes", mb)
        }
        // ADR 023 G1: `0` is a valid value (unbounded), so parse with `int`
        // (allows 0/negative ⇒ treated as unbounded), NOT `positiveInt`.
        var mlxCacheLimit: Int?
        if let cl = scalar("mlx_cache_limit_bytes", in: toml) {
            mlxCacheLimit = try int("mlx_cache_limit_bytes", cl)
        }
        var queueTtl: Int?
        if let qt = scalar("queue_result_ttl_secs", in: toml) {
            queueTtl = try int("queue_result_ttl_secs", qt)
        }
        var queueMax: Int?
        if let qm = scalar("queue_max_rows", in: toml) {
            queueMax = try int("queue_max_rows", qm)
        }
        var vectorTtl: Int?
        if let vt = scalar("vector_ttl_secs", in: toml) {
            vectorTtl = try int("vector_ttl_secs", vt)
        }
        // J1 (M66.4): all bool keys parse via the strict truthy `bool`
        // helper; an unrecognized value is a ParseError, not silent false.
        func bool(_ key: String) throws -> Bool? {
            guard let v = scalar(key, in: toml) else { return nil }
            return try parseBool(key, v)
        }
        let spec = try bool("speculative")
        let pcEnabled = try bool("prompt_cache_enabled")
        var pcMaxEntries: Int?
        if let pm = scalar("prompt_cache_max_entries", in: toml) {
            pcMaxEntries = try int("prompt_cache_max_entries", pm)
        }
        var pcMaxBytes: Int?
        if let pb = scalar("prompt_cache_max_bytes", in: toml) {
            pcMaxBytes = try int("prompt_cache_max_bytes", pb)
        }
        var pcIdleTtl: Int?
        if let pt = scalar("prompt_cache_idle_ttl_secs", in: toml) {
            pcIdleTtl = try int("prompt_cache_idle_ttl_secs", pt)
        }
        let dflashEnabled = try bool("dflash_enabled")
        let preload = try bool("preload")
        let dropReq = try bool("drop_request_content")
        let encStore = try bool("encrypt_store")

        return AthenaConfig(
            listenHost: host,
            listenPort: try int("listen_port", portRaw),
            budgetBytes: budget,
            engine: scalar("engine", in: toml),
            model: scalar("model", in: toml),
            modelStore: scalar("model_store", in: toml),
            dataDir: scalar("data_dir", in: toml),
            logLevel: scalar("log_level", in: toml),
            maxTokens: maxTok,
            temperature: scalar("temperature", in: toml),
            speculative: spec,
            vectorCapBytes: vecCap,
            kvCompression: scalar("kv_compression", in: toml),
            promptCacheEnabled: pcEnabled,
            promptCacheMaxEntries: pcMaxEntries,
            promptCacheMaxBytes: pcMaxBytes,
            promptCacheIdleTtlSecs: pcIdleTtl,
            promptCacheScope: scalar("prompt_cache_scope", in: toml),
            dflashEnabled: dflashEnabled,
            authKeysFile: scalar("auth_keys_file", in: toml),
            tlsCert: scalar("tls_cert", in: toml),
            tlsKey: scalar("tls_key", in: toml),
            rateLimit: scalar("rate_limit", in: toml),
            rateBurst: rateBurst,
            maxConcurrency: maxConc,
            maxConcurrencyPerPrincipal: maxConcPP,
            auditRetentionDays: auditDays,
            tokenMaxAgeDays: tokenMaxAge,
            requestTimeoutSecs: reqTimeout,
            coldLoadWaitSecs: coldLoadWait,
            maxAudioUploadBytes: maxAudioUpload,
            maxVideoUploadBytes: maxVideoUpload,
            maxRequestBodyBytes: maxRequestBody,
            mlxCacheLimitBytes: mlxCacheLimit,
            preload: preload,
            queueResultTtlSecs: queueTtl,
            queueMaxRows: queueMax,
            vectorTtlSecs: vectorTtl,
            dropRequestContent: dropReq,
            encryptStore: encStore,
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
