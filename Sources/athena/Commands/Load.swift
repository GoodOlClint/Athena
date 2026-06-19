import ArgumentParser
import AthenaClient
import AthenaCore
import AthenaDeploy
import AthenaEmbedding
import AthenaLLM
import AthenaServerKit
import AthenaStore
import AthenaStructured
import AthenaTranscription
import Darwin
import Foundation
import Logging
import MLX

// `Engine` lives in AthenaCore (NB4 / M70.1b) so ConfigEditor's
// `engine ∈ Engine.allCases` validation could move to the MLX-free
// AthenaDeploy. The ArgumentParser conformance (for the `--engine`
// @Option) needs ArgumentParser, so it stays here as an extension. The
// default RawRepresentable init satisfies the protocol.
extension Engine: ExpressibleByArgument {}

/// Run the governed HTTP surface in the foreground. This is the
/// launchd-able daemon body. (Was `serve`; renamed for symmetry with
/// `unload` — `serve` kept as an ollama-compat alias. M9.4.)
struct Load: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "load",
        abstract: "Run the governed HTTP inference surface (foreground).",
        aliases: ["serve"]
    )

    // Default auxiliary-module HF ids — the single source of truth for
    // both `load`'s option defaults below and `athena init`. The LLM has
    // no hard default (the operator picks one via `athena default`).
    static let defaultEmbeddingModel = "BAAI/bge-small-en-v1.5"
    static let defaultTranscriptionModel =
        "mlx-community/whisper-large-v3-turbo"
    /// Additive Parakeet-TDT transcription backend (ADR 020) — selectable via
    /// the `model` form field for higher multilingual ASR quality / ~63× RT.
    /// Not the default transcription engine; seeded into the allowlist
    /// alongside Whisper (the default stays Whisper until a WER/throughput A/B
    /// justifies otherwise — ADR 020 open decision).
    static let defaultParakeetTranscriptionModel =
        "mlx-community/parakeet-tdt-0.6b-v3"
    static let defaultDiarizationModel =
        "mlx-community/diar_streaming_sortformer_4spk-v2.1-fp16"
    /// Additive pyannote segmentation backend (ADR 018) — selectable via
    /// `method=pyannote` for >4 / overlapping speakers. Not the default
    /// diarizer; seeded into the allowlist alongside Sortformer.
    static let defaultPyannoteSegmentationModel =
        "aufklarer/Pyannote-Segmentation-MLX"
    static let defaultSpeakerEmbeddingModel =
        "aufklarer/WeSpeaker-ResNet34-LM-MLX"

    @Option(help: "Listen host.")
    var host: String = "127.0.0.1"

    @Option(help: "Listen port. Default 7447 — Athena's own port.")
    var port: Int = GovernorConfig.defaultPort

    @Option(help: "Global memory budget in bytes. Defaults to 75% of RAM.")
    var budgetBytes: Int?

    @Option(
        help:
            "Max KV/prompt-cache bytes per request. Default: ¼ of budget."
    )
    var promptCacheCapBytes: Int?

    @Option(help: "LLM engine: mlx (real Qwen3.5) or stub (no model).")
    var engine: Engine = .mlx

    @Option(help: "Sampling temperature (0 = deterministic greedy).")
    var temperature: Double?

    @Option(help: "Max generated tokens.")
    var maxTokens: Int = 1024

    @Flag(
        help:
            "Enable MTP speculative decoding. Bit-identical-greedy at temperature 0; Leviathan/Chen sampling speculative (distributionally identical to non-speculative sampling at the same temp/top_p/seed) at temperature > 0. Requires the loaded model to have an MTP head. Per-request `speculative` override available on /v1 + /api."
    )
    var speculative = false

    @Option(help: "Model directory path, or a name under the model store.")
    var model: String?

    @Option(
        name: .customLong("llm-model"),
        help: """
            LLM model id (store name or absolute directory). Repeatable: \
            pass it more than once to make several models selectable \
            per-request via the `model` body field; the FIRST is the \
            default (used when a request omits `model`). An id outside \
            this set is a 400 (`model_not_available`) — never an \
            on-request download. When unset, falls back to --model.
            """
    )
    var llmModels: [String] = []

    @Option(help: "Model store root. Default: ~/.athena/models.")
    var modelStore: String?

    @Option(
        name: .customLong("embedding-model"),
        help: """
            Text-embedding model HF id. Repeatable: pass it more than once \
            to make several models selectable per-request via the `model` \
            body field; the FIRST is the default (used when a request omits \
            `model`). A request for a model not in this set is a 400 (never \
            an on-request download). Default BAAI/bge-small-en-v1.5.
            """
    )
    var embeddingModels: [String] = [Load.defaultEmbeddingModel]

    @Option(
        name: .customLong("whisper-model"),
        help: """
            Speech-to-text model HF id — Whisper or Parakeet-TDT (the \
            resident model's class picks the engine, ADR 020). Repeatable: \
            pass it more than once to make several models selectable \
            per-request via the `model` form field; the FIRST is the default.
            """
    )
    var transcriptionModels: [String] = [
        Load.defaultTranscriptionModel,
        Load.defaultParakeetTranscriptionModel,
    ]

    @Option(
        name: .customLong("diarization-model"),
        help: """
            Speaker-diarization model HF id. Repeatable: pass it more \
            than once to make several diarization models selectable \
            per-request via the `model` form field; the FIRST is the \
            default.
            """
    )
    var diarizationModels: [String] = [
        Load.defaultDiarizationModel, Load.defaultPyannoteSegmentationModel,
    ]

    @Option(
        name: .customLong("speaker-embedding-model"),
        help: """
            Speaker-embedding (voice/speaker-verification) model HF id. \
            Repeatable: pass it more than once to make several speaker- \
            embedding models selectable per-request via the `model` \
            form field; the FIRST is the default.
            """
    )
    var speakerEmbeddingModels: [String] = [
        Load.defaultSpeakerEmbeddingModel
    ]

    @Option(
        help:
            "Data dir for the embedded store (vectors + queue). Default ~/.athena."
    )
    var dataDir: String?

    @Option(
        help: "Built-in vector store byte cap. Default: budget/8.")
    var vectorCapBytes: Int?

    @Option(
        help:
            "Terminal verbosity (trace|debug|info|notice|warning|error|critical; default info). Foreground only — the macOS unified log captures everything regardless. Use `sudo log config --mode \"level:debug\" --subsystem athena` for live tuning of the unified-log gate."
    )
    var logLevel: String?

    @Flag(
        help: ArgumentHelp(
            "Set by the launchd plist. Drops the foreground stdout sink so the macOS unified log is the sole surface. Operators rarely set this directly.",
            visibility: .hidden))
    var background = false

    @Option(
        help:
            "Bearer-auth keys file (lines: `admin <key>` / `inference <key>`). Also ATHENA_ADMIN_KEYS/ATHENA_INFERENCE_KEYS env. No keys + loopback = open; + non-loopback = refuse to start."
    )
    var authKeysFile: String?

    @Option(
        help:
            "TLS certificate chain (PEM, leaf first). Set with --tls-key to serve HTTPS; one without the other is a startup error. Absent ⇒ plaintext HTTP."
    )
    var tlsCert: String?

    @Option(
        help:
            "TLS private key (PEM) matching --tls-cert. Keep it chmod 600."
    )
    var tlsKey: String?

    @Option(
        help:
            "Per-principal rate limit (sustained requests/sec). 0/absent ⇒ disabled (opt-in). Over the limit ⇒ 429 + Retry-After. Only enforced when auth is on."
    )
    var rateLimit: Double?

    @Option(
        help:
            "Rate-limit token-bucket capacity (max burst). Defaults to one second's worth of --rate-limit when unset."
    )
    var rateBurst: Int?

    @Option(
        help:
            "Max in-flight requests daemon-wide. 0/absent ⇒ unlimited. Over the cap ⇒ 429 (concurrency_limit). Only enforced when auth is on."
    )
    var maxConcurrency: Int?

    @Option(
        help:
            "Max in-flight requests per principal. 0/absent ⇒ unlimited. Over the cap ⇒ 429 (concurrency_limit)."
    )
    var maxConcurrencyPerPrincipal: Int?

    @Option(
        help:
            "Audit-log retention in days. 0/absent ⇒ keep forever. Older audit rows are pruned as the trail grows."
    )
    var auditRetentionDays: Int?

    @Option(
        help:
            "Global cap on a managed bearer token's age in days. 0/absent ⇒ no cap (tokens never expire unless minted with --ttl). A token older than this ⇒ 401, enforced relative to mint time."
    )
    var tokenMaxAgeDays: Int?

    @Option(
        help:
            "Per-request inference timeout in seconds. 0/absent ⇒ no deadline (opt-in). A decode past this ⇒ 504 (sync) / truncated stream, and the work is cancelled. Set it above your slowest legitimate generation."
    )
    var requestTimeoutSecs: Int?

    @Option(
        help:
            "Block-until-ready budget for a cold-load, in seconds (ADR 015). A request for a non-resident-but-on-disk model waits up to this long for the local load, then serves — matching peer runners. On timeout it falls back to 503 module_loading + Retry-After. Distinct from --request-timeout-secs (which bounds generation, after the load). Absent ⇒ 120s; 0 ⇒ legacy immediate-503 (revert switch). Downloads (operator pull) are never waited on."
    )
    var coldLoadWaitSecs: Int?

    @Option(
        help:
            "Max audio upload size in bytes for /v1/audio/* (ADR 017) — bounds the raw multipart body (file + MIME framing). Over the cap ⇒ 413 payload_too_large. Absent ⇒ 100 MiB (104857600). Must be positive. Worst-case transient memory ≈ this × in-flight audio requests (bound it with --max-concurrency)."
    )
    var maxAudioUploadBytes: Int?

    @Option(
        help:
            "Max video upload size in bytes for /v1/video/* (ADR 022) — bounds the raw multipart body. Over the cap ⇒ 413 payload_too_large. Absent ⇒ 1 GiB (1073741824). Must be positive. Worst-case transient memory ≈ this × in-flight video requests (bound it with --max-concurrency)."
    )
    var maxVideoUploadBytes: Int?

    @Option(
        help:
            "Max JSON request-body size in bytes for /v1/chat/completions, /v1/embeddings, etc. (ADR 017). Over the cap ⇒ 413 payload_too_large. Absent ⇒ 4 MiB (4194304). Must be positive."
    )
    var maxRequestBodyBytes: Int?

    @Option(
        help:
            "Serve-path MLX buffer-cache limit in bytes (ADR 023). Bounds MLX.Memory.cacheLimit so the reclaimable buffer pool can't grow to fill the whole Metal budget. Absent ⇒ ~1/3 of --budget-bytes; 0 ⇒ unbounded (MLX default)."
    )
    var mlxCacheLimitBytes: Int?

    @Flag(
        help:
            "Warm every module that has a configured default model (one per LLM/embedding/transcription/diarization/speaker-embedding class with an `is_default=1` allowlist row) at startup, instead of lazily on first request. The HTTP surface still comes up immediately; warms run concurrently in the background (best-effort — a per-module failure falls back to lazy load for that module). Modules without a configured default stay lazy."
    )
    var preload = false

    @Option(
        help:
            "Queue-result TTL in seconds. 0/absent ⇒ keep forever (opt-in). Terminal (done/error/canceled) results older than this are pruned on the worker idle path; pending jobs are never touched."
    )
    var queueResultTtlSecs: Int?

    @Option(
        help:
            "Max total queue job rows. 0/absent ⇒ unbounded. Over the cap, the oldest terminal results are trimmed first; pending jobs are never deleted."
    )
    var queueMaxRows: Int?

    @Option(
        help:
            "Vector-store TTL in seconds. 0/absent ⇒ keep forever (opt-in). On each upsert, vectors written longer ago than this are pruned. Vectors stored before this feature (no timestamp) are never auto-pruned."
    )
    var vectorTtlSecs: Int?

    @Flag(
        help:
            "Clear a queued job's prompt (request) blob once it finishes, so inference inputs don't persist on disk past completion. The result the client polls for is retained (bounded by --queue-result-ttl-secs). Off by default."
    )
    var dropRequestContent = false

    @Flag(
        help:
            "Encrypt the SQLite store at rest with SQLCipher (AES-256). The key resolves from ATHENA_STORE_KEY env or the Keychain; if absent, a random key is generated and stored in the Keychain. A plaintext store is migrated to encrypted on first start. Off by default."
    )
    var encryptStore = false

    @Option(
        name: .customLong("module"),
        help: """
            Rebind a module's slot on the RUNNING daemon at \
            --host:--port (no daemon-start). Pair with --id (omit ⇒ the \
            module's default). One of: llm, textEmbedding, \
            transcription, diarization, speakerEmbedding.
            """
    )
    var rebindModule: String?

    @Option(
        name: .customLong("id"),
        help: "Model id within the module's allowlist (rebind target).")
    var rebindId: String?

    @Option(
        name: .customLong("key"),
        help:
            "Bearer key for the rebind path (else ATHENA_KEY env / Keychain)."
    )
    var rebindKey: String?

    mutating func run() async throws {
        // M41.1 dual-mode: when `--module` is set, this invocation is a
        // rebind on the running daemon at --host:--port, NOT a fresh
        // daemon-start. The daemon flags below stay no-ops on this
        // path. With no --module, fall through to the daemon-foreground
        // entrypoint.
        if let m = rebindModule, !m.isEmpty {
            var opts = DaemonOptions()
            opts.host = host
            opts.port = port
            opts.key = rebindKey
            try await RemoteModels.load(opts, module: m, id: rebindId)
            return
        }
        // Centralized logging first — must precede any Logger creation
        // (Hummingbird/NIO included) so everything routes through the
        // unified-log handler (M10 → M45.1). Invalid level ⇒ warn +
        // info. `--log-level` gates only the foreground terminal
        // sink; under `--background` (launchd) it's inert because the
        // terminal handler isn't constructed.
        if let lv = logLevel, AthenaLog.level(lv) == nil {
            // M45.2 F2: this fires BEFORE bootstrap so Logger isn't
            // available yet. Prefix the line with an ISO 8601 UTC
            // timestamp so it sorts cleanly against post-bootstrap
            // TerminalLogHandler output and the unified-log clock
            // (audit-flagged: pre-bootstrap warnings used to float
            // untimestamped in the launchd-captured stderr file).
            let f = ISO8601DateFormatter()
            f.formatOptions = [
                .withInternetDateTime, .withFractionalSeconds,
            ]
            let msg =
                "\(f.string(from: Date())) warning daemon: "
                + "invalid --log-level '\(lv)', using info\n"
            FileHandle.standardError.write(Data(msg.utf8))
        }
        AthenaLog.bootstrap(
            background: background,
            terminalLevel: AthenaLog.level(logLevel) ?? .info)

        // HF download cache: if an `hf-cache` sits beside the model
        // store (the configured root, or the default), use it;
        // otherwise leave HF_HOME unset so the Hub client falls back
        // to ~/.cache/huggingface. Never override an operator-set
        // HF_HOME.
        let env = ProcessInfo.processInfo.environment
        if env["HF_HOME"] == nil, env["HF_HUB_CACHE"] == nil {
            let msRoot =
                modelStore.map {
                    URL(fileURLWithPath: $0, isDirectory: true)
                } ?? ModelStore.defaultRoot
            let hfCache = msRoot
                .deletingLastPathComponent()
                .appendingPathComponent("hf-cache", isDirectory: true)
            if FileManager.default.fileExists(atPath: hfCache.path) {
                setenv("HF_HOME", hfCache.path, 1)
            }
        }

        // Egress proxy for the model-fetch outbound: TOML [network]
        // fills any unset *_PROXY env var (operator env wins),
        // Keychain proxy creds spliced in. M13.2.
        ProxyEnv.applyConfigAndAuth()

        // HF token for gated/private on-demand fetches: export the
        // Keychain-stored token unless the operator already set one
        // (same never-override rule as HF_HOME above). M13.
        HFAuth.exportToEnv()

        // KV-cache compression codec: env ATHENA_KV_COMPRESSION > TOML
        // kv_compression > built-in none. Resolved once at startup; an
        // unrecognized value throws here and aborts the daemon
        // (fail-closed — never a silent fallback). M20.2.
        let tomlCfg = try? AthenaConfig.parse(
            file: ConfigEditor.resolvePath(nil))
        let kvCompression = try KVCompression.resolve(
            config: tomlCfg?.kvCompression)
        // M59.1/.2 — cross-request prompt-prefix KV cache (`[prompt_cache]`),
        // default OFF. Bit-identical greedy reuse on the MTP path. Enable
        // precedence mirrors kv_compression: env ATHENA_PROMPT_CACHE
        // (1/true/0/false) > TOML prompt_cache_enabled > built-in false.
        let prefixEnvEnabled = ProcessInfo.processInfo
            .environment["ATHENA_PROMPT_CACHE"]
            .map { $0 == "1" || $0.lowercased() == "true" }
        let prefixCacheEnabled =
            prefixEnvEnabled ?? tomlCfg?.promptCacheEnabled ?? false

        // M63 — DFlash lossless speculative decoding, default OFF. Same
        // precedence: env ATHENA_DFLASH (1/true/0/false) > TOML
        // dflash_enabled > built-in false.
        let dflashEnvEnabled = ProcessInfo.processInfo
            .environment["ATHENA_DFLASH"]
            .map { $0 == "1" || $0.lowercased() == "true" }
        let dflashEnabled =
            dflashEnvEnabled ?? tomlCfg?.dflashEnabled ?? false

        let config = GovernorConfig(
            totalBudgetBytes: budgetBytes,
            listenHost: host,
            listenPort: port,
            promptCacheCapBytes: promptCacheCapBytes
        )
        // M5.2: cap MLX's own allocator at the global budget so its
        // buffer pool can't overshoot the box into a Metal OOM.
        MLX.Memory.memoryLimit = config.totalBudgetBytes
        // ADR 023 G1: bound the reclaimable MLX buffer cache so it can't grow
        // to fill the whole budget (the field finding: 79 GiB ungoverned cache,
        // phys_footprint at 100 GiB over a 96 GiB budget). Configured value
        // (CLI > TOML) wins; absent ⇒ ~1/3 of budget; 0 ⇒ unbounded. nil ⇒
        // leave MLX's default. The decision is the MLX-free unit-pinned resolver.
        if let cacheLimit = GovernorMemory.resolveCacheLimit(
            configured: mlxCacheLimitBytes ?? tomlCfg?.mlxCacheLimitBytes,
            budgetBytes: config.totalBudgetBytes)
        {
            MLX.Memory.cacheLimit = cacheLimit
            Logging.Logger(label: AthenaLog.daemonLabel).notice(
                "MLX cache limit bounded to \(cacheLimit) bytes (ADR 023 G1)")
        }

        // M59.2 — build the ONE shared pool instance now (when enabled), so
        // the SAME object is injected into the LLM module (reuse) and the
        // governor (pool-byte snapshot + pressure relief). `max_bytes`
        // defaults to the governor's prompt-cache cap so the persistent pool
        // and the per-request admission guard share one cap and don't starve
        // each other. PrefixKVCache is a lock-guarded @unchecked Sendable, so
        // the governor's sync probe/relief closures can call it directly.
        let prefixMaxEntries = tomlCfg?.promptCacheMaxEntries ?? 4
        let prefixIdleTTL = tomlCfg?.promptCacheIdleTtlSecs ?? 600
        let prefixCfgBytes = tomlCfg?.promptCacheMaxBytes ?? 0
        let prefixMaxBytes =
            prefixCfgBytes > 0 ? prefixCfgBytes : config.promptCacheCapBytes
        let prefixScope =
            PrefixKVCache.ScopeMode(
                rawValue: tomlCfg?.promptCacheScope ?? "principal")
            ?? .principal
        let prefixCache: PrefixKVCache? =
            prefixCacheEnabled
            ? PrefixKVCache(
                maxEntries: prefixMaxEntries,
                maxBytes: prefixMaxBytes,
                idleTTLSecs: prefixIdleTTL,
                scope: prefixScope)
            : nil
        var prefixPoolProbe: MemoryGovernor.PromptCachePoolProbe?
        var prefixPoolRelief: MemoryGovernor.PromptCacheReliefHook?
        if let c = prefixCache {
            prefixPoolProbe = { c.poolBytesAndEntries() }
            prefixPoolRelief = { _ = c.flushIdle(); MLX.Memory.clearCache() }
        }

        // M5.1 + M5.5 (M41 follow-up): reconcile reservations to the
        // real Metal/MLX footprint. The probe is process RSS, NOT
        // `MLX.Memory.activeMemory` — the latter sees only the MLX
        // allocator's buffer pool and MISSES file-backed mmaps from
        // the HF hub cache (the embedder / whisper / diarizer / speaker
        // load that way). On a 4B embedder with HF-cache mmaps the
        // activeMemory delta is ~120 MB even though the process holds
        // ~8 GB of weights; reconcile to that under-counted number
        // and the governor's admission math is off by an order of
        // magnitude. RSS (`mach_task_basic_info.resident_size`)
        // captures both MLX bytes AND mmap'd weight pages — the
        // honest "what's in this process" number.
        // M5.2: trim the MLX buffer pool whenever a module unloads so
        // freed bytes actually leave the process.
        let governor = MemoryGovernor(
            config: config,
            memoryProbe: { Self.processResidentBytes() },
            onUnloaded: { MLX.Memory.clearCache() },
            onEvent: { id, msg in
                Logger(label: AthenaLogLabel.model(id))
                    .notice("model \(id.rawValue) \(msg)")
            },
            promptCachePoolProbe: prefixPoolProbe,
            promptCacheRelief: prefixPoolRelief)

        let store = ModelStore(
            rootDirectory: modelStore.map {
                URL(fileURLWithPath: $0, isDirectory: true)
            } ?? ModelStore.defaultRoot)

        // M42.1: open the SQLite store BEFORE building modules so each
        // module's allowlist resolves from the persisted
        // `model_allowlist` table. An empty table seeds from the
        // operator CLI flags (so a fresh install is never blank) and
        // every later edit via `/api/models/allow` survives a restart.
        let dataRoot =
            dataDir.map { URL(fileURLWithPath: $0, isDirectory: true) }
            ?? AthenaEnv.userHome()
                .appendingPathComponent(".athena", isDirectory: true)
        let dbPath = dataRoot.appendingPathComponent("athena.sqlite")
        let storeKey: String?
        if encryptStore {
            let key = try StoreKey.ensure()
            // NH1 (M66.1): if a prior encrypt-migration was interrupted
            // mid-swap, finish or roll it back before probing — so a crash
            // never leaves the daemon without a usable database.
            try AthenaStore.recoverInterruptedMigration(at: dbPath)
            if AthenaStore.isPlaintextDatabase(at: dbPath) {
                Logger(label: AthenaLogLabel.daemon).notice(
                    "encrypt_store: migrating plaintext store to encrypted")
                try AthenaStore.migrateToEncrypted(at: dbPath, key: key)
                Logger(label: AthenaLogLabel.daemon).notice(
                    "encrypt_store: store is now encrypted at rest")
            }
            storeKey = key
        } else {
            storeKey = StoreKey.resolve()
        }
        let athenaStore = try AthenaStore(path: dbPath, key: storeKey)

        // M41.2: --llm-model is the new repeatable allowlist. If unset,
        // desugar from the single --model so existing scripts keep
        // working unchanged. Either way the FIRST entry is the default
        // (used when a request omits `model`).
        let llmRefs: [String?] =
            llmModels.isEmpty ? [model] : llmModels.map { Optional($0) }
        let llmURLsFromFlags = llmRefs.map { store.resolve($0) }
        let llmIdsFromFlags = llmURLsFromFlags.map { $0.lastPathComponent }
        // M42.1: resolve each module's effective allowlist from the
        // persisted table (seeding on empty). The default sits at
        // position 0 — modules keep treating `[0]` as default and need
        // no awareness of the DB layer.
        let llmIds = await Self.resolveAllowlist(
            store: athenaStore, module: .llm, seed: llmIdsFromFlags)
        let llmURLs = llmIds.map { name in
            llmURLsFromFlags.first { $0.lastPathComponent == name }
                ?? store.resolve(name)
        }
        let modelURL = llmURLs[0]
        let embeddingIds = await Self.resolveAllowlist(
            store: athenaStore, module: .textEmbedding,
            seed: embeddingModels)
        let transcriptionIds = await Self.resolveAllowlist(
            store: athenaStore, module: .transcription,
            seed: transcriptionModels)
        let diarizationIds = await Self.resolveAllowlist(
            store: athenaStore, module: .diarization,
            seed: diarizationModels)
        let speakerEmbeddingIds = await Self.resolveAllowlist(
            store: athenaStore, module: .speakerEmbedding,
            seed: speakerEmbeddingModels)

        // M23 fork B: a VALID kv_compression codec that can't serve the
        // loaded architecture (TriAttention eviction is Qwen3.5-only)
        // warns + runs uncompressed — fail-closed is reserved for an
        // unrecognized VALUE (handled at resolve() above). An unknown
        // arch (no/unreadable config.json) is left silent — can't tell.
        if engine == .mlx, kvCompression != .none {
            let modelType = ModelConfigInfo.read(
                modelDirectory: modelURL)?.modelType
            if !kvCompression.servesArch(modelType: modelType) {
                Logger(label: AthenaLogLabel.daemon).warning(
                    """
                    kv_compression=\(kvCompression.rawValue) is inert for \
                    model type '\(modelType ?? "unknown")' — running \
                    uncompressed (\(kvCompression.rawValue) applies to \
                    Qwen3.5 models only)
                    """)
            }
        }

        // The LLM is non-evictable (the primary workload); transcription and
        // embedding remain governed stubs (real impls land in M4) and are
        // evictable so the governor can reclaim their budget under pressure.
        // M42.1: each module's allowlist resolves from the persisted
        // `model_allowlist` table (seeded above from CLI flags on first
        // boot). The default is `llmIds[0]` / etc. — already
        // DB-default-first.
        // M53: the structured-output engine (llguidance) parses
        // incrementally, so a `maxItems`-bounded schema can no longer blow
        // up memory — the M49.5 pre-compile complexity gate and its
        // `structured_max_unbounded_subarrays` TOML key are gone.
        let llm: any LLMModule
        switch engine {
        case .stub:
            llm = StubLLMModule(modelIds: llmIds)
        case .mlx:
            llm = MLXLLMModule(
                modelDirectories: llmURLs,
                modelStoreRoot: store.rootDirectory,
                parameters: .init(
                    maxTokens: maxTokens,
                    temperature: Float(temperature ?? 0.7),
                    speculative: speculative,
                    kvCompression: kvCompression),
                promptCacheCapBytes: config.promptCacheCapBytes,
                prefixCache: prefixCache,
                dflashEnabled: dflashEnabled)
        }
        let embedding: any EmbeddingModule
        switch engine {
        case .stub:
            embedding = StubEmbeddingModule(modelIds: embeddingIds)
        case .mlx:
            embedding = MLXEmbeddingModule(
                modelIds: embeddingIds,
                modelStoreRoot: store.rootDirectory)
        }
        let transcription: any TranscriptionModule
        switch engine {
        case .stub:
            transcription = StubTranscriptionModule(
                modelIds: transcriptionIds)
        case .mlx:
            transcription = MLXTranscriptionModule(
                modelIds: transcriptionIds,
                modelStoreRoot: store.rootDirectory)
        }
        let diarization: any DiarizationModule
        switch engine {
        case .stub:
            diarization = StubDiarizationModule(modelIds: diarizationIds)
        case .mlx:
            diarization = MLXDiarizationModule(
                modelIds: diarizationIds,
                modelStoreRoot: store.rootDirectory)
        }
        let speakerEmbedding: any SpeakerEmbeddingModule
        switch engine {
        case .stub:
            speakerEmbedding = StubSpeakerEmbeddingModule(
                modelIds: speakerEmbeddingIds)
        case .mlx:
            speakerEmbedding = MLXSpeakerEmbeddingModule(
                modelIds: speakerEmbeddingIds,
                modelStoreRoot: store.rootDirectory)
        }
        await governor.register(llm, evictable: false)
        await governor.register(transcription, evictable: true)
        await governor.register(embedding, evictable: true)
        await governor.register(diarization, evictable: true)
        await governor.register(speakerEmbedding, evictable: true)

        // Startup marker — `.notice` so it persists to `log show`
        // (`.info` and `.debug` are memory-only by default). Foreground
        // operators see this via the stderr TerminalLogHandler with a
        // sortable ISO timestamp; the redundant pre-M45.2 `print()`
        // bare-stdout duplicate was dropped (it landed untimestamped in
        // launchd's stdout capture and was the original "log file
        // missing timestamps" symptom that motivated docs/logging-audit.md).
        Logging.Logger(label: AthenaLog.daemonLabel).notice(
            """
            athena daemon up — engine=\(engine.rawValue) \
            model=\(modelURL.path) \
            listen=\(config.listenHost):\(config.listenPort) \
            budget=\(config.totalBudgetBytes)B
            """)

        // M7: one embedded SQLite store (vectors + queue) under the
        // data dir. The handle was opened above (M42.1 needed it for
        // allowlist resolution); the rest of the data plane reuses it.
        let vectorStore = VectorStore(
            store: athenaStore,
            capBytes: vectorCapBytes
                ?? (config.totalBudgetBytes / 8))
        let queue = RequestQueue(store: athenaStore)

        // Inbound bearer auth (M12). Fail-safe: a non-loopback bind
        // with no keys refuses to start.
        let nTokens = await athenaStore.tokenCount()
        let nUsers = await athenaStore.userCount()
        let dbHasCreds = nTokens > 0 || nUsers > 0
        let authConfig = AuthConfig.load(
            file: authKeysFile,
            env: ProcessInfo.processInfo.environment,
            log: Logger(label: AthenaLogLabel.daemon)
        ).bound(
            to: athenaStore, dbHasCredentials: dbHasCreds,
            tokenMaxAgeDays: tokenMaxAgeDays ?? 0)
        try authConfig.validateStartup(
            listenHost: config.listenHost)
        Logger(label: AthenaLogLabel.daemon).notice(
            authConfig.isEnabled
                ? "auth: enabled (RBAC; bearer→user→roles, env/file + DB)"
                : "auth: DISABLED (loopback, no credentials)")
        // ADR 004 (warn-only, audit A2/K1) — an auth-on daemon on a
        // non-loopback bind serving plaintext starts, but loudly: bearer
        // tokens and the session cookie cross the wire in clear. Mirrors
        // the doctor TLS-posture finding (M28.2). No fail-closed, no
        // `--insecure` gate. Loopback binds (incl. the e2e suite) never
        // trip this.
        do {
            let loopback: Set<String> = ["127.0.0.1", "::1", "localhost"]
            let plaintext = tlsCert == nil && tlsKey == nil
            if authConfig.isEnabled, plaintext,
                !loopback.contains(config.listenHost)
            {
                Logger(label: AthenaLogLabel.daemon).warning(
                    """
                    TLS: serving plaintext on non-loopback \
                    \(config.listenHost) with auth ENABLED — bearer tokens \
                    and the session cookie are exposed on the wire. Set \
                    tls_cert/tls_key or front the daemon with a \
                    TLS-terminating reverse proxy (see docs/quickstart.md).
                    """)
            }
        }

        var server = AthenaServer(
            config: config, governor: governor, llm: llm,
            embedding: embedding, transcription: transcription,
            diarization: diarization,
            speakerEmbedding: speakerEmbedding,
            vectorStore: vectorStore,
            queue: queue, store: athenaStore,
            modelName: modelURL.lastPathComponent,
            modelStoreRoot: store.rootDirectory, auth: authConfig,
            tlsCertPath: tlsCert, tlsKeyPath: tlsKey,
            rateLimit: rateLimit ?? 0, rateBurst: rateBurst ?? 0,
            maxConcurrency: maxConcurrency ?? 0,
            maxConcurrencyPerPrincipal: maxConcurrencyPerPrincipal ?? 0,
            auditRetentionDays: auditRetentionDays ?? 0,
            requestTimeoutSecs: requestTimeoutSecs ?? 0,
            coldLoadWaitSecs: Double(coldLoadWaitSecs ?? 120),
            maxAudioUploadBytes: maxAudioUploadBytes ?? 104_857_600,
            maxVideoUploadBytes: maxVideoUploadBytes ?? 1_073_741_824,
            maxRequestBodyBytes: maxRequestBodyBytes ?? 4_194_304,
            preload: preload,
            prefixCache: prefixCache,
            queueResultTtlSecs: queueResultTtlSecs ?? 0,
            queueMaxRows: queueMaxRows ?? 0,
            vectorTtlSecs: vectorTtlSecs ?? 0,
            dropRequestContent: dropRequestContent)
        // M54.3 — operator-action pull: at startup, fetch any configured
        // model that has an HF-style id and isn't in the store, in the
        // BACKGROUND. The HTTP surface comes up immediately (server.run
        // below); an inference request for a not-yet-present model gets
        // 503 via the governor's `pulling` flag until the pull lands.
        // Bare store-dir ids (the LLM convention) aren't Hub-pullable and
        // are skipped — they're provisioned via `athena pull`/`convert`.
        let configuredForPull: [(ModuleID, [String])] = [
            (.llm, llmIds), (.textEmbedding, embeddingIds),
            (.transcription, transcriptionIds),
            (.diarization, diarizationIds),
            (.speakerEmbedding, speakerEmbeddingIds),
        ]
        let pullStoreRoot = store.rootDirectory
        Task.detached {
            await Self.pullMissingConfigured(
                configuredForPull, storeRoot: pullStoreRoot,
                governor: governor)
        }
        // M60.2 — hold a system-sleep power assertion for the lifetime of the
        // serve loop. Without it an unattended Mac sleeps and SUSPENDS the
        // daemon mid-generation (the confirmed root cause of the consuming application's
        // overnight "freeze"/timeouts — see docs/m60-plan.md). Released on
        // shutdown via the defer. Acquisition failure is non-fatal: we log it
        // and keep serving (/healthz reports the held state).
        let powerAssertion = PowerAssertion(
            reason: "Athena serving inference (prevent idle system sleep)")
        if powerAssertion.acquire() {
            Logger(label: AthenaLogLabel.daemon).notice(
                "power: holding PreventUserIdleSystemSleep (the machine will not idle-sleep while serving)")
        } else {
            Logger(label: AthenaLogLabel.daemon).warning(
                "power: could NOT acquire a sleep assertion — an unattended Mac may suspend inference mid-request. Run under `caffeinate -s` or set `pmset -c sleep 0` as a workaround.")
        }
        defer { powerAssertion.release() }
        server.powerAssertionHeld = powerAssertion.isHeld
        // M60.3 — sudoless GPU-clock telemetry on /healthz (IOReport via the
        // AppleSiliconMetrics package). Background-sampled, nil-safe: if
        // IOReport is unavailable the fields are simply omitted.
        let gpuProbe = GPUTelemetryProbe()
        server.gpuProbe = gpuProbe
        if gpuProbe.isAvailable {
            Logger(label: AthenaLogLabel.daemon).notice(
                "gpu: IOReport telemetry available — reporting gpuClockMHz on /healthz")
        }
        try await server.run()
    }

    /// M54.3 — pull each configured model that is HF-pullable (id contains
    /// `/`) and not yet materialized in the store, marking the governor
    /// `pulling` for that module while its fetch is in flight (so inference
    /// 503s rather than auto-downloading). Best-effort + sequential (avoid
    /// concurrent multi-GB fetches); a per-model failure is logged and
    /// skipped (the model stays absent → its requests keep 503ing).
    static func pullMissingConfigured(
        _ configured: [(ModuleID, [String])], storeRoot: URL,
        governor: MemoryGovernor
    ) async {
        let log = Logger(label: AthenaLogLabel.daemon)
        for (module, ids) in configured {
            for id in ids
            where id.contains("/")
                && ModelStoreLayout.localDirectory(
                    for: id, storeRoot: storeRoot) == nil
            {
                await governor.setPulling(module, true)
                log.notice(
                    """
                    operator-pull: fetching \(id) for \
                    \(module.rawValue) (not in store)
                    """)
                do {
                    _ = try await ModelPull.pull(id: id, into: storeRoot)
                    log.notice("operator-pull: \(id) ready")
                } catch {
                    log.warning(
                        """
                        operator-pull: \(id) failed: \
                        \(ModelPull.friendlyError(error))
                        """)
                }
                await governor.setPulling(module, false)
            }
        }
    }

    /// Resolve a module's effective allowlist from the persisted
    /// `model_allowlist` table. On every boot any id in the CLI seed
    /// that isn't already in the DB is idempotently added; the CLI
    /// never REMOVES rows (operator `athena allowlist rm` is preserved)
    /// and never re-flips the default once the table has any rows
    /// (operator `athena allowlist default` is preserved). On a TRULY
    /// fresh table the first seed entry becomes the default, so a new
    /// install still gets a sensible default from the CLI/TOML alone.
    ///
    /// Behaviour history. M42.1 originally seeded the table on first
    /// boot and treated later CLI edits as no-ops (DB wins); M43.4
    /// added a divergence notice. M44.2 turned the CLI into a merge
    /// writer on every boot so the operator's `--*-model` intent
    /// actually takes effect on restart — at the cost that a flag
    /// they removed via `athena allowlist rm` will come back unless
    /// they also strike it from the CLI/TOML.
    static func resolveAllowlist(
        store: AthenaStore, module: ModuleID, seed: [String]
    ) async -> [String] {
        let seedClean = seed.filter { !$0.isEmpty }
        let emptyAtStart =
            await store.modelAllowlistCount(
                module: module.rawValue) == 0
        for (idx, id) in seedClean.enumerated() {
            // Only the FIRST seed on a fresh table becomes default;
            // every subsequent boot (or later seed in this list) adds
            // with isDefault=false so an operator's `allowlist default`
            // choice is never silently overwritten.
            let asDefault = emptyAtStart && idx == 0
            try? await store.addModelAllowlist(
                module: module.rawValue, id: id,
                isDefault: asDefault)
        }
        let rows = await store.listModelAllowlist(
            module: module.rawValue)
        guard !rows.isEmpty else { return seedClean }
        // Default first, then declaration order.
        let def = rows.first { $0.isDefault } ?? rows[0]
        var ordered = [def.id]
        for row in rows where row.id != def.id {
            ordered.append(row.id)
        }
        // M54 — store-identity collision guard. Two configured ids that
        // share a store-dir basename (e.g. `a/m` and `b/m`) both resolve
        // to the same store directory, so a request for either silently
        // loads whichever is first. The store keys by basename, so it
        // can't even hold both — warn loudly rather than alias quietly.
        let identities = ordered.map { $0.modelStoreIdentity }
        let collided = Set(
            identities.filter { id in
                identities.filter { $0.caseInsensitiveCompare(id)
                    == .orderedSame }.count > 1
            })
        if !collided.isEmpty {
            Logger(label: AthenaLogLabel.daemon).warning(
                """
                \(module.rawValue) allowlist has ids sharing a store-dir \
                name \(collided.sorted()) — they resolve to the same local \
                directory; requests will load whichever is declared first. \
                Use distinct model names or remove the duplicate.
                """)
        }
        return ordered
    }

    /// Process resident-set bytes (RSS) for the governor's M5 reconcile.
    /// Includes the file-backed mmaps (the HF hub cache / model weights),
    /// MLX allocator pages, heap, and stack — "what this process holds"
    /// in its address space. The reconcile reserves against the steady
    /// post-load weight footprint, so RSS (not `phys_footprint`) is the
    /// right gauge here: it excludes the transient Metal/GPU KV/prompt-
    /// cache buffers that come and go per request and would otherwise
    /// inflate a module's reservation. (M55 routed this through the
    /// shared `ProcessMemory` probe; `phys_footprint` — which DOES count
    /// those GPU buffers — is surfaced in the heartbeat + /healthz for
    /// observability.) Returns 0 on a Mach failure (governor then
    /// degrades to the pre-load estimate, the prior behaviour).
    static func processResidentBytes() -> Int {
        ProcessMemory.residentBytes()
    }
}
