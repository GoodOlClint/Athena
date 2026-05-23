import ArgumentParser
import AthenaClient
import AthenaCore
import AthenaDeploy
import AthenaEmbedding
import AthenaLLM
import AthenaStore
import AthenaTranscription
import Foundation
import Logging
import MLX

enum Engine: String, CaseIterable, ExpressibleByArgument {
    case mlx
    case stub
}

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
    static let defaultDiarizationModel =
        "mlx-community/diar_streaming_sortformer_4spk-v2.1-fp16"
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

    @Flag(help: "Enable MTP speculative decoding (greedy/temp 0 only).")
    var speculative = false

    @Option(help: "Model directory path, or a name under the model store.")
    var model: String?

    @Option(help: "Model store root. Default: ~/.athena/models.")
    var modelStore: String?

    @Option(
        help:
            "Text-embedding model HF id (default BAAI/bge-small-en-v1.5).")
    var embeddingModel: String = Load.defaultEmbeddingModel

    @Option(
        help:
            "Speech-to-text model HF id (default mlx-community/whisper-large-v3-turbo)."
    )
    var transcriptionModel: String = Load.defaultTranscriptionModel

    @Option(
        help:
            "Speaker-diarization model HF id (default mlx-community/diar_streaming_sortformer_4spk-v2.1-fp16)."
    )
    var diarizationModel: String = Load.defaultDiarizationModel

    @Option(
        help:
            "Speaker-embedding (voice/speaker-verification) model HF id (default aufklarer/WeSpeaker-ResNet34-LM-MLX)."
    )
    var speakerEmbeddingModel: String = Load.defaultSpeakerEmbeddingModel

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
            "Log level trace|debug|info|notice|warning|error|critical (default info; debug/trace = max)."
    )
    var logLevel: String?

    @Option(
        help:
            "Opt-in remote syslog sink udp://host[:514] (logs only; the one passive-oracle exception; default off)."
    )
    var syslogRemote: String?

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
            "Per-request inference timeout in seconds. 0/absent ⇒ no deadline (opt-in). A decode past this ⇒ 504 (sync) / truncated stream, and the work is cancelled. Set it above your slowest legitimate generation."
    )
    var requestTimeoutSecs: Int?

    @Flag(
        help:
            "Warm the LLM at startup instead of lazily on first request. The HTTP surface still comes up immediately; the warm runs in the background (best-effort — a failure falls back to lazy load)."
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

    mutating func run() async throws {
        // Centralized logging first — must precede any Logger creation
        // (Hummingbird/NIO included) so everything routes through the
        // stdout + unified-logging multiplex (M10). Invalid level ⇒
        // warn + info.
        if let lv = logLevel, AthenaLog.level(lv) == nil {
            let msg =
                "warning: invalid --log-level '\(lv)', using info\n"
            FileHandle.standardError.write(Data(msg.utf8))
        }
        AthenaLog.bootstrap(
            level: AthenaLog.level(logLevel) ?? .info,
            syslogRemote: syslogRemote)

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

        let config = GovernorConfig(
            totalBudgetBytes: budgetBytes,
            listenHost: host,
            listenPort: port,
            promptCacheCapBytes: promptCacheCapBytes
        )
        // M5.2: cap MLX's own allocator at the global budget so its
        // buffer pool can't overshoot the box into a Metal OOM.
        MLX.Memory.memoryLimit = config.totalBudgetBytes

        // M5.1: reconcile reservations to the real Metal/MLX footprint.
        // M5.2: trim the MLX buffer pool whenever a module unloads so
        // freed bytes actually leave the process.
        let governor = MemoryGovernor(
            config: config,
            memoryProbe: { MLX.Memory.activeMemory },
            onUnloaded: { MLX.Memory.clearCache() },
            onEvent: { id, msg in
                Logger(label: AthenaLogLabel.model(id))
                    .notice("model \(id.rawValue) \(msg)")
            })

        let store = ModelStore(
            rootDirectory: modelStore.map {
                URL(fileURLWithPath: $0, isDirectory: true)
            } ?? ModelStore.defaultRoot)
        let modelURL = store.resolve(model)

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
        let llm: any LLMModule
        switch engine {
        case .stub:
            llm = StubLLMModule()
        case .mlx:
            llm = MLXLLMModule(
                modelDirectory: modelURL,
                parameters: .init(
                    maxTokens: maxTokens,
                    temperature: Float(temperature ?? 0.7),
                    speculative: speculative,
                    kvCompression: kvCompression),
                promptCacheCapBytes: config.promptCacheCapBytes)
        }
        // Embeddings: real MLX module under the mlx engine, stub under
        // the stub engine. Evictable — the LLM is the primary workload.
        let embedding: any EmbeddingModule
        switch engine {
        case .stub:
            embedding = StubEmbeddingModule()
        case .mlx:
            embedding = MLXEmbeddingModule(modelId: embeddingModel)
        }
        // Transcription: real Whisper under the mlx engine, stub under
        // the stub engine. Evictable — the LLM is the primary workload.
        let transcription: any TranscriptionModule
        switch engine {
        case .stub:
            transcription = StubTranscriptionModule()
        case .mlx:
            transcription = MLXTranscriptionModule(
                modelId: transcriptionModel)
        }
        // Diarization: vendored Sortformer under mlx, stub under stub.
        let diarization: any DiarizationModule
        switch engine {
        case .stub:
            diarization = StubDiarizationModule()
        case .mlx:
            diarization = MLXDiarizationModule(
                modelId: diarizationModel)
        }
        // Speaker embeddings: vendored WeSpeaker under mlx, stub under
        // stub. Evictable — the LLM is the primary workload.
        let speakerEmbedding: any SpeakerEmbeddingModule
        switch engine {
        case .stub:
            speakerEmbedding = StubSpeakerEmbeddingModule()
        case .mlx:
            speakerEmbedding = MLXSpeakerEmbeddingModule(
                modelId: speakerEmbeddingModel)
        }
        await governor.register(llm, evictable: false)
        await governor.register(transcription, evictable: true)
        await governor.register(embedding, evictable: true)
        await governor.register(diarization, evictable: true)
        await governor.register(speakerEmbedding, evictable: true)

        print(
            "athena: engine=\(engine.rawValue) "
                + "model=\(modelURL.path) "
                + "budget=\(config.totalBudgetBytes)B "
                + "listen=\(config.listenHost):\(config.listenPort)")

        // Persisted unified-log marker (notice ⇒ survives `log show`);
        // also confirms the centralized-logging pipeline is live.
        Logging.Logger(label: AthenaLog.daemonLabel).notice(
            """
            athena daemon up — engine=\(engine.rawValue) \
            listen=\(config.listenHost):\(config.listenPort) \
            budget=\(config.totalBudgetBytes)B
            """)

        // M7: one embedded SQLite store (vectors + queue) under the
        // data dir. Governor-capped resident vector working set.
        let dataRoot =
            dataDir.map { URL(fileURLWithPath: $0, isDirectory: true) }
            ?? AthenaEnv.userHome()
                .appendingPathComponent(".athena", isDirectory: true)
        // M34.3b: at-rest encryption. `encrypt_store` ensures a key
        // exists (env > Keychain, else mint+store — fail-closed) and
        // migrates a plaintext store to SQLCipher-encrypted on first
        // start. Otherwise honor an already-configured key if present, so
        // an existing encrypted store still opens.
        let dbPath = dataRoot.appendingPathComponent("athena.sqlite")
        let storeKey: String?
        if encryptStore {
            let key = try StoreKey.ensure()
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
        ).bound(to: athenaStore, dbHasCredentials: dbHasCreds)
        try authConfig.validateStartup(
            listenHost: config.listenHost)
        Logger(label: AthenaLogLabel.daemon).notice(
            authConfig.isEnabled
                ? "auth: enabled (RBAC; bearer→user→roles, env/file + DB)"
                : "auth: DISABLED (loopback, no credentials)")

        let server = AthenaServer(
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
            preload: preload,
            queueResultTtlSecs: queueResultTtlSecs ?? 0,
            queueMaxRows: queueMaxRows ?? 0,
            vectorTtlSecs: vectorTtlSecs ?? 0,
            dropRequestContent: dropRequestContent)
        try await server.run()
    }
}
