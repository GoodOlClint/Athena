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

    @Option(
        help:
            "Cap on prompt length in tokens (post chat-template). Absent ⇒ unbounded. Prefill attention is O(seq²); past a model/hardware-specific length a single score buffer exceeds Metal's per-buffer limit and the MLX eval aborts the daemon — this refuses an oversized prompt with a 400 (input_too_long) before prefill. Set to a value your model+GPU can hold (calibration knob)."
    )
    var maxPromptTokens: Int?

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
            "Data dir for the embedded store (auth/audit/usage). Default ~/.athena."
    )
    var dataDir: String?

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
            "Per-principal token budget per period (ADR 041). 0/absent ⇒ unlimited. Over budget ⇒ 429 (quota_exceeded) on the token-bearing routes only. Auth-on only — inert in loopback dev mode. A per-user override (`athena auth user budget`) wins over this default."
    )
    var tokenBudget: Int?

    @Option(
        help:
            "Budget period for --token-budget: day | month (default month). Boundaries are LOCAL midnight / first of the local month."
    )
    var tokenBudgetWindow: String?

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

    @Option(
        help:
            "Governor admission accounting mode (ADR 023 G2): 'footprint' (default) meters admission against max(committed, reserved) where committed = phys_footprint − reclaimable MLX cache, so the governor stops overcommitting; 'estimate' reverts to the pre-G2 reservation-only denominator. Unknown ⇒ footprint."
    )
    var governorAdmissionMode: String?

    @Flag(
        help:
            "Warm every module that has a configured default model (one per LLM/embedding/transcription/diarization/speaker-embedding class with an `is_default=1` allowlist row) at startup, instead of lazily on first request. The HTTP surface still comes up immediately; warms run concurrently in the background (best-effort — a per-module failure falls back to lazy load for that module). Modules without a configured default stay lazy."
    )
    var preload = false

    @Flag(
        help:
            "Encrypt the SQLite store at rest with SQLCipher (AES-256). The key resolves from ATHENA_STORE_KEY env or the Keychain; if absent, a random key is generated and stored in the Keychain. A plaintext store is migrated to encrypted on first start. Off by default."
    )
    var encryptStore = false

    @Flag(
        help:
            "Force a persistent on-disk SQLite store even in loopback dev mode with no credentials (ADR 025 S4). By default such a run is STATELESS — no athena.sqlite is created and audit/usage are not persisted. Set this to keep audit/usage on disk anyway. Off by default; ignored (always persistent) when auth keys are configured, a store already exists, encrypt_store is on, or the bind is non-loopback."
    )
    var persistStore = false

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

        // ADR 024 Tier 2 — disable core dumps before any secret (HF token,
        // store key) or model weight touches memory, so a later crash can never
        // dump the address space to a core file. Always on; the opt-in debugger
        // denial is applied below once config is parsed.
        if ProcessHardening.disableCoreDumps() {
            Logger(label: AthenaLogLabel.daemon).notice(
                "hardening: core dumps disabled (RLIMIT_CORE=0)")
        } else {
            Logger(label: AthenaLogLabel.daemon).warning(
                "hardening: could not disable core dumps")
        }

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
            let hfCache =
                msRoot
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
        // ADR 037 slice 1 — static plist: the launchd plist no longer freezes
        // config into `ProgramArguments`, so the daemon reads the FULL TOML at
        // boot here. Precedence per key: an explicit CLI flag wins; else the
        // TOML value; else the built-in default. (Ceiling: for the four
        // built-in-defaulted knobs — host/port/max-tokens/engine — an explicit
        // flag set to the exact built-in default cannot override a differing
        // TOML value; edit the TOML for those. Flags only turn ON.)
        if let t = tomlCfg {
            budgetBytes = budgetBytes ?? t.budgetBytes
            model = model ?? t.model
            // Per-module default model ids (ADR 026): the TOML value replaces
            // the seed default only when the operator didn't pass the flag
            // (array still equals its seed) — mirrors the old plist, which set
            // the flag to a single TOML value.
            if embeddingModels == [Load.defaultEmbeddingModel],
                let m = t.embeddingModel
            {
                embeddingModels = [m]
            }
            if transcriptionModels == [
                Load.defaultTranscriptionModel,
                Load.defaultParakeetTranscriptionModel,
            ], let m = t.transcriptionModel {
                transcriptionModels = [m]
            }
            if diarizationModels == [
                Load.defaultDiarizationModel,
                Load.defaultPyannoteSegmentationModel,
            ], let m = t.diarizationModel {
                diarizationModels = [m]
            }
            if speakerEmbeddingModels == [Load.defaultSpeakerEmbeddingModel],
                let m = t.speakerEmbeddingModel
            {
                speakerEmbeddingModels = [m]
            }
            modelStore = modelStore ?? t.modelStore
            dataDir = dataDir ?? t.dataDir
            logLevel = logLevel ?? t.logLevel
            maxPromptTokens = maxPromptTokens ?? t.maxPromptTokens
            temperature = temperature ?? t.temperature.flatMap(Double.init)
            authKeysFile = authKeysFile ?? t.authKeysFile
            tlsCert = tlsCert ?? t.tlsCert
            tlsKey = tlsKey ?? t.tlsKey
            rateLimit = rateLimit ?? t.rateLimit.flatMap(Double.init)
            rateBurst = rateBurst ?? t.rateBurst
            tokenBudget = tokenBudget ?? t.tokenBudget
            tokenBudgetWindow = tokenBudgetWindow ?? t.tokenBudgetWindow
            maxConcurrency = maxConcurrency ?? t.maxConcurrency
            maxConcurrencyPerPrincipal =
                maxConcurrencyPerPrincipal ?? t.maxConcurrencyPerPrincipal
            auditRetentionDays = auditRetentionDays ?? t.auditRetentionDays
            tokenMaxAgeDays = tokenMaxAgeDays ?? t.tokenMaxAgeDays
            requestTimeoutSecs = requestTimeoutSecs ?? t.requestTimeoutSecs
            coldLoadWaitSecs = coldLoadWaitSecs ?? t.coldLoadWaitSecs
            maxAudioUploadBytes = maxAudioUploadBytes ?? t.maxAudioUploadBytes
            maxVideoUploadBytes = maxVideoUploadBytes ?? t.maxVideoUploadBytes
            maxRequestBodyBytes = maxRequestBodyBytes ?? t.maxRequestBodyBytes
            mlxCacheLimitBytes = mlxCacheLimitBytes ?? t.mlxCacheLimitBytes
            governorAdmissionMode =
                governorAdmissionMode ?? t.governorAdmissionMode
            if host == "127.0.0.1" { host = t.listenHost }
            if port == GovernorConfig.defaultPort { port = t.listenPort }
            if maxTokens == 1024, let m = t.maxTokens { maxTokens = m }
            if engine == .mlx, let e = t.engine,
                let ee = Engine(rawValue: e)
            {
                engine = ee
            }
            speculative = speculative || (t.speculative ?? false)
            preload = preload || (t.preload ?? false)
            encryptStore = encryptStore || (t.encryptStore ?? false)
            persistStore = persistStore || (t.persistStore ?? false)
        }
        let kvCompression = try KVCompression.resolve(
            config: tomlCfg?.kvCompression)
        // ADR 041 — resolve the budget window fail-closed like kv_compression: a
        // `--token-budget-window` typo (which bypasses the TOML parse check)
        // must refuse to start rather than quietly enforce a window nobody
        // asked for.
        let quotaWindow = try QuotaWindow.resolve(tokenBudgetWindow)
        // ADR 024 Tier 2 (opt-in) — deny debugger attach. Precedence mirrors
        // the other startup toggles: env ATHENA_DENY_DEBUGGER (1/true/0/false)
        // > TOML deny_debugger_attach > built-in false. Redundant with the
        // Tier-1 lockdown (which already denies the task port) and
        // kernel-bypassable, so off by default.
        let denyDbgEnv = ProcessInfo.processInfo
            .environment["ATHENA_DENY_DEBUGGER"]
            .map { $0 == "1" || $0.lowercased() == "true" }
        let denyDebugger = denyDbgEnv ?? tomlCfg?.denyDebuggerAttach ?? false
        if denyDebugger {
            let ok = ProcessHardening.denyDebuggerAttachNow()
            Logger(label: AthenaLogLabel.daemon).notice(
                "hardening: debugger attach denied (PT_DENY_ATTACH\(ok ? "" : " — call failed"))")
        }

        let config = GovernorConfig(
            totalBudgetBytes: budgetBytes,
            listenHost: host,
            listenPort: port,
            promptCacheCapBytes: promptCacheCapBytes
        )
        // Capture MLX runtime errors before they hit the default
        // `fatalError`, which prints to stderr that launchd discards — so a
        // crash during eval (e.g. the gemma4-MoE long-context graphify repro)
        // leaves no message anywhere. MUST be the global handler, not
        // `withError {}`: async_eval errors fire on MLX's own worker thread
        // (`default-qos.cooperative`) where withError's @TaskLocal handler is
        // invisible (mlx-swift `ErrorHandler.dispatch`). ponytail:
        // `setErrorHandler` is deprecated upstream, but its replacements
        // (`withError`/`withErrorHandler`) are TaskLocal-scoped and provably
        // can't see cross-thread async-eval faults — this is the only lever.
        // ADR 030 Part 2 (WP2) — degrade a RECOGNIZED allocation/buffer-size
        // fault (`[metal::malloc]` / "maximum allowed buffer size" — a
        // pre-submission device-cap rejection that leaves the allocator intact)
        // to a classified 503 instead of aborting: record it in the process-
        // global latch and RETURN, keeping the daemon alive. The offending
        // tenant's gated span (`InferenceGate.withExclusiveExecution`, ADR 029
        // guarantees single-tenant execution) consumes the latch on exit and
        // throws `metalOutOfMemory` → 503; the decode loop breaks on the latch
        // first so it never touches the invalid arrays. Any UNrecognized fault
        // still re-`fatalError`s (MLX state undefined ⇒ clean launchd restart
        // with a logged message, as before). Gated on BOTH default-on revert
        // knobs: the degrade needs the InferenceGate to attribute+consume the
        // fault, so if either is off we fall back to the safe fatalError.
        MLX.setErrorHandler({ message, _ in
            let m = message.map { String(cString: $0) } ?? "(no message)"
            if MetalFaultDegrade.enabled, InferenceGate.enabled,
                AthenaError.isMetalOOMMessage(m)
            {
                MetalFaultLatch.shared.record(m)
                Logging.Logger(label: "athena.mlx").critical(
                    "MLX allocation fault degraded to 503 (daemon kept alive): \(m)")
                return
            }
            Logging.Logger(label: "athena.mlx").critical(
                "MLX eval error: \(m)")
            fatalError("MLX eval error: \(m)")
        })

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
            // Tell the decode loops MLX now bounds the cache, so they can drop
            // their legacy per-256-token clearCache() (which defeats the pool).
            GovernorMemory.serveCacheBounded = true
            Logging.Logger(label: AthenaLog.daemonLabel).notice(
                "MLX cache limit bounded to \(cacheLimit) bytes (ADR 023 G1)")
        }

        // ADR 029 — the inference execution gate (one Metal-executing tenant at
        // a time). Default ON; precedence env > TOML > built-in true. Set once
        // here, before the HTTP surface accepts requests.
        if let gateEnv = ProcessInfo.processInfo
            .environment["ATHENA_INFERENCE_GATE"]?.lowercased()
        {
            InferenceGate.enabled = !["0", "false", "no", "off"].contains(gateEnv)
        } else {
            InferenceGate.enabled = tomlCfg?.inferenceGateEnabled ?? true
        }
        if !InferenceGate.enabled {
            Logging.Logger(label: AthenaLog.daemonLabel).notice(
                "inference execution gate DISABLED (ADR 029 revert knob)")
        }

        // ADR 039 S2 — continuous batching (fixed-batch, minimal core). DEFAULT
        // OFF; opt-in via env > TOML. The admission provider is wired after the
        // governor is constructed (below). Enable only when /metrics shows
        // routine gateWaiters >= 2 (ADR 038 trigger).
        if let batchEnv = ProcessInfo.processInfo
            .environment["ATHENA_BATCHING"]?.lowercased()
        {
            BatchScheduler.enabled = ["1", "true", "yes", "on"].contains(batchEnv)
        } else {
            BatchScheduler.enabled = tomlCfg?.batchingEnabled ?? false
        }
        if BatchScheduler.enabled {
            Logging.Logger(label: AthenaLog.daemonLabel).notice(
                "continuous batching ENABLED (ADR 039, default-off knob)")
        }

        // ADR 030 Part 2 (WP2) — degrade recognized MLX allocation faults to a
        // 503 instead of aborting the daemon. Default ON; env > TOML > true.
        if let degradeEnv = ProcessInfo.processInfo
            .environment["ATHENA_METAL_FAULT_DEGRADE"]?.lowercased()
        {
            MetalFaultDegrade.enabled =
                !["0", "false", "no", "off"].contains(degradeEnv)
        } else {
            MetalFaultDegrade.enabled = tomlCfg?.metalFaultDegrade ?? true
        }
        if !MetalFaultDegrade.enabled {
            Logging.Logger(label: AthenaLog.daemonLabel).notice(
                "MLX allocation-fault degrade DISABLED (ADR 030 P2 revert knob — recognized OOM will re-fatal, as pre-WP2)"
            )
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
        // ADR 023 G2: admission accounting mode (CLI > TOML > default
        // `footprint`). `footprint` meters admission against the real Metal
        // footprint (`committed = phys_footprint − reclaimable cache`) so the
        // governor stops overcommitting; `estimate` is the revert switch.
        let admissionMode = GovernorMemory.AdmissionMode.parse(
            governorAdmissionMode ?? tomlCfg?.governorAdmissionMode)
        if admissionMode == .estimate {
            Logger(label: AthenaLogLabel.daemon).notice(
                "governor admission mode: estimate (ADR 023 G2 revert switch — metering reservations only, not the live footprint)"
            )
        }
        let governor = MemoryGovernor(
            config: config,
            memoryProbe: { Self.processResidentBytes() },
            onUnloaded: { MLX.Memory.clearCache() },
            onEvent: { id, msg in
                Logger(label: AthenaLogLabel.model(id))
                    .notice("model \(id.rawValue) \(msg)")
            },
            // ADR 023 G2 — the live-footprint probe (phys_footprint + the
            // reclaimable MLX cache) and the cache-reclaim hook. Both read MLX/
            // Mach counters HERE, at the serve seam; the admission algebra they
            // feed stays MLX-free in `GovernorMemory` (ADR 008/009).
            footprintProbe: {
                (
                    physFootprint: ProcessMemory.sample().physFootprint,
                    cacheBytes: MLX.Memory.cacheMemory
                )
            },
            // ADR 029 WP1 — gated reclaim: the MLX free runs under the gate.
            reclaimCache: {
                try? await InferenceGate.shared.withExclusiveExecution {
                    MLX.Memory.clearCache()
                }
            },
            admissionMode: admissionMode)

        // ADR 039 S2 — feed the batch scheduler the governor's live admission
        // inputs (denominator + budget) so per-sequence KV reservations meter
        // against the same ADR-023 truthful number as module-load admission.
        BatchScheduler.admissionInputsProvider = { await governor.admissionInputs() }

        let store = ModelStore(
            rootDirectory: modelStore.map {
                URL(fileURLWithPath: $0, isDirectory: true)
            } ?? ModelStore.defaultRoot)

        // Open the auth/audit/usage SQLite store (ADR 025/026: queue, vector,
        // and allowlist tenants are gone) before building modules. ADR 025 S4
        // may make this an ephemeral in-memory store (see the mode decision
        // below) — a loopback dev daemon with no credentials writes no file.
        let dataRoot =
            dataDir.map { URL(fileURLWithPath: $0, isDirectory: true) }
            ?? AthenaEnv.userHome()
            .appendingPathComponent(".athena", isDirectory: true)
        let dbPath = dataRoot.appendingPathComponent("athena.sqlite")

        // ADR 025 S4 — decide whether to persist the auth/audit/usage store
        // on disk or run STATELESS (in-memory, no file). Bootstrap auth keys
        // are loaded here once (reused below for `.bound`) so the decision can
        // see whether anything actually needs to authenticate. A loopback dev
        // daemon with no credentials creates no `athena.sqlite` at all.
        let bootstrapAuth = AuthConfig.load(
            file: authKeysFile,
            env: ProcessInfo.processInfo.environment,
            log: Logger(label: AthenaLogLabel.daemon))
        let storeMode = StoreMode.resolve(
            hasBootstrapKeys: bootstrapAuth.isEnabled,
            dbFileExists: FileManager.default.fileExists(atPath: dbPath.path),
            isLoopback: StoreMode.isLoopback(config.listenHost),
            encryptStore: encryptStore,
            persistOverride: persistStore)

        let athenaStore: AthenaStore
        switch storeMode {
        case .ephemeral:
            Logger(label: AthenaLogLabel.daemon).notice(
                """
                store: ephemeral (stateless loopback) — no athena.sqlite on \
                disk; audit/usage are kept in memory only and not persisted
                """)
            athenaStore = try AthenaStore(ephemeral: true)
        case .persistent:
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
            athenaStore = try AthenaStore(path: dbPath, key: storeKey)
        }

        // ADR 026 — selection is store-backed; there is no allowlist. Each
        // module's per-request selectable set is the store classified by
        // ModelSupport; the per-module DEFAULT comes from config (the `--model`
        // / `--*-model` flags, themselves seeded from the TOML keys via the
        // launchd plist). The FIRST configured id is the default (used when a
        // request omits `model`); `athena init` / startup still pulls every
        // configured id so both engine families (whisper+parakeet,
        // sortformer+pyannote) land in the store and become selectable.
        let llmDefaultName =
            llmModels.first ?? model ?? ModelStore.defaultModelName
        let modelURL = store.resolve(llmDefaultName)
        // Stub seed sets (the stub has no disk; these stand in for the store).
        let llmStubIds =
            llmModels.isEmpty
            ? [model ?? ModelStore.defaultModelName] : llmModels
        let embeddingDefault = embeddingModels.first
        let transcriptionDefault = transcriptionModels.first
        let diarizationDefault = diarizationModels.first
        let speakerEmbeddingDefault = speakerEmbeddingModels.first

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
        // embedding are evictable so the governor can reclaim their budget
        // under pressure.
        // ADR 026: each module's selectable set is the model store classified
        // by ModelSupport; the per-module default is `configuredDefault` (the
        // flag's first id). The stub engine has no disk, so it gets the
        // configured ids as a stand-in selectable set + the same default.
        // M53: the structured-output engine (llguidance) parses incrementally,
        // so a `maxItems`-bounded schema can no longer blow up memory — the
        // M49.5 pre-compile complexity gate and its
        // `structured_max_unbounded_subarrays` TOML key are gone.
        let llm: any LLMModule
        switch engine {
        case .stub:
            llm = StubLLMModule(
                modelIds: llmStubIds, configuredDefault: llmDefaultName)
        case .mlx:
            llm = MLXLLMModule(
                modelStoreRoot: store.rootDirectory,
                configuredDefault: llmDefaultName,
                parameters: .init(
                    maxTokens: maxTokens,
                    temperature: Float(temperature ?? 0.7),
                    speculative: speculative,
                    kvCompression: kvCompression,
                    maxPromptTokens: maxPromptTokens,
                    mtpDrafter: tomlCfg?.mtpDrafter,  // ADR 032
                    dataDir: dataDir.map {
                        URL(fileURLWithPath: $0, isDirectory: true)
                    }
                        ?? AthenaEnv.userHome().appendingPathComponent(
                            ".athena", isDirectory: true)),
                promptCacheCapBytes: config.promptCacheCapBytes)
        }
        let embedding: any EmbeddingModule
        switch engine {
        case .stub:
            embedding = StubEmbeddingModule(
                modelIds: embeddingModels,
                configuredDefault: embeddingDefault)
        case .mlx:
            embedding = MLXEmbeddingModule(
                configuredDefault: embeddingDefault,
                modelStoreRoot: store.rootDirectory)
        }
        let transcription: any TranscriptionModule
        switch engine {
        case .stub:
            transcription = StubTranscriptionModule(
                modelIds: transcriptionModels,
                configuredDefault: transcriptionDefault)
        case .mlx:
            transcription = MLXTranscriptionModule(
                configuredDefault: transcriptionDefault,
                modelStoreRoot: store.rootDirectory)
        }
        let diarization: any DiarizationModule
        switch engine {
        case .stub:
            diarization = StubDiarizationModule(
                modelIds: diarizationModels,
                configuredDefault: diarizationDefault)
        case .mlx:
            diarization = MLXDiarizationModule(
                configuredDefault: diarizationDefault,
                modelStoreRoot: store.rootDirectory)
        }
        let speakerEmbedding: any SpeakerEmbeddingModule
        switch engine {
        case .stub:
            speakerEmbedding = StubSpeakerEmbeddingModule(
                modelIds: speakerEmbeddingModels,
                configuredDefault: speakerEmbeddingDefault)
        case .mlx:
            speakerEmbedding = MLXSpeakerEmbeddingModule(
                configuredDefault: speakerEmbeddingDefault,
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

        // Inbound bearer auth (M12). Fail-safe: a non-loopback bind
        // with no keys refuses to start.
        let nTokens = await athenaStore.tokenCount()
        let nUsers = await athenaStore.userCount()
        let dbHasCreds = nTokens > 0 || nUsers > 0
        // Reuse the bootstrap auth loaded above for the store-mode decision
        // (ADR 025 S4) — same file/env keys, now bound to the store + DB rows.
        let authConfig = bootstrapAuth.bound(
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
            store: athenaStore,
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
            maxPromptTokens: maxPromptTokens,
            tokenBudget: tokenBudget,
            tokenBudgetWindow: quotaWindow,
            preload: preload)
        // M54.3 — operator-action pull: at startup, fetch any configured
        // model that has an HF-style id and isn't in the store, in the
        // BACKGROUND. The HTTP surface comes up immediately (server.run
        // below); an inference request for a not-yet-present model gets
        // 503 via the governor's `pulling` flag until the pull lands.
        // Bare store-dir ids (the LLM convention) aren't Hub-pullable and
        // are skipped — they're provisioned via `athena pull`/`convert`.
        // ADR 026 — pull every CONFIGURED id (the flag seed lists), so both
        // engine families land in the store and become selectable; selection
        // itself then reads the store, not this list.
        let configuredForPull: [(ModuleID, [String])] = [
            (.llm, llmModels.isEmpty ? [model].compactMap { $0 } : llmModels),
            (.textEmbedding, embeddingModels),
            (.transcription, transcriptionModels),
            (.diarization, diarizationModels),
            (.speakerEmbedding, speakerEmbeddingModels),
        ]
        let pullStoreRoot = store.rootDirectory
        Task.detached {
            await Self.pullMissingConfigured(
                configuredForPull, storeRoot: pullStoreRoot,
                governor: governor)
        }
        // M60.2 — hold a system-sleep power assertion for the lifetime of the
        // serve loop. Without it an unattended Mac sleeps and SUSPENDS the
        // daemon mid-generation (the confirmed root cause of a downstream client's
        // overnight "freeze"/timeouts — see docs/m60-plan.md). Released on
        // shutdown via the defer. Acquisition failure is non-fatal: we log it
        // and keep serving (/healthz reports the held state).
        let powerAssertion = PowerAssertion(
            reason: "Athena serving inference (prevent idle system sleep)")
        if powerAssertion.acquire() {
            Logger(label: AthenaLogLabel.daemon).notice(
                "power: holding PreventUserIdleSystemSleep (the machine will not idle-sleep while serving)"
            )
        } else {
            Logger(label: AthenaLogLabel.daemon).warning(
                "power: could NOT acquire a sleep assertion — an unattended Mac may suspend inference mid-request. Run under `caffeinate -s` or set `pmset -c sleep 0` as a workaround."
            )
        }
        defer { powerAssertion.release() }
        server.powerAssertionHeld = powerAssertion.isHeld
        // M60.3 — sudoless GPU clock + die-temperature telemetry on /healthz
        // (via the swift-soc-metrics `SoCMetrics` module). Background-sampled,
        // nil-safe: if a signal is unavailable that field is simply omitted.
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
