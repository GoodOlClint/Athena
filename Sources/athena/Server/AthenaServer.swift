import AthenaCore
import AthenaDeploy
import AthenaEmbedding
import AthenaLLM
import AthenaServerKit
import AthenaStore
import AthenaStructured
import AthenaTranscription
import Foundation
import HTTPTypes
import Hummingbird
import MLX
import HummingbirdCore
import HummingbirdTLS
import Logging
import NIOCore
import NIOSSL

/// The single Athena HTTP listener. Passive oracle: it only answers inbound
/// inference queries and exposes governor state — it initiates no
/// connections. M0 surface: `GET /healthz`, `POST /v1/chat/completions`.
struct AthenaServer {
    let config: GovernorConfig
    let governor: MemoryGovernor
    let llm: any LLMModule
    let embedding: any EmbeddingModule
    let transcription: any TranscriptionModule
    let diarization: any DiarizationModule
    let speakerEmbedding: any SpeakerEmbeddingModule
    /// Shared SQLite store (auth/audit/usage). ADR 025 dropped the queue
    /// and vector tenants; ADR 026 retired the allowlist.
    let store: AthenaStore
    /// Served model's display name, echoed in native `/api/*` replies.
    let modelName: String
    /// Model-store root for `/api/models*` (M16.2). `var = default` so
    /// it stays a memberwise-init parameter (see the `auth` note);
    /// Load injects the configured `--model-store` root.
    var modelStoreRoot: URL = ModelStore.defaultRoot
    /// In-process request metrics (M11.1). Defaulted so the
    /// memberwise init is unchanged for existing call sites.
    let metrics = AthenaMetrics()
    /// Inbound bearer auth (M12). Defaulted to disabled (open) so
    /// existing call sites/tests are unchanged; Load injects the
    /// loaded config. `var` (not `let`) so it's a memberwise-init
    /// parameter — set once at construction, never mutated after.
    var auth: AuthConfig = AuthConfig()
    /// In-daemon TLS (M28). PEM cert-chain + private-key paths. Both
    /// set ⇒ serve HTTPS; both nil ⇒ plaintext HTTP; exactly one set ⇒
    /// `run()` throws (fail-closed). `var = nil` so they're
    /// memberwise-init params (same convention as `auth`/`modelStoreRoot`).
    var tlsCertPath: String? = nil
    var tlsKeyPath: String? = nil
    /// True when the daemon itself terminates TLS (both cert+key set —
    /// the same both-or-neither contract `serverBuilder` enforces). Used
    /// to mark the session cookie `Secure` (A12).
    var tlsEnabled: Bool { tlsCertPath != nil && tlsKeyPath != nil }
    /// Inbound rate limiting (M29.1). `rateLimit` = sustained
    /// requests/sec per principal; `rateBurst` = bucket capacity. A
    /// non-positive `rateLimit` disables it (no middleware installed) —
    /// opt-in, off by default. `var = 0` so they're memberwise-init
    /// params (same convention as `auth`/`tlsCertPath`).
    var rateLimit: Double = 0
    var rateBurst: Int = 0
    /// Inbound concurrency caps (M29.2). `maxConcurrency` = max in-flight
    /// requests daemon-wide; `maxConcurrencyPerPrincipal` = max in-flight
    /// per caller. 0 ⇒ unlimited (that dimension off). Both 0 ⇒ no
    /// middleware installed — opt-in, off by default. `var = 0` so
    /// they're memberwise-init params.
    var maxConcurrency: Int = 0
    var maxConcurrencyPerPrincipal: Int = 0
    /// Audit-log retention in days (M30.3). 0 ⇒ keep forever (opt-in).
    /// When > 0, `audit()` opportunistically prunes rows older than the
    /// window so the trail stays bounded as it grows. `var = 0` so it's
    /// a memberwise-init param.
    var auditRetentionDays: Int = 0
    /// Per-request inference timeout in seconds (M33.1). 0 ⇒ no deadline
    /// (opt-in, off by default — a runaway decode is otherwise bounded
    /// only by `max_tokens`). When > 0, each generation is deadline-
    /// wrapped: a sync request that overruns becomes a classified 504
    /// (`inference_timeout`), a streamed one is truncated at the wire.
    /// `var = 0` so it's a memberwise-init param.
    var requestTimeoutSecs: Int = 0
    /// ADR 015 — block-until-ready budget for a request-path cold-load, in
    /// seconds. A request for a non-resident-but-on-disk model waits up to this
    /// long for the local load, then serves; on timeout it falls back to
    /// `503 module_loading` + `Retry-After`. Distinct from `requestTimeoutSecs`
    /// (which bounds generation, after the load). `0` ⇒ legacy immediate-503
    /// (revert switch). Downloads (operator pull) are never waited on.
    /// `var = 120` so it's a memberwise-init param (the block-until-ready
    /// default; peer-runner behavior out of the box).
    var coldLoadWaitSecs: Double = 120
    /// Inbound upload caps (ADR 017). `maxAudioUploadBytes` bounds the raw
    /// multipart body of the three `/v1/audio/*` routes; `maxRequestBodyBytes`
    /// bounds the JSON request bodies (`/v1/chat/completions`, `/v1/embeddings`,
    /// the native control bodies). Over the cap ⇒ a clean `413
    /// payload_too_large` (an up-front `Content-Length` check + a streamed
    /// `collect(upTo:)` backstop). `var = ` defaults (100 MiB / 4 MiB) so
    /// they're memberwise-init params. A `100 Continue` channel handler
    /// (`ExpectContinueHandler`) keeps `Expect:`-using clients from hanging
    /// while a large body streams.
    var maxAudioUploadBytes: Int = 104_857_600
    /// Upload cap for the `/v1/video/*` routes (ADR 022). Video dwarfs audio,
    /// so the default is larger (1 GiB); over it ⇒ 413 payload_too_large, same
    /// machinery as the audio cap. `var = ` so it's a memberwise-init param.
    var maxVideoUploadBytes: Int = 1_073_741_824
    var maxRequestBodyBytes: Int = 4_194_304
    /// Warm the LLM at startup instead of lazily on first request
    /// (M33.3). `var = false` so it's a memberwise-init param. Best-
    /// effort: the warm runs concurrently with serving (the HTTP surface
    /// is up immediately) and a failure falls back to the lazy path.
    var preload: Bool = false
    /// M59.4 — the shared cross-request prompt-prefix KV pool (or nil when
    /// `[prompt_cache]` is disabled). The SAME lock-guarded instance the LLM
    /// module and governor hold; the operator surface (`GET/DELETE
    /// /api/cache/prompt`) reads its stats and flushes it. `var = nil` so it's
    /// a memberwise-init param.
    var prefixCache: PrefixKVCache? = nil
    /// M60.2 — whether the daemon holds a `PreventUserIdleSystemSleep` power
    /// assertion (set by `Load.run()` after acquiring it). Surfaced on
    /// `/healthz` so an operator can confirm the appliance won't idle-sleep
    /// and suspend inference mid-request. `var = false`, set post-init.
    var powerAssertionHeld: Bool = false
    /// M60.3 — sudoless GPU clock probe (or nil when GPU telemetry is
    /// unavailable). Set by `Load.run()`; read by `/healthz`. `var = nil`,
    /// set post-init.
    var gpuProbe: GPUTelemetryProbe? = nil
    /// WebUI session signer (M12.2). Per-process random secret —
    /// sessions invalidate on restart (acceptable for an appliance).
    let session = Session()
    /// Per-IP login throttle (M65.6 / audit A3). `POST /ui/login` is
    /// exempt from the global per-principal rate limiter (it's pre-auth —
    /// there's no principal yet), so brute-forcing the password was
    /// unthrottled. A small token bucket keyed on the TCP peer address
    /// gives a short burst then forces a wait (the bucket's Retry-After
    /// hint), capping sustained guesses without locking a real operator out
    /// for long. burst 5 / 0.2 per sec ⇒ ~5 immediate tries, then 1 every
    /// 5s. Behind a reverse proxy every client shares the proxy's bucket
    /// (ADR 004) — throttle at the proxy for per-client login limits.
    let loginLimiter = RateLimiter(rate: 0.2, burst: 5)

    func run() async throws {
        // ADR 025 S5 (Option D) migration insurance: the upload decode paths no
        // longer stage temp files, but a pre-S5 crash could have orphaned an
        // `athena-*` upload under NSTemporaryDirectory(). Sweep any on boot so no
        // readable request bytes linger. No-op in steady state.
        InMemoryAsset.sweepLegacyUploadTempFiles()
        // M65.6: custom context carries the TCP peer address (for the A3
        // login limiter) and lets AuthMiddleware publish the single
        // resolved caller (A5). Middlewares are generic over Context, so
        // they specialize to AppRequestContext unchanged.
        let router = Router(context: AppRequestContext.self)
        // Auth outermost — reject at the edge, before timing work.
        router.add(
            middleware: AuthMiddleware(
                config: auth, session: session))
        // Rate limiting just inside auth (M29.1): only authenticated
        // callers reach it, keyed by their resolved principal. Opt-in —
        // installed only when a positive rate is configured; auth-off
        // loopback bypasses inside the middleware.
        if rateLimit > 0 {
            let burst = rateBurst > 0
                ? Double(rateBurst) : max(1, rateLimit.rounded(.up))
            router.add(
                middleware: RateLimitMiddleware(
                    limiter: RateLimiter(rate: rateLimit, burst: burst),
                    auth: auth))
        }
        // Concurrency caps just inside the rate limiter (M29.2): admit by
        // in-flight COUNT, holding a slot for the handler's full duration.
        if maxConcurrency > 0 || maxConcurrencyPerPrincipal > 0 {
            router.add(
                middleware: ConcurrencyMiddleware(
                    limiter: ConcurrencyLimiter(
                        global: maxConcurrency,
                        perPrincipal: maxConcurrencyPerPrincipal),
                    auth: auth))
        }
        router.add(middleware: MetricsMiddleware(metrics: metrics))

        router.get("/healthz") { [metrics] _, _ -> Response in
            let snapshot = await governor.snapshot()
            // Join each module slot to the id of its resident model (the
            // governor tracks bytes/state, not model identity).
            var residentModels: [ModuleID: String] = [:]
            for m in snapshot.modules {
                if let r = await self.selectable(m.id).residentModelId() {
                    residentModels[m.id] = r
                }
            }
            let (inflight, lastAt, decodeTps) = await metrics.healthFields()
            let gpu = gpuProbe?.current ?? (mhz: nil, activeResidency: nil)
            let gate = await InferenceGate.shared.stats()
            return Self.json(
                HealthResponse(
                    snapshot: snapshot,
                    residentModels: residentModels,
                    inflight: inflight,
                    lastRequestAt: lastAt,
                    lastDecodeTokensPerSec: decodeTps,
                    powerAssertionHeld: powerAssertionHeld,
                    gate: gate,
                    gpuClockMHz: gpu.mhz,
                    gpuActiveResidency: gpu.activeResidency))
        }

        router.get("/metrics") { request, _ -> Response in
            // Content-negotiated (M37): Prometheus text 0.0.4 by default
            // (the scrape target); JSON only when explicitly asked, so
            // the prior JSON consumers keep working.
            let snap = await metrics.snapshot()
            if (request.headers[.accept] ?? "")
                .contains("application/json")
            {
                return Self.json(snap)
            }
            // ADR 038 slice 1 — append inference-gate queue depth + wait time
            // so FIFO starvation and the batching-trigger contention signal are
            // scrapeable (gate stats live on the actor, not in AthenaMetrics).
            let gate = await InferenceGate.shared.stats()
            return Self.prometheusText(
                AthenaMetrics.prometheus(
                    snap, now: Date().timeIntervalSince1970)
                    + InferenceGate.prometheus(gate))
        }

        // Machine-readable API description (M32.1) — always open, like
        // /healthz, so the appliance can describe itself. Hand-authored
        // OpenAPI 3.0.3 with the running daemon's version stamped in.
        router.get("/openapi.json") { _, _ -> Response in
            Self.jsonString(OpenAPISpec.json(version: Athena.appVersion))
        }

        // WebUI dashboard + model/daemon/RBAC consoles + session login
        // (`AthenaServer+UI.swift`).
        registerUIRoutes(router)

        // OpenAI chat completions (`AthenaServer+ChatOpenAI.swift`).
        registerChatRoutes(router)

        // ADR 036 — Anthropic Messages dialect over the same inference engine
        // (`AthenaServer+Anthropic.swift`).
        registerAnthropicRoutes(router)

        // OpenAI embeddings. The native `/api/embed` alias was removed (ADR
        // 013/031 — `/api/*` is control-plane only, and it had 0 callers).
        router.post("/v1/embeddings") { request, _ -> Response in
            await handleEmbeddings(request)
        }

        // The media surface — `/v1/audio/*` + `/v1/video/*`
        // (`AthenaServer+Audio.swift`).
        registerAudioRoutes(router)

        // ADR 025 S2 — the async request queue (`/v1/queue*`) was removed
        // entirely. Inference is synchronous on `/v1/*`; long-running model
        // ops stream SSE progress on `/api/models/{pull,convert,prune}`.

        // Control plane — OpenAI `/v1/models` discovery + the native `/api/*`
        // surface (model store/lifecycle, RBAC, usage/audit/logs/cache)
        // (`AthenaServer+Admin.swift`).
        registerAdminRoutes(router)

        // M33.3 / M46.2: optionally warm every module that has a
        // configured default model at startup so the first request to
        // that module doesn't pay the cold-load latency.
        //
        // Pre-M46.2 this only warmed `.llm`, which left every other
        // module class (textEmbedding, transcription, diarization,
        // speakerEmbedding) lazy — each consumer ate a 503
        // `module_loading` retry dance on its first call. M46.2 widens
        // the warmup to ANY module that resolves a default model. ADR 026
        // re-points "has a default" from an `is_default=1` allowlist row to
        // the module's `defaultModelId()` (configured TOML default, or the
        // store's sole model of the class). A module with no resolvable
        // default (empty / ambiguous store) stays lazy: warming it would just
        // spam a `moduleLoadFailed`/`ambiguousModel` warning.
        //
        // Best-effort and concurrent — the HTTP surface (below) still
        // comes up immediately; each module's warm runs in its own
        // child task, a failure logs and leaves that module's lazy
        // path intact. `governor.ensureLoaded` is idempotent, so a
        // request racing a warm is safe.
        if preload {
            let governor = self.governor
            Task {
                let log = Logger(label: AthenaLogLabel.daemon)
                // ADR 026 — warm any module with a resolvable default
                // (`defaultModelId()` non-empty), sorted for deterministic
                // log ordering.
                var configured: [ModuleID] = []
                for id in ModuleID.allCases
                where await !selectable(id).defaultModelId().isEmpty {
                    configured.append(id)
                }
                configured.sort { $0.rawValue < $1.rawValue }
                guard !configured.isEmpty else {
                    log.notice(
                        """
                        preload: no module has a configured default; \
                        skipping warm (every module stays lazy)
                        """)
                    return
                }
                let names = configured.map { $0.rawValue }
                    .joined(separator: ",")
                log.notice(
                    """
                    preload: warming \(configured.count) module(s) \
                    at startup: \(names)
                    """)
                // M46.7 — serialize module warms. The original M46.2
                // used `withTaskGroup` to load in parallel; that broke
                // the governor's RSS reconciliation: each module's
                // `before` probe in `performLoad` was captured before
                // ANY of them had grown memory, and the `after` probe
                // saw the cumulative growth from everyone, so each
                // module's reconciled `reservedBytes` over-counted by
                // the sum of every later-completing module. Across 5
                // modules at startup the over-count compounded to
                // peaks near the budget cap (97.9 GB observed on a
                // 103 GB budget) and triggered spurious governor
                // evictions of modules that had genuinely fit. Serial
                // warms cost more wall time at startup, but each
                // `before`/`after` pair sees only its own module's
                // RSS delta — the reconciled `reservedBytes` matches
                // the real process footprint and no spurious eviction
                // fires. Each per-module warm is still best-effort: a
                // failure logs and leaves the lazy path intact for
                // that module.
                for id in configured {
                    do {
                        try await governor.ensureLoaded(id)
                        log.notice("preload: \(id.rawValue) warm")
                    } catch {
                        log.warning(
                            """
                            preload: \(id.rawValue) warm failed, \
                            will load lazily: \(error)
                            """)
                    }
                }
            }
        }

        // `runService` installs SIGTERM/SIGINT graceful shutdown; on signal
        // the HTTP server drains its in-flight requests within the stop
        // window — no abrupt mid-request teardown. (ADR 025 S2 removed the
        // queue worker service that previously joined this group.)
        let app = Application(
            router: router,
            server: try Self.serverBuilder(
                tlsCertPath: tlsCertPath, tlsKeyPath: tlsKeyPath),
            configuration: .init(
                address: .hostname(
                    config.listenHost, port: config.listenPort),
                serverName: "athena"
            )
        )
        try await app.runService()

        // ADR 027 — graceful shutdown completed (in-flight requests drained, so
        // entries are unreferenced): spill idle prompt-cache entries to the disk
        // L2 so a restart resumes them. No-op unless `persist_to_disk` is on.
        // Writes are atomic, so a SIGKILL past the shutdown window leaves no
        // corrupt blob (a missed entry just cold-prefills next time).
        if let prefixCache, prefixCache.persistsToDisk {
            let freed = prefixCache.flushIdle(reason: .shutdown)
            Logger(label: AthenaLogLabel.daemon).notice(
                "prompt-cache: spilled \(freed) idle entries to disk on shutdown (ADR 027)")
        }
    }

    /// Build the HTTP(S) listener. Both cert+key ⇒ TLS; neither ⇒
    /// plaintext HTTP; exactly one ⇒ a hard error (fail-closed — never
    /// silently fall back to plaintext when TLS was half-configured).
    /// PEM load failures (missing/unreadable/malformed) propagate from
    /// NIOSSL and abort daemon start with a clear error.
    static func serverBuilder(
        tlsCertPath: String?, tlsKeyPath: String?
    ) throws -> HTTPServerBuilder {
        // ADR 017 — inject the `Expect: 100-continue` handler into the HTTP1
        // channel pipeline so `Expect:`-using clients (URLSession /
        // AsyncHTTPClient, for large bodies) don't hang waiting for a `100`
        // the daemon otherwise never sends. The autoclosure builds a fresh
        // per-connection handler. One config object covers both transports.
        let http1 = HTTPServerBuilder.http1(
            configuration: .init(
                additionalChannelHandlers: [ExpectContinueHandler()]))
        switch (tlsCertPath, tlsKeyPath) {
        case (nil, nil):
            return http1
        case (let cert?, let key?):
            let chain = try NIOSSLCertificate.fromPEMFile(cert)
                .map { NIOSSLCertificateSource.certificate($0) }
            let pkey = try NIOSSLPrivateKey(file: key, format: .pem)
            let tls = TLSConfiguration.makeServerConfiguration(
                certificateChain: chain,
                privateKey: .privateKey(pkey))
            Logger(label: AthenaLogLabel.daemon).notice(
                "TLS: serving HTTPS (cert \(cert))")
            return try .tls(http1, tlsConfiguration: tls)
        case (.some, nil), (nil, .some):
            throw TLSConfigError.incomplete
        }
    }

    /// ADR 036 S1b — the protocol-agnostic chat orchestration seam. Runs the
    /// model admission + cold-load decision (ADR 015) and, for a resident model,
    /// the model-DEPENDENT checks (vision capability + prompt-cap preflight),
    /// independent of wire dialect. Every chat adapter (OpenAI
    /// `/v1/chat/completions`, Anthropic `/v1/messages`) calls this; only the
    /// decode of `requestedModel`/`messages`/`tools` and the encode of the result
    /// differ. Error outcomes are carried as a neutral `(message,type,code)` +
    /// HTTP status so each adapter renders them in its own envelope.
    ///
    /// NOTE (ADR 036): the caller decodes `messages` BEFORE invoking this, so a
    /// decode fault (e.g. a bad image part → 400) now precedes admission for a
    /// request that trips both — a deliberate, documented precedence change vs
    /// the pre-seam handler (rejects a malformed request before touching the
    /// model). Single-fault behaviour is unchanged.
    enum ChatPrep {
        /// A terminal PRE-commitment fault (before any SSE 200): the carried
        /// Response is final. For OpenAI this is the canonical `{"error":…}`
        /// envelope verbatim (so OpenAI stays byte-identical); a future dialect
        /// may translate it — admission/server faults using Athena's canonical
        /// envelope is an accepted honesty boundary (ADR 036), the success and
        /// in-stream paths are dialect-shaped.
        case failed(Response)
        /// Warm: the model is resident and the model-dependent checks passed
        /// inline. `model` is the served id for the response envelope.
        case ready(model: String)
        /// ADR 015 cold-load streaming: the model is not resident; the SSE
        /// producer must await the load (`load`) then run the model-dependent
        /// checks (`prepareAfterLoad`). Both closures are dialect-neutral
        /// `(message,type,code)` so each adapter's streaming consumer renders
        /// the in-stream error in its own shape.
        case deferToStream(
            load: @Sendable () async -> ColdStreamLoad,
            prepareAfterLoad:
                @Sendable () async -> (
                    message: String, type: String, code: String
                )?)
    }

    /// Run the orchestration seam (see `ChatPrep`). `requestedModel` is the
    /// dialect's model field; `messages`/`tools`/`chatTemplateKwargs` are the
    /// already-decoded native inputs the vision/preflight checks need.
    func prepareChat(
        request: Request, requestedModel: String?,
        messages: [ChatTurn], tools: [[String: any Sendable]]?,
        chatTemplateKwargs: [String: any Sendable]?, wantStream: Bool
    ) async -> ChatPrep {
        var deferLoadIntoStream = false
        if wantStream {
            // Validate the requested id up front, then decide: wait inline
            // (warm) vs stream the wait (cold).
            do {
                try await llm.selectColdLoadModel(requestedModel)
            } catch let e as AthenaError {
                return .failed(
                    Self.error(
                        status: HTTPResponse.Status(code: e.httpStatus),
                        message: e.message, type: "server_error", code: e.code))
            } catch {
                return .failed(Self.classified(error, module: .llm))
            }
            switch await governor.peekLoad(.llm) {
            case .loaded: break
            case .pulling: return .failed(Self.coldLoadResponse(.llm))
            case .needsLoad: deferLoadIntoStream = true
            }
        }
        if !deferLoadIntoStream {
            if let err = await governedLLM(
                request: request, requestedModel: requestedModel)
            {
                return .failed(err)
            }
        }
        let hasImages = messages.contains { !$0.images.isEmpty }
        if deferLoadIntoStream {
            // ADR 015 — the model-dependent checks move into the SSE producer.
            return .deferToStream(
                load: { [governor, coldLoadWaitSecs] in
                    do {
                        switch try await governor.awaitLoad(
                            .llm, within: coldLoadWaitSecs)
                        {
                        case .loaded: return .ready
                        case .loading: return .timedOut
                        }
                    } catch let e as AthenaError {
                        return .failed(
                            message: e.message, type: "server_error",
                            code: e.code)
                    } catch {
                        let c = AthenaError.classify(error, module: .llm)
                        return .failed(
                            message: c.message, type: "server_error",
                            code: c.code)
                    }
                },
                prepareAfterLoad: { [llm] in
                    if hasImages, await llm.servesVision == false {
                        return (
                            "image input is not supported by the requested "
                                + "model", "invalid_request_error",
                            "vision_not_supported")
                    }
                    do {
                        try await llm.preflightPromptCache(
                            messages: messages, tools: tools,
                            chatTemplateKwargs: chatTemplateKwargs)
                        return nil
                    } catch let e as AthenaError {
                        return (e.message, "server_error", e.code)
                    } catch {
                        let c = AthenaError.classify(error, module: .llm)
                        return (c.message, "server_error", c.code)
                    }
                })
        }
        // Warm/blocking: run the model-dependent checks inline.
        if hasImages, await llm.servesVision == false {
            return .failed(
                Self.error(
                    status: .badRequest,
                    message:
                        "image input is not supported by the requested model",
                    type: "invalid_request_error", code: "vision_not_supported"))
        }
        do {
            try await llm.preflightPromptCache(
                messages: messages, tools: tools,
                chatTemplateKwargs: chatTemplateKwargs)
        } catch let e as AthenaError {
            return .failed(
                Self.error(
                    status: HTTPResponse.Status(code: e.httpStatus),
                    message: e.message, type: "server_error", code: e.code))
        } catch {
            return .failed(Self.classified(error, module: .llm))
        }
        return .ready(model: await servedLLMModel())
    }

    /// ADR 036 — render an Anthropic-shaped error
    /// (`{"type":"error","error":{type,message}}`, inner type from the status).
    static func anthropicError(
        status: HTTPResponse.Status, message: String
    ) -> Response {
        Self.json(
            AnthropicErrorBody(
                error: .init(
                    type: AnthropicErrorBody.errorType(forStatus: status.code),
                    message: message)),
            status: status)
    }

    private func handleEmbeddings(_ request: Request) async -> Response {
        let t0 = Date()
        let body: EmbeddingRequest
        do {
            let buffer = try await request.body.collect(upTo: maxRequestBodyBytes)
            body = try JSONDecoder().decode(
                EmbeddingRequest.self, from: Data(buffer: buffer))
        } catch {
            return Self.error(
                status: .badRequest,
                message: "Invalid request body: \(error)",
                type: "invalid_request_error",
                code: "invalid_body")
        }
        guard !body.input.isEmpty else {
            return Self.error(
                status: .badRequest,
                message: "'input' must be a non-empty string or array",
                type: "invalid_request_error", code: "invalid_input")
        }

        // M39: `body.model` selects among the configured set. governedEmbed
        // gates the cold-load, audits a real per-request rebind (M41.4), and
        // reports the id actually served, which we echo back.
        let batch: EmbeddingBatch
        switch await governedEmbed(
            request, body.input, module: .textEmbedding, model: body.model)
        {
        case .fail(let r): return r
        case .ok(let b): batch = b
        }
        // M27.1/.2: embeddings have no completion — prompt == total.
        await meter(
            principal: usagePrincipal(request),
            usage: TokenUsage(
                promptTokens: batch.promptTokens, completionTokens: 0))

        let response = EmbeddingResponse(
            object: "list",
            data: batch.vectors.enumerated().map {
                EmbeddingObject(
                    object: "embedding", embedding: $0.element,
                    index: $0.offset)
            },
            model: batch.model,
            usage: Usage(
                prompt_tokens: batch.promptTokens, completion_tokens: 0,
                total_tokens: batch.promptTokens))
        // M56 — per-request summary (req/principal tags auto-attach via the
        // M45 LogScope) so an operator can see embedding traffic, latency,
        // and the model actually served — the non-LLM paths logged nothing.
        Logger(label: AthenaLogLabel.model(.textEmbedding)).notice(
            """
            embeddings done model=\(batch.model) \
            inputs=\(body.input.count) vectors=\(batch.vectors.count) \
            prompt_tokens=\(batch.promptTokens) \
            elapsed_ms=\(Self.elapsedMs(t0))
            """)
        return Self.json(response)
    }

    // MARK: - Principal resolution & metering (M12.6 / M27)

    /// Resolve the caller's principal + admin/enforced flags from the
    /// AuthMiddleware-published resolution. `enforced` is auth being on;
    /// `principal` identifies the bearer's owning subject (`u:<user>` for a
    /// managed token, `t:<hash>` for a bootstrap key). `isAdmin` = the
    /// caller holds the full permission set (the `admin` role). Used for
    /// per-principal usage metering + the `/api/usage` admin/own scoping.
    func bearerPrincipal(_ request: Request) async -> (
        principal: String?, isAdmin: Bool, enforced: Bool
    ) {
        // M65.6 (A5): read the single resolution AuthMiddleware published
        // (the bearer surface this gates always passes through the bearer
        // branch, which bound it) instead of resolving the token again.
        guard auth.isEnabled else { return (nil, false, false) }
        guard let caller = ResolvedCaller.current else {
            return (nil, false, true)
        }
        let isAdmin = Set(Permission.allCases).isSubset(
            of: caller.permissions)
        return (caller.principal, isAdmin, true)
    }

    /// Metering principal for the unauthenticated loopback caller (auth
    /// disabled). `xenos` — Greek ξένος, "guest/stranger" — the guest
    /// who arrives without credentials; distinct from the `u:`/`t:`
    /// prefixes used for authenticated subjects (M27.2).
    static let xenos = "xenos"

    /// The principal a request should be metered against:
    /// the authenticated subject, or nil when auth is off (mapped to
    /// `xenos` by `meter`). Inference handlers only run after the auth
    /// middleware admits the request, so an enabled-auth request always
    /// resolves a real principal here.
    func usagePrincipal(_ request: Request) async -> String? {
        await bearerPrincipal(request).principal
    }

    /// Record one request's token usage (M27): bump the global metrics
    /// counter and the persisted per-principal counter. nil principal ⇒
    /// auth off ⇒ the `xenos` sentinel. Persistence failures are
    /// non-fatal — metering must never break inference.
    func meter(principal: String?, usage: TokenUsage) async {
        await metrics.addTokens(usage.totalTokens)
        try? await store.addUsage(
            principal: principal ?? Self.xenos,
            promptTokens: usage.promptTokens,
            completionTokens: usage.completionTokens)
    }

    // MARK: - Native /api inference (M16)

    /// `Response` isn't `Error`, so a plain success-or-error-response.
    enum Outcome<T> {
        case ok(T)
        case fail(Response)

        /// The carried error `Response` — for the decode sites'
        /// `guard case .ok(let body) = decoded else { return decoded.orFail }`.
        /// Total (the enum is binary); the `.ok` arm is unreachable by
        /// construction (callers read this only in the `else` of a `case .ok`
        /// match) and returns a safe 500 rather than trapping, so a future
        /// misuse degrades instead of aborting the daemon. Replaces the 9×
        /// `if case .fail(let r) = decoded { return r }; fatalError()` boilerplate.
        var orFail: Response {
            switch self {
            case .fail(let r): return r
            case .ok:
                return AthenaServer.error(
                    status: .internalServerError,
                    message: "internal error", type: "server_error",
                    code: "internal_error")
            }
        }
    }

    func decodeJSON<T: Decodable>(
        _ request: Request, _ type: T.Type
    ) async -> Outcome<T> {
        do {
            let buf = try await request.body.collect(upTo: maxRequestBodyBytes)
            return .ok(
                try JSONDecoder().decode(T.self, from: Data(buffer: buf)))
        } catch {
            return .fail(
                Self.error(
                    status: .badRequest,
                    message: "Invalid request body: \(error)",
                    type: "invalid_request_error", code: "invalid_body"))
        }
    }

    /// Governed embedding path for `/v1/embeddings`: cold-load gate →
    /// per-request model rebind (audited on a real resident-id change,
    /// M41.4) → embed. Returns the whole batch so the caller echoes the
    /// model ACTUALLY served (M39). `model` selects among the configured
    /// set (nil ⇒ default); an unknown id surfaces as a classified 400
    /// `model_not_available`.
    private func governedEmbed(
        _ request: Request, _ inputs: [String],
        module: ModuleID, model: String? = nil
    ) async -> Outcome<EmbeddingBatch> {
        do {
            // M43.2: never block the request thread on a multi-GB cold-load.
            // `.loading` ⇒ 503+Retry-After so the client paces its retries.
            switch try await governor.awaitLoad(.textEmbedding, within: coldLoadWaitSecs) {
            case .loaded: break
            case .loading:
                return .fail(Self.coldLoadResponse(.textEmbedding))
            }
            // M41.4: a real resident-id change from per-request `model` is
            // audited (the embedder also self-rebinds inside embed(), which
            // becomes a no-op once the slot already matches).
            if let m = model, !m.isEmpty {
                try await auditedRebind(
                    request, module: .textEmbedding, target: m)
            }
        } catch {
            // issue #6: classify (correct 4xx + type, OOM→503, no leak)
            // instead of a catch-all 500.
            return .fail(Self.classified(error, module: module))
        }
        do {
            return .ok(try await embedding.embed(inputs, model: model))
        } catch {
            return .fail(Self.classified(error, module: module))
        }
    }

    // MARK: - Model store (M16.2)

    /// WP9 — one shared formatter instead of allocating an
    /// `ISO8601DateFormatter` per call on request paths (they're expensive to
    /// construct). `nonisolated(unsafe)`: `Foundation` date formatters are
    /// documented thread-safe for read-only formatting (`.string(from:)`), which
    /// is the only use here, so the shared instance is safe across request tasks.
    nonisolated(unsafe) static let isoFormatter = ISO8601DateFormatter()

    // MARK: - Per-module model lifecycle (M41.1)

    /// M41.2 — governed LLM gate: ensure the slot is loaded, then if
    /// `requestedModel` is non-empty and differs from the resident,
    /// rebind in place (validated against the allowlist; an unknown id
    /// becomes a classified 400 `model_not_available`). Returns an
    /// error `Response` to short-circuit, or nil when the request may
    /// proceed. M41.4: an actual rebind (resident-id changed) emits an
    /// `model.rebind` audit record (M30 trail), so a per-request slot
    /// swap is attributable to the caller.
    private func governedLLM(
        request: Request? = nil, requestedModel: String?
    ) async -> Response? {
        do {
            // M62 — bind the requested model on the (possibly cold) load,
            // not the default. Setting the cold-load target BEFORE
            // awaitLoad means the background load binds the requested
            // model directly; without this a cold/just-restarted slot loaded
            // the DEFAULT and 503'd before the rebind ran, so a request for a
            // non-default model silently got the default (a downstream client's
            // 4bit→8bit). Validated here so an unknown id is a 400 before a
            // doomed multi-GB load starts; nil/empty ⇒ the default.
            try await llm.selectColdLoadModel(requestedModel)
            // ADR 015: block-until-ready — wait up to `coldLoadWaitSecs` for an
            // on-disk cold-load, then serve (peer-runner behavior); only a
            // timeout or an in-flight download (pull) still 503s. Covers
            // /v1/chat/completions and /v1/messages. (The streaming path layers
            // SSE keep-alives over this wait — see handleChatCompletions.)
            switch try await governor.awaitLoad(.llm, within: coldLoadWaitSecs) {
            case .loaded: break
            case .loading: return Self.coldLoadResponse(.llm)
            }
            // Warm slot already holding a different model ⇒ swap in place
            // (the cold-load target above only applies while unloaded).
            if let m = requestedModel, !m.isEmpty {
                try await auditedRebind(
                    request, module: .llm, target: m)
            }
        } catch let e as AthenaError {
            return Self.error(
                status: HTTPResponse.Status(code: e.httpStatus),
                message: e.message, type: "server_error", code: e.code)
        } catch {
            return Self.classified(error, module: .llm)
        }
        return nil
    }

    /// M41.4 — rebind `module`'s slot to `target` and audit an actual
    /// resident-id change (no-op rebinds are not audited; that would
    /// drown the trail on every request). `request` is optional so
    /// non-HTTP-driven callers can skip the audit.
    func auditedRebind(
        _ request: Request?, module: ModuleID, target: String
    ) async throws {
        let sel = selectable(module)
        let before = await sel.residentModelId()
        // ADR 029 — a warm swap loads the new model's weights; gate it so it
        // can't run while a decode holds the slot (which retains the OLD
        // container via ARC → transient double-residency → OOM) or while
        // another tenant executes. Cold-load (`performLoad`) stays UNgated —
        // that is the governor's load wait, not Metal execution.
        try await InferenceGate.shared.withExclusiveExecution {
            try await sel.rebind(to: target)
        }
        let after = await sel.residentModelId()
        guard before != after, let request else { return }
        await audit(
            request, action: "model.rebind",
            target: "\(module.rawValue):\(target)", result: "ok",
            detail: "from=\(before ?? "-") to=\(after ?? "-") "
                + "trigger=inference")
    }

    /// The id actually resident in the LLM slot; falls back to the
    /// module's default when the slot is somehow still empty. Used for
    /// truthful `model` echo on responses (M41.2 — same discipline as
    /// the M39 embedding batch).
    func servedLLMModel() async -> String {
        let sel = selectable(.llm)
        if let r = await sel.residentModelId() { return r }
        return await sel.defaultModelId()
    }


    /// The `any ModelSelectable` corresponding to `id`. Every concrete
    /// module conforms (the stubs too), so this is total — no `nil`.
    func selectable(_ id: ModuleID) -> any ModelSelectable {
        switch id {
        case .llm: return llm as! any ModelSelectable
        case .textEmbedding:
            return embedding as! any ModelSelectable
        case .transcription:
            return transcription as! any ModelSelectable
        case .diarization:
            return diarization as! any ModelSelectable
        case .speakerEmbedding:
            return speakerEmbedding as! any ModelSelectable
        }
    }

    /// `GET /api/models/resident` (M41.1) — every slot's allowlist +
    /// default + currently-resident id. Read-only model-store
    /// projection: gated `model.read` by AuthPolicy.
    /// Live resident model id per module class (module rawValue → loaded
    /// id), omitting unloaded slots. Shared by `/api/models/resident` and
    /// the `/ui` dashboard so both report the SAME resident truth.
    func residentModelMap() async -> [String: String] {
        var out: [String: String] = [:]
        for id in ModuleID.allCases {
            if let m = await selectable(id).residentModelId() {
                out[id.rawValue] = m
            }
        }
        return out
    }

    // MARK: - RBAC admin (M16.4)

    /// The CALLER's effective permission set. Auth-off loopback is a
    /// single trusted operator (mirrors the offline CLI's implicit-
    /// admin grantor). With auth on: a Bearer resolves to its
    /// subject; ELSE (M18.4) a valid WebUI session cookie resolves
    /// to the logged-in user's role-perms, so the SAME M16.4 RBAC
    /// handlers (canGrant / last-admin / cross-rank guards) enforce
    /// against the cookie user when reused by /ui — never the page.
    /// Bearer `/api/*` requests carry no cookie ⇒ unchanged.
    /// Anything unresolved ⇒ empty set ⇒ every escalation check
    /// fails closed (AuthMiddleware already gated the route too).
    func callerPermissions(_ request: Request) async
        -> Set<Permission>
    {
        // M65.6 (A5): read the single resolution AuthMiddleware published
        // for this request rather than re-deriving it from the headers
        // (which is what drifted from the gate). Auth-off keeps the
        // single-trusted-operator full-perms behavior.
        guard auth.isEnabled else { return Set(Permission.allCases) }
        return ResolvedCaller.current?.permissions ?? []
    }

    /// The acting principal for an audit record, resolved for BOTH
    /// surfaces — a Bearer subject (`/api/*`), a WebUI session cookie
    /// (`/ui/*` → `u:<user>`), or the auth-off loopback operator
    /// (`xenos`). Mirrors `callerPermissions` so neither path is
    /// missed.
    func auditPrincipal(_ request: Request) async -> String {
        // M65.6 (A5): the published resolution carries the principal for
        // BOTH surfaces (bearer `u:`/`t:`, cookie `u:<user>`); no re-derive.
        guard auth.isEnabled else { return Self.xenos }
        return ResolvedCaller.current?.principal ?? "unknown"
    }

    /// Record one admin/security mutation to the M30 audit trail. The
    /// dual sink chosen for M30: an append-only SQLite row AND a
    /// `.notice` unified-log line (category `audit`, riding M10 + the
    /// opt-in remote syslog). Called from inside the shared `handle*`
    /// chokepoints so the Bearer `/api/*` and cookie `/ui/*` callers
    /// are both captured. A failed write is swallowed — an audit
    /// hiccup must never sink a mutation that already happened.
    /// Outcomes recorded: `ok` (applied) and `denied` (an
    /// authorization guard refused it); plain input-validation 400s
    /// and not-found 404s changed nothing and are not audited.
    func audit(
        _ request: Request, action: String, target: String?,
        result: String, detail: String? = nil
    ) async {
        let principal = await auditPrincipal(request)
        do {
            try await store.addAudit(
                principal: principal, action: action,
                target: target, result: result, detail: detail)
        } catch {
            // H14 (M66.1): a dropped audit write leaves a gap in the
            // security trail — log it at ERROR (not warning) and bump the
            // /metrics counter so the gap is observable, not silent. The
            // mutation that triggered the audit already happened; we never
            // fail it on an audit hiccup.
            Self.auditLog.error(
                "audit write FAILED action=\(action): \(error)")
            await metrics.recordAuditWriteFailure()
        }
        // Opportunistic age-based retention (M30.3): bound the trail as
        // it grows. 0 ⇒ keep forever. Non-fatal.
        if auditRetentionDays > 0 {
            let cutoff = Date().timeIntervalSince1970
                - Double(auditRetentionDays) * 86_400
            let removed =
                (try? await store.pruneAudit(olderThan: cutoff)) ?? 0
            if removed > 0 {
                Self.auditLog.notice(
                    """
                    audit retention pruned \(removed) row(s) older \
                    than \(auditRetentionDays)d
                    """)
            }
        }
        Self.auditLog.notice(
            """
            audit principal=\(principal) action=\(action) \
            target=\(target ?? "-") result=\(result)\
            \(detail.map { " detail=\($0)" } ?? "")
            """)
    }

    // MARK: - Response helpers

    static func json<T: Encodable>(
        _ value: T, status: HTTPResponse.Status = .ok
    ) -> Response {
        let data =
            (try? JSONEncoder().encode(value)) ?? Data("{}".utf8)
        var buffer = ByteBuffer()
        buffer.writeBytes(data)
        var headers = HTTPFields()
        headers[.contentType] = "application/json"
        return Response(
            status: status, headers: headers,
            body: ResponseBody(byteBuffer: buffer))
    }

    /// Serve a pre-rendered JSON string verbatim (the embedded OpenAPI
    /// document, M32.1) — no re-encode of an already-formed document.
    static func jsonString(
        _ s: String, status: HTTPResponse.Status = .ok
    ) -> Response {
        var buffer = ByteBuffer()
        buffer.writeBytes(Data(s.utf8))
        var headers = HTTPFields()
        headers[.contentType] = "application/json"
        return Response(
            status: status, headers: headers,
            body: ResponseBody(byteBuffer: buffer))
    }

    /// Prometheus text-exposition response (M37) — the content type a
    /// scraper expects for format version 0.0.4.
    static func prometheusText(_ s: String) -> Response {
        var buffer = ByteBuffer()
        buffer.writeBytes(Data(s.utf8))
        var headers = HTTPFields()
        headers[.contentType] =
            "text/plain; version=0.0.4; charset=utf-8"
        return Response(
            status: .ok, headers: headers,
            body: ResponseBody(byteBuffer: buffer))
    }

    static func error(
        status: HTTPResponse.Status, message: String, type: String,
        code: String, hint: String? = nil
    ) -> Response {
        // Operator legibility (M43.4 follow-up): every 5xx the daemon
        // sends must leave a server-side trace. Classified errors that
        // skip ``classified(_:module:)`` were previously silent — a
        // ``requestTimedOut`` 504 / cold-load 503 / OOM 503 reached the
        // wire with no log line, so an operator reading `log show` saw
        // the client hang up but no daemon explanation. 4xx is left
        // silent — it's a client-shape problem, surfaced by the audit
        // log (M30) when it matters and noisy otherwise.
        if status.code >= 500 {
            log.warning(
                """
                governed request failed status=\(status.code) \
                code=\(code) \(message)
                """)
        }
        return json(
            APIErrorBody(
                error: .init(
                    message: message, type: type, code: code,
                    hint: hint)),
            status: status)
    }

    /// ADR 017 — the `413 payload_too_large` envelope (clean message,
    /// stating the cap; no leaked internal type names).
    static func tooLargeResponse(cap: Int) -> Response {
        error(
            status: .contentTooLarge,
            message: UploadLimit.tooLargeMessage(cap: cap),
            type: "invalid_request_error", code: "payload_too_large")
    }

    /// ADR 017 — up-front 413 when a declared `Content-Length` already
    /// exceeds `cap`, so an oversized upload fails fast without reading
    /// the body (and, for an `Expect: 100-continue` client, a final status
    /// the client accepts in lieu of `100`). nil ⇒ proceed; the streamed
    /// `collect(upTo: cap)` backstop still enforces the cap when the
    /// header is absent, chunked, or understated.
    static func payloadTooLarge(_ request: Request, cap: Int)
        -> Response?
    {
        let declared = request.headers[.contentLength].flatMap { Int($0) }
        switch UploadLimit.check(contentLength: declared, cap: cap) {
        case .proceed: return nil
        case .rejectTooLarge: return tooLargeResponse(cap: cap)
        }
    }

    /// M71.1 — build chat turns from OpenAI messages, decoding any `image_url`
    /// content-parts into `ChatImage` bytes. A message with neither text nor
    /// images carries nothing (dropped). Throws `ChatImageError` for an
    /// unacceptable image URL (remote/unknown-scheme/malformed) so the caller
    /// can map it to a 400 — the passive-oracle reject of `http(s)` lives in
    /// `ChatImage.fromImageURL`.
    static func chatTurns(from messages: [ChatMessage]) throws -> [ChatTurn] {
        try messages.compactMap { m -> ChatTurn? in
            let images = try m.imageURLs.map { try ChatImage.fromImageURL($0) }
            // ADR 034 — carry assistant tool_calls + tool-result tool_call_id
            // so the template renders a coherent call→result history.
            let toolCalls = (m.tool_calls ?? []).map {
                ChatToolCall(
                    id: $0.id, name: $0.function.name,
                    argumentsJSON: $0.function.arguments)
            }
            guard let c = m.content else {
                // An assistant tool-call turn has null content + tool_calls —
                // keep it (don't drop), so the model sees its own prior call.
                return images.isEmpty && toolCalls.isEmpty
                    ? nil
                    : ChatTurn(
                        role: m.role, content: "", images: images,
                        toolCalls: toolCalls, toolCallID: m.tool_call_id)
            }
            return ChatTurn(
                role: m.role, content: c, images: images,
                toolCalls: toolCalls, toolCallID: m.tool_call_id)
        }
    }

    /// Human-readable 400 message for an image content-part failure (M71.1).
    static func imageErrorMessage(_ error: Error) -> String {
        switch error {
        case ChatImageError.remoteURLUnsupported:
            return
                "remote image URLs (http/https) are not supported; inline the "
                + "image as a base64 data: URL. Athena performs no outbound "
                + "image fetch (passive oracle)."
        case ChatImageError.unsupportedScheme:
            return
                "unsupported image URL scheme; only inline data: URLs are "
                + "accepted"
        case ChatImageError.malformedDataURL:
            return "malformed image data: URL"
        default:
            return "invalid image content part"
        }
    }

    /// 400 response for an image content-part failure (M71.1).
    static func imageContentError(_ error: Error) -> Response {
        Self.error(
            status: .badRequest, message: Self.imageErrorMessage(error),
            type: "invalid_request_error", code: "invalid_image")
    }

    /// M43.2 — response for a request that hit a still-cold module. The
    /// load runs detached on the background path; the caller gets a
    /// `503` with a fixed `Retry-After: 5` so the next attempt is paced
    /// instead of hammering the daemon while the multi-GB download
    /// finishes. 5 s is the brief's locked default; long enough to not
    /// thrash, short enough to feel responsive.
    static func coldLoadResponse(_ id: ModuleID) -> Response {
        // Route through the canonical `error(...)` envelope helper (one error
        // shape, SSOT) instead of hand-building the JSON — same wire body, plus
        // the server-side warn-log side effect — then attach the paced retry.
        var response = error(
            status: .serviceUnavailable,
            message: "module \(id.rawValue) is loading; retry shortly",
            type: "server_error", code: "module_loading")
        response.headers[.retryAfter] = "5"
        return response
    }

    /// Classify an arbitrary inference error: a genuine MLX/Metal OOM
    /// becomes a governed 503 (`metal_oom`), never a bare 500 / process
    /// abort (brief item 4a). Existing `AthenaError`s pass through.
    static let log = Logger(label: AthenaLog.daemonLabel)
    private static let auditLog = Logger(label: AthenaLogLabel.audit)

    /// WP5 (audit P3) — an RBAC-admin store operation (putUser / grantRole /
    /// putToken) failed. Log the raw detail (SQLite message / constraint) to
    /// os_log but return only a stable, detail-free message to the client,
    /// mirroring the `classified` suppression boundary the inference paths use.
    /// The four admin sites previously returned `"\(error)"` verbatim.
    static func storeError(_ err: any Error) -> Response {
        log.warning("admin store operation failed: \(err)")
        return error(
            status: .internalServerError,
            message: "internal store error",
            type: "server_error", code: "store_error")
    }

    static func classified(
        _ err: any Error, module: ModuleID
    ) -> Response {
        let e = AthenaError.classify(err, module: module)
        // NE7: log the FULL detail (substrate paths/ids/state) to os_log,
        // but return only the stable, detail-free `e.message` to the client.
        log.warning(
            """
            governed request failed module=\(module) \
            status=\(e.httpStatus) code=\(e.code) \(e.serverDetail ?? e.message)
            """)
        return error(
            status: HTTPResponse.Status(code: e.httpStatus),
            message: e.message, type: e.type, code: e.code)
    }
}

/// M43.1 — /healthz response, flattening `GovernorSnapshot` for
/// consumers that read top-level `residentBytes` etc., plus three
/// live signals so a hung daemon is legible without scraping
/// /metrics: `inflight` (active request count) and `lastRequestAt`
/// (epoch seconds; 0 ⇒ none since boot). (The `queueDepth` field was
/// removed with the async queue — ADR 025 S2.)
///
/// M46.5 renamed the bytes field from `reservedBytes` to
/// `residentBytes` (the governor's reconciled value tracks real
/// process RSS post-M46.7, so the honest name applies); the
/// per-module entries now also carry `unloadedReason` (nil while
/// loaded, otherwise idle_evict / memory_pressure / operator_unload
/// / load_failed) so an operator can tell why a slot is empty
/// without reaching for `athena audit`.
struct HealthResponse: Encodable {
    let totalBudgetBytes: Int
    let residentBytes: Int
    let freeBytes: Int
    let promptCacheCapBytes: Int
    /// M55 — live process `phys_footprint` (the Activity Monitor "Memory"
    /// number): counts the Metal/GPU KV/prompt-cache/activation buffers
    /// that the governor's per-module `residentBytes` (a reconciled
    /// reservation against mmap'd weights) and `residentBytes` (RSS) do
    /// not. Surfaced so an operator can see the true footprint — and the
    /// GPU-transient gap above it — without scraping Activity Monitor.
    let physFootprintBytes: Int
    /// M59.2 — cross-request prompt-prefix KV reuse pool (0 when disabled).
    let promptCachePoolBytes: Int
    let promptCachePoolEntries: Int
    /// M60.1 — macOS thermal-pressure level from
    /// `ProcessInfo.thermalState`: `nominal` / `fair` / `serious` /
    /// `critical`. No elevated privilege required (unlike `powermetrics`).
    /// This is the throttle indicator a client should poll and back off on:
    /// at `serious`/`critical` the GPU clock is reduced and the
    /// compute-bound prefill slows first, so a large call that completes
    /// at `nominal` can cross its deadline. 0-cost to read.
    let thermalState: String
    /// M60.1 — most recent decode throughput (tok/s) observed by the
    /// heartbeat; holds the last value while idle (0 ⇒ none since boot).
    /// Read with `thermalState` to decide whether a planned output length
    /// fits the client timeout at the current rate.
    let lastDecodeTokensPerSec: Double
    /// M60.1 — MLX allocator counters (the heartbeat's `mlx_active` /
    /// `mlx_cache`), surfaced here so an operator can watch the recyclable
    /// buffer pool across a sustained run without scraping the log. A
    /// monotonically climbing `mlxCacheBytes` under steady load is the
    /// tell for buffer-pool growth.
    let mlxActiveBytes: Int
    let mlxCacheBytes: Int
    /// ADR 023 G1 — the live `MLX.Memory.cacheLimit` (the serve-path cache
    /// bound). Lets an operator confirm the cache is bounded vs MLX's default;
    /// `mlxCacheBytes` should plateau at/under this.
    let mlxCacheLimitBytes: Int
    /// ADR 023 G2 — the active admission accounting mode (`footprint` =
    /// truthful, meters `budget − max(committed, reserved)`; `estimate` = the
    /// revert switch). `freeBytes` above is computed under this mode.
    let admissionMode: String
    /// M60.2 — whether the daemon holds a `PreventUserIdleSystemSleep` power
    /// assertion. `false` ⇒ an unattended Mac can idle-sleep and SUSPEND
    /// inference mid-request (the root cause of the M60 throughput-decay
    /// investigation). An operator/monitor should alert if this is ever false
    /// on a serving appliance.
    let powerAssertionHeld: Bool
    /// M60.3 — current GPU clock in MHz from in-process IOReport telemetry
    /// (sudoless). `null` when GPU telemetry is unavailable on this host. Read
    /// with `thermalState`: a clock well below the boost ceiling under load is
    /// the throttle, in absolute terms.
    let gpuClockMHz: Double?
    /// M60.3 — fraction of the recent window the GPU was active (`0...1`), or
    /// `null` when unavailable.
    let gpuActiveResidency: Double?
    let modules: [ModuleHealth]
    let inflight: Int
    let lastRequestAt: Double
    /// ADR 038 slice 1 — inference-execution-gate observability (was test-only).
    /// `gateWaiters` is the live queue depth behind the ADR-029 gate;
    /// `gateHeldMs` is how long the current holder has held it (0 when free);
    /// `gateMaxWaiters` is the high-water depth since boot; `gateWaitP95Ms` is
    /// the recent queue wait — routine `gateWaiters ≥ 2` / rising `gateWaitP95Ms`
    /// is the evidence trigger for in-span batching (Decision 2).
    let gateWaiters: Int
    let gateHeld: Bool
    let gateHeldMs: Double
    let gateMaxWaiters: Int
    let gateWaitP95Ms: Double

    /// A governor module snapshot enriched with the id of the model currently
    /// resident in that module's slot (`nil` when unloaded) — so `athena ps`
    /// and any /healthz scraper can see *which* model is loaded per module, not
    /// just that the slot is occupied. The governor itself does not track model
    /// ids (that is a `ModelSelectable` concern), so it is joined in here.
    struct ModuleHealth: Encodable {
        let id: ModuleID
        let state: ModuleState
        let residentBytes: Int
        let evictable: Bool
        let unloadedReason: UnloadedReason?
        let model: String?
        /// ADR 023 G3 — true ⇒ `residentBytes` is a measured load-time
        /// footprint; false ⇒ the pre-load estimate (not yet reconciled).
        let measured: Bool
    }

    init(
        snapshot: GovernorSnapshot, residentModels: [ModuleID: String],
        inflight: Int,
        lastRequestAt: Double,
        lastDecodeTokensPerSec: Double, powerAssertionHeld: Bool,
        gate: InferenceGate.GateStats,
        gpuClockMHz: Double? = nil, gpuActiveResidency: Double? = nil
    ) {
        self.totalBudgetBytes = snapshot.totalBudgetBytes
        self.residentBytes = snapshot.residentBytes
        self.freeBytes = snapshot.freeBytes
        self.promptCacheCapBytes = snapshot.promptCacheCapBytes
        self.physFootprintBytes = ProcessMemory.sample().physFootprint
        self.promptCachePoolBytes = snapshot.promptCachePoolBytes
        self.promptCachePoolEntries = snapshot.promptCachePoolEntries
        self.thermalState = HealthResponse.thermalLabel(
            ProcessInfo.processInfo.thermalState)
        self.lastDecodeTokensPerSec = lastDecodeTokensPerSec
        self.powerAssertionHeld = powerAssertionHeld
        self.gpuClockMHz = gpuClockMHz
        self.gpuActiveResidency = gpuActiveResidency
        self.mlxActiveBytes = MLX.Memory.activeMemory
        self.mlxCacheBytes = MLX.Memory.cacheMemory
        self.mlxCacheLimitBytes = MLX.Memory.cacheLimit
        self.admissionMode = snapshot.admissionMode
        self.modules = snapshot.modules.map { m in
            ModuleHealth(
                id: m.id, state: m.state, residentBytes: m.residentBytes,
                evictable: m.evictable, unloadedReason: m.unloadedReason,
                model: residentModels[m.id], measured: m.measured)
        }
        self.inflight = inflight
        self.lastRequestAt = lastRequestAt
        self.gateWaiters = gate.waiters
        self.gateHeld = gate.held
        self.gateHeldMs = gate.heldMs
        self.gateMaxWaiters = gate.maxWaiters
        self.gateWaitP95Ms = gate.waitP95Ms
    }

    /// M60.1 — stable string for the `ProcessInfo.ThermalState` enum so
    /// the field is self-describing in JSON (the raw enum is an Int).
    static func thermalLabel(_ s: ProcessInfo.ThermalState) -> String {
        switch s {
        case .nominal: return "nominal"
        case .fair: return "fair"
        case .serious: return "serious"
        case .critical: return "critical"
        @unknown default: return "unknown"
        }
    }
}

enum TLSConfigError: Error, CustomStringConvertible {
    case incomplete
    var description: String {
        switch self {
        case .incomplete:
            return
                "TLS misconfigured: set BOTH tls_cert and tls_key (or "
                + "neither). One without the other refuses to start so "
                + "the daemon never silently serves plaintext."
        }
    }
}
