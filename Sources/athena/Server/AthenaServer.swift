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
            return Self.json(
                HealthResponse(
                    snapshot: snapshot,
                    residentModels: residentModels,
                    inflight: inflight,
                    lastRequestAt: lastAt,
                    lastDecodeTokensPerSec: decodeTps,
                    powerAssertionHeld: powerAssertionHeld,
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
            return Self.prometheusText(
                AthenaMetrics.prometheus(
                    snap, now: Date().timeIntervalSince1970))
        }

        // Machine-readable API description (M32.1) — always open, like
        // /healthz, so the appliance can describe itself. Hand-authored
        // OpenAPI 3.0.3 with the running daemon's version stamped in.
        router.get("/openapi.json") { _, _ -> Response in
            Self.jsonString(OpenAPISpec.json(version: Athena.appVersion))
        }

        // Monitoring dashboard (M11.2) — inbound-only.
        router.get("/ui") { request, _ -> Response in
            await handleUIDashboard(request)
        }
        router.get("/ui/api/state") { _, _ -> Response in
            await handleUIState()
        }
        router.get("/ui/config") { request, _ -> Response in
            await handleUIConfigPage(request)
        }
        router.get("/ui/api/config") { _, _ -> Response in
            await handleUIConfigGet()
        }
        router.post("/ui/api/config") { request, _ -> Response in
            await handleUIConfigPost(request)
        }

        // Model console (M18.2). SESSION-cookie authed (AuthPolicy
        // gates /ui* on daemonAdmin); each handler ALSO re-checks the
        // logged-in user's model.read/model.write and (mutations) the
        // CSRF token, then REUSES the M16 op methods (ModelStoreOps /
        // performModelOp) — no self-HTTP, no duplication. All static
        // literals ⇒ no Hummingbird trie-order hazard.
        router.get("/ui/models") { request, _ -> Response in
            await handleUIModelsPage(request)
        }
        router.get("/ui/api/models") { request, _ -> Response in
            await uiModelsList(request)
        }
        router.get("/ui/api/models/show") { request, _
            -> Response in
            await uiModelShow(request)
        }
        router.get("/ui/api/models/default") { request, _
            -> Response in
            await uiDefaultGet(request)
        }
        router.post("/ui/api/models/default") { request, _
            -> Response in
            await uiModelMutate(request) {
                await self.handleDefaultModelSet($0)
            }
        }
        router.post("/ui/api/models/rm") { request, _
            -> Response in
            await uiModelRemove(request)
        }
        router.post("/ui/api/models/copy") { request, _
            -> Response in
            await uiModelMutate(request) {
                await self.handleModelCopy($0)
            }
        }
        // ADR 025 S2 — model ops run synchronously now (no async queue).
        // The browser console isn't an SSE consumer (EventSource is
        // GET-only), so these POSTs BLOCK until the op completes and return
        // the terminal result JSON (or the error envelope). The CLI gets
        // streamed SSE progress on the same `/api/models/*` routes.
        router.post("/ui/api/models/pull") { request, _
            -> Response in
            await uiModelMutate(request) { r in
                await self.uiModelOp(kind: "model_pull", r)
            }
        }
        router.post("/ui/api/models/convert") { request, _
            -> Response in
            await uiModelMutate(request) { r in
                await self.uiModelOp(kind: "model_convert", r)
            }
        }
        router.post("/ui/api/models/prune") { request, _
            -> Response in
            await uiModelMutate(request) { r in
                await self.uiModelOp(kind: "model_prune", r)
            }
        }

        // Daemon control console (M18.3). The WebUI runs INSIDE the
        // daemon ⇒ it controls a RUNNING daemon (warm/unload the
        // model, read posture); it CANNOT cold-start a stopped
        // daemon (nothing serves /ui then — use launchd/CLI). Cookie
        // + per-action daemonAdmin re-check; CSRF on the mutations.
        router.get("/ui/daemon") { request, _ -> Response in
            await handleUIDaemonPage(request)
        }
        router.get("/ui/api/admin/status") { request, _
            -> Response in
            await uiAdminStatus(request)
        }
        router.post("/ui/api/admin/load") { request, _
            -> Response in
            await uiAdminLoad(request)
        }
        router.post("/ui/api/admin/stop") { request, _
            -> Response in
            await uiAdminStop(request)
        }

        // ADR 026 — the allowlist is retired; availability IS the model store
        // (classified by ModelSupport), so the `/ui/allowlist` console and its
        // `/ui/api/allowlist*` mutators are gone. The read-only resident/store
        // projection stays via `/ui` + `/api/models/resident`.

        // RBAC admin console (M18.4). Cookie + per-action
        // users.admin/tokens.admin re-check + CSRF, then REUSE the
        // M16.4 handlers (now cookie-aware via callerPermissions) so
        // server-side canGrant / last-admin / cross-rank guards bind
        // to the logged-in user. All static literals (name/role/
        // prefix ride the JSON body, not a :param) ⇒ no trie hazard.
        router.get("/ui/users") { request, _ -> Response in
            await handleUIUsersPage(request)
        }
        router.get("/ui/api/users") { request, _ -> Response in
            await uiUsersList(request)
        }
        router.post("/ui/api/users") { request, _ -> Response in
            await uiUserCreate(request)
        }
        router.post("/ui/api/users/delete") { request, _
            -> Response in
            await uiUserDelete(request)
        }
        router.post("/ui/api/users/role/grant") { request, _
            -> Response in
            await uiRoleGrant(request)
        }
        router.post("/ui/api/users/role/revoke") { request, _
            -> Response in
            await uiRoleRevoke(request)
        }
        router.get("/ui/api/roles") { request, _ -> Response in
            await uiRolesList(request)
        }
        router.get("/ui/api/tokens") { request, _ -> Response in
            await uiTokensList(request)
        }
        router.post("/ui/api/tokens") { request, _ -> Response in
            await uiTokenCreate(request)
        }
        router.post("/ui/api/tokens/delete") { request, _
            -> Response in
            await uiTokenDelete(request)
        }

        // WebUI session login (M12.2). /ui/login + /ui/logout are
        // open (AuthPolicy); the rest of /ui* needs the cookie.
        router.get("/ui/login") { _, _ -> Response in
            Self.html(Self.loginPage(error: nil))
        }
        router.post("/ui/login") { request, context -> Response in
            // A3: throttle by the TCP peer address only (ADR 004 — never
            // trust X-Forwarded-For). `remoteAddress` comes off the channel
            // via AppRequestContext.
            await handleUILoginPost(
                request, peerIP: context.remoteAddress?.ipAddress)
        }
        router.get("/ui/logout") { _, _ -> Response in
            Self.logoutResponse(secure: tlsEnabled)
        }
        router.post("/ui/logout") { _, _ -> Response in
            Self.logoutResponse(secure: tlsEnabled)
        }

        router.post("/v1/chat/completions") { request, _ -> Response in
            await handleChatCompletions(request)
        }

        // ADR 036 — Anthropic Messages dialect over the same inference engine.
        router.post("/v1/messages") { request, _ -> Response in
            await handleAnthropicMessages(request)
        }

        // OpenAI model discovery (M31.1). Read-only projection of the
        // SAME model store the native `/api/models` serves, in the
        // OpenAI list/retrieve shape so drop-in SDK/LiteLLM clients can
        // probe. Gated `model.read` (AuthPolicy maps `/v1/models*`).
        router.get("/v1/models") { _, _ -> Response in
            handleOpenAIModelsList()
        }
        router.get("/v1/models/:id") { _, context -> Response in
            handleOpenAIModelRetrieve(context.parameters.get("id"))
        }

        router.post("/v1/embeddings") { request, _ -> Response in
            await handleEmbeddings(request)
        }

        router.post("/v1/audio/transcriptions") { request, _ -> Response in
            await handleTranscriptions(request)
        }

        router.post("/v1/audio/diarizations") { request, _ -> Response in
            await handleDiarizations(request)
        }

        router.post("/v1/audio/embeddings") { request, _ -> Response in
            await handleSpeakerEmbeddings(request)
        }

        // ADR 022 — Athena-native (NOT OpenAI; OpenAI has no video API). Demux
        // the audio track and transcribe it via the same Whisper/Parakeet
        // tenant; the response shape mirrors /v1/audio/transcriptions.
        router.post("/v1/video/transcriptions") { request, _ -> Response in
            await handleVideoTranscriptions(request)
        }

        // ADR 025 S2 — the async request queue (`/v1/queue*`) was removed
        // entirely. Inference is synchronous on `/v1/*`; long-running model
        // ops stream SSE progress on `/api/models/{pull,convert,prune}`.

        // Athena-native API (M16). `/v1/*` is the single inference surface;
        // `/api/*` is the CONTROL plane (clean minimal JSON, NOT Ollama, NOT
        // OpenAI). Native inference `/api/chat` was removed (ADR 031/013) — it
        // duplicated `/v1/chat/completions` with zero callers; `/api/embed`
        // stays for now on the shared governed embed path (ADR 013 deprecated,
        // 0 callers).
        router.post("/api/embed") { request, _ -> Response in
            await handleNativeEmbed(request)
        }
        router.post("/api/admin/stop") { request, _ -> Response in
            await adminUnloadLLM(request)
        }
        router.get("/api/admin/status") { _, _ -> Response in
            await adminStatus()
        }

        // Per-principal token usage (M27.3). Inference-tier (any
        // authenticated caller sees its OWN usage); an admin sees every
        // principal — owner-scoped like the queue. Pull only — the
        // passive oracle never pushes usage out.
        router.get("/api/usage") { request, _ -> Response in
            await handleUsage(request)
        }

        // Append-only RBAC/admin audit trail (M30.2). Admin-only
        // (AuthPolicy → daemon.admin); filterable, pull only.
        router.get("/api/audit") { request, _ -> Response in
            await handleAudit(request)
        }

        // M45.5 — daemon unified-log oversight. Admin-only; mirrors
        // the audit trail's sensitivity profile (entries carry
        // req=/principal=/function= across all users + reveal internal
        // call sites). `/api/logs` = one-shot historical;
        // `/api/logs/stream` = SSE follow.
        router.get("/api/logs") { request, _ -> Response in
            await handleLogs(request)
        }
        router.get("/api/logs/stream") { request, _ -> Response in
            await handleLogsStream(request)
        }

        // M59.4 — prompt-prefix cache operator surface. Admin-only
        // (AuthPolicy → daemon.admin). GET = pool stats; DELETE = flush the
        // pool (entries not in use), audited at the shared chokepoint.
        router.get("/api/cache/prompt") { _, _ -> Response in
            handlePromptCacheStats()
        }
        router.delete("/api/cache/prompt") { request, _ -> Response in
            await handlePromptCacheFlush(request)
        }

        // Model store (M16.2). Literal sub-paths are registered
        // BEFORE `:name` so Hummingbird's trie (sibling match in
        // registration order) resolves them first; `default`/`copy`
        // are therefore reserved names for show/rm.
        router.get("/api/models") { _, _ -> Response in
            handleModelsList()
        }
        router.get("/api/models/default") { _, _ -> Response in
            handleDefaultModelGet()
        }
        router.put("/api/models/default") { request, _ -> Response in
            await handleDefaultModelSet(request)
        }
        router.post("/api/models/copy") { request, _ -> Response in
            await handleModelCopy(request)
        }
        // Long-running ops (model.write-gated routes). ADR 025 S2: they run
        // SYNCHRONOUSLY and stream SSE progress directly on this route — no
        // job id, no async queue, no persistence. `data: {"event":…}` frames
        // (progress / done / error) end with `[DONE]`.
        router.post("/api/models/pull") { request, _ -> Response in
            await handleModelPull(request)
        }
        router.post("/api/models/convert") { request, _ -> Response in
            await handleModelConvert(request)
        }
        router.post("/api/models/prune") { request, _ -> Response in
            await handleModelPrune(request)
        }
        // Explicit per-module model lifecycle (M41.1): rebind a slot or
        // release it without bouncing the daemon. Generalizes M39's
        // embedding pattern across every module class (llm /
        // textEmbedding / transcription / diarization / speakerEmbedding).
        // Literal sub-paths so they win over the `:name` route below.
        router.get("/api/models/resident") { _, _ -> Response in
            await handleModelsResident()
        }
        router.post("/api/models/load") { request, _ -> Response in
            await handleModelsLoad(request)
        }
        router.post("/api/models/unload") { request, _ -> Response in
            await handleModelsUnload(request)
        }
        // ADR 026 — `/api/models/allow*` retired. Availability = the model
        // store classified by ModelSupport; the per-module default is a TOML
        // config key (`athena default --module M <id>`). The read-only
        // `/api/models/resident` projection + the M41 load/unload endpoints
        // remain, re-pointed at the store.
        router.get("/api/models/:name") { _, context -> Response in
            handleModelShow(context.parameters.get("name"))
        }
        router.delete("/api/models/:name") { request, context
            -> Response in
            await handleModelRemove(
                context.parameters.get("name"), request)
        }

        // RBAC administration over HTTP (M16.4). Every mutation
        // enforces RBAC.canGrant against the CALLER's permission set
        // (NOT implicit-admin like the offline CLI) + last-admin
        // protection. Perm-gated by AuthPolicy
        // (users.read/users.admin/tokens.admin).
        router.get("/api/users") { _, _ -> Response in
            await handleUsersList()
        }
        router.post("/api/users") { request, _ -> Response in
            await handleUserCreate(request)
        }
        router.delete("/api/users/:name") { request, context
            -> Response in
            await handleUserDelete(
                context.parameters.get("name"), request)
        }
        router.post("/api/users/:name/roles/:role") {
            request, context -> Response in
            await handleRoleGrant(
                context.parameters.get("name"),
                context.parameters.get("role"), request)
        }
        router.delete("/api/users/:name/roles/:role") {
            request, context -> Response in
            await handleRoleRevoke(
                context.parameters.get("name"),
                context.parameters.get("role"), request)
        }
        router.get("/api/roles") { _, _ -> Response in
            Self.rolesCatalogResponse()
        }
        router.get("/api/tokens") { _, _ -> Response in
            await handleTokensList()
        }
        router.post("/api/tokens") { request, _ -> Response in
            await handleTokenCreate(request)
        }
        router.delete("/api/tokens/:prefix") { request, context
            -> Response in
            await handleTokenDelete(
                context.parameters.get("prefix"), request)
        }
        router.post("/api/tokens/:prefix/rotate") { request, context
            -> Response in
            await handleTokenRotate(
                context.parameters.get("prefix"), request)
        }

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

    private func handleChatCompletions(_ request: Request) async -> Response {
        let body: ChatCompletionRequest
        do {
            let buffer = try await request.body.collect(upTo: maxRequestBodyBytes)
            let data = Data(buffer: buffer)
            body = try JSONDecoder().decode(
                ChatCompletionRequest.self, from: data)
        } catch {
            return Self.error(
                status: .badRequest,
                message: "Invalid request body: \(error)",
                type: "invalid_request_error",
                code: "invalid_body")
        }

        // M31.3: reject params that fight the greedy/MTP/structured
        // determinism up front (n>1, logprobs, logit_bias) — a clear 400
        // instead of silently ignoring them, so a drop-in OpenAI client
        // gets honest feedback rather than wrong assumptions.
        if let bad = body.unsupportedParameter() {
            return Self.error(
                status: .badRequest,
                message:
                    "'\(bad)' is not supported by this deterministic "
                    + "(greedy/structured) inference path",
                type: "invalid_request_error",
                code: "unsupported_parameter")
        }

        // The governed path: load the LLM under the global budget and
        // (M41.2) rebind the slot to body.model when the request asks for
        // a specific allowlist member — a budget event still becomes a
        // 503 here; an unknown id becomes a 400 `model_not_available`
        // (never a silent fallback or on-request download). M41.4: an
        // actual rebind emits a `model.rebind` audit record.
        //
        // ADR 015 — block-until-ready. For a STREAMING request whose model
        // isn't resident, defer the load into the SSE producer so it can emit
        // `: loading` keep-alives; that commits the response to 200, so the
        // model-DEPENDENT checks (vision / prompt-cap / rebind) also move into
        // the producer as in-stream errors. Everything else (non-stream, warm
        // stream, download-in-progress) resolves the load here as a clean HTTP
        // status, exactly as before.
        let wantStream = (body.stream == true)
        // M24.1/M71.1 — decode the FULL conversation (system/user/assistant/
        // tool) plus any OpenAI `image_url` content-parts. ADR 036 S1b: decode
        // runs BEFORE the orchestration seam, so a decode fault (a bad/remote
        // image URL → 400, passive-oracle: no outbound image fetch) precedes
        // admission for a request that trips both.
        let turns: [ChatTurn]
        do {
            turns = try Self.chatTurns(from: body.messages)
        } catch {
            return Self.imageContentError(error)
        }
        let toolSpecs = body.toolSpecs()
        // ADR 036 S1b — the protocol-agnostic orchestration seam: model
        // admission + cold-load decision (ADR 015) + resident-model vision /
        // prompt-cap preflight. `.deferToStream` carries the in-producer
        // load/check closures (cold-load streaming, ADR 015); `.ready` means the
        // model is resident and the model-dependent checks passed inline. The
        // Anthropic `/v1/messages` adapter (ADR 036 S2) calls this same seam.
        let model: String
        let deferredLoad: (@Sendable () async -> ColdStreamLoad)?
        let deferredPrepare:
            (@Sendable () async -> (message: String, type: String, code: String)?)?
        switch await prepareChat(
            request: request, requestedModel: body.model, messages: turns,
            tools: toolSpecs, chatTemplateKwargs: body.chatTemplateKwargsContext(),
            wantStream: wantStream)
        {
        case .failed(let response):
            return response
        case .ready(let resolved):
            model = resolved
            deferredLoad = nil
            deferredPrepare = nil
        case .deferToStream(let load, let prepare):
            model = ""
            deferredLoad = load
            deferredPrepare = prepare
        }
        let deferLoadIntoStream = (deferredLoad != nil)

        // G4 fail-closed: a `response_format: json_schema` with a
        // missing/unserializable schema is a 400 here, never a silent
        // fall-through to unconstrained output.
        if let problem = body.structuredRequestError() {
            return Self.error(
                status: .badRequest, message: problem,
                type: "invalid_request_error",
                code: "invalid_response_format")
        }

        let created = Int(Date().timeIntervalSince1970)
        let id = "chatcmpl-\(UUID().uuidString)"
        let effective = body.effectiveSchema()
        let schemaJSON = effective?.json
        // `toolSpecs` is computed once above (ADR 036 S1b, before the seam).

        // C2 (ADR 013 §4): honor logprobs on the deterministic decode path
        // (greedy temp==0, or structured where temperature is inert); 400 on a
        // sampling request, whose path has no logit-capture seam. Resolved here
        // so BOTH the streamed and non-streamed branches share the verdict.
        let deterministic = (body.temperature == 0) || (schemaJSON != nil)
        if let (msg, code) = body.logprobsValidationError(
            deterministic: deterministic)
        {
            return Self.error(
                status: .badRequest, message: msg,
                type: "invalid_request_error", code: code)
        }
        let logprobsReq =
            body.wantsLogprobs
            ? LogprobsRequest(topLogprobs: body.topLogprobsValue) : nil

        let stops = body.stopSequences()
        // M59.3 — resolve the principal once: it scopes the prompt-prefix
        // cache (so reuse never crosses callers) on BOTH the streamed and
        // non-streamed branches, and meters usage.
        let principal = await usagePrincipal(request)

        // ADR 036 S1a — the dialect-agnostic engine request. Built once from the
        // resolved OpenAI params and consumed identically by all three terminal
        // paths (deferred-cold-load stream, warm stream, blocking), so the
        // engine call no longer appears as three hand-kept-in-sync argument
        // lists. The Anthropic adapter (S2) maps its wire request onto this same
        // type.
        let native = NativeChatRequest(
            model: body.model,  // WP6 — bind this model inside the decode gate
            messages: turns, schemaJSON: schemaJSON, tools: toolSpecs,
            maxTokens: body.tokenCap, temperature: body.temperature,
            topP: body.top_p, seed: body.seed, speculative: body.speculative,
            chatTemplateKwargs: body.chatTemplateKwargsContext(),
            promptCacheKey: body.prompt_cache_key, principal: principal,
            logprobs: logprobsReq)

        if body.stream == true {
            // M27.4: meter streamed requests too, and emit a terminal
            // usage chunk when the client opted in via stream_options.
            let includeUsage = body.stream_options?.include_usage == true
            // M46.3 — per-request `timeout` overrides the daemon-wide
            // `request_timeout_secs`. nil ⇒ inherit; 0/negative ⇒
            // disable the deadline for this call only.
            let deadlineSecs =
                body.timeout.map { $0 > 0 ? $0 : 0 }
                ?? requestTimeoutSecs
            // A8/E3/E13 (M68.4) — the streamed path doesn't go through
            // `collectMetered`, so pre-fix it bound NO `DecodeProgress.counter`
            // and never called `cancelGeneration()`: a client disconnect or a
            // deadline truncation ended the SSE wire but the synchronous decode
            // loop (polling the counter, not `Task.isCancelled`) ran on to
            // `maxTokens`. Bind a cancel counter HERE — `generateMetered`'s
            // (non-detached) Task, created synchronously inside this
            // `withValue` scope, inherits the TaskLocal so the loop's task sees
            // it (E13) — and flip it on BOTH a downstream disconnect (A8) and a
            // deadline truncation (E3).
            let cancelCounter = HeartbeatCounter()
            if deferLoadIntoStream {
                // ADR 015 — cold-load streaming: open the SSE 200, emit
                // `: loading` keep-alives while the model loads, run the
                // model-dependent checks in-band, then decode. A load timeout
                // or failure becomes an in-stream error, not a dropped wire.
                return DecodeProgress.$counter.withValue(cancelCounter) {
                    Self.streamSSEAwaitingLoad(
                        id: id, created: created,
                        modelName: { await servedLLMModel() },
                        // ADR 036 S1b — the cold-load + model-dependent-check
                        // closures come from the orchestration seam (`prepareChat`
                        // → `.deferToStream`), shared with every chat adapter.
                        // Force-unwrap is safe: this branch is `deferLoadIntoStream`,
                        // which is exactly `deferredLoad != nil`.
                        load: deferredLoad!,
                        prepareAfterLoad: deferredPrepare!,
                        eventsBuilder: {
                            deadlineBounded(
                                seconds: deadlineSecs,
                                llm.generateMetered(native),
                                onTimerFired: {
                                    cancelCounter.cancelGeneration()
                                    Self.log.warning(
                                        """
                                        streamed request truncated by deadline \
                                        path=/v1/chat/completions seconds=\
                                        \(deadlineSecs)
                                        """)
                                })
                        },
                        includeUsage: includeUsage,
                        isToolCall: effective?.isToolCall == true, stops: stops,
                        onConsumerCancel: { cancelCounter.cancelGeneration() },
                        record: { usage in
                            await meter(principal: principal, usage: usage)
                        })
                }
            }
            return DecodeProgress.$counter.withValue(cancelCounter) {
                Self.streamSSE(
                    id: id, model: model, created: created,
                    events: deadlineBounded(
                        seconds: deadlineSecs,
                        llm.generateMetered(native),
                        onTimerFired: {
                            // E3 — a deadline truncation must reach the decode
                            // loop, not just close the wire.
                            cancelCounter.cancelGeneration()
                            Self.log.warning(
                                """
                                streamed request truncated by deadline \
                                path=/v1/chat/completions seconds=\
                                \(deadlineSecs) model=\(model)
                                """)
                        }),
                    includeUsage: includeUsage,
                    isToolCall: effective?.isToolCall == true, stops: stops,
                    onConsumerCancel: { cancelCounter.cancelGeneration() },
                    record: { usage in
                        await meter(principal: principal, usage: usage)
                    })
            }
        }

        let collected: GenCollected
        do {
            // M46.3 — per-request `timeout` overrides the daemon-wide
            // `request_timeout_secs` for this single call.
            let deadlineSecs =
                body.timeout.map { $0 > 0 ? $0 : 0 }
                ?? requestTimeoutSecs
            collected = try await collectMetered(seconds: deadlineSecs) {
                llm.generateMetered(native)
            }
        } catch let e as AthenaError {
            // M33.1: the only AthenaError collectMetered raises is the
            // per-request timeout → classified 504.
            return Self.error(
                status: HTTPResponse.Status(code: e.httpStatus),
                message: e.message, type: "server_error", code: e.code)
        } catch {
            return Self.classified(error, module: .llm)
        }
        // ADR 035 — pull channel-delimited reasoning (`<|channel>thought…
        // <channel|>`) out of the content before anything else; surface it as
        // `reasoning_content`. No-op for models that don't emit the markers.
        let split = splitReasoningChannel(collected.text)
        var text = split.content
        let reasoning = split.reasoning.isEmpty ? nil : split.reasoning
        let usage = collected.usage
        var finish = collected.finish
        // M31.3: truncate at the first stop sequence; a stop hit reports
        // finish_reason "stop" (it overrides a length cap reached later in
        // the same generation).
        if !stops.isEmpty {
            let cut = StopStreamFilter.truncate(text, stops: stops)
            if cut.stopped {
                text = cut.text
                finish = .stop
            }
        }
        // M27.1/.2: real token counts feed the response `usage` object,
        // the global metrics counter, and the persisted per-principal
        // counter (keyed by the caller's auth principal).
        // NA8 — reuse the principal already resolved at the top of this
        // handler instead of a second full bearer resolution (SHA-256 +
        // two SQLite lookups) per request.
        await meter(principal: principal, usage: usage)

        return Self.json(
            Self.chatCompletionResponse(
                id: id, model: model, created: created, text: text,
                reasoning: reasoning,
                isToolCall: effective?.isToolCall == true,
                detectedToolCall: collected.toolCall, usage: usage,
                finish: finish, logprobs: collected.logprobs))
    }

    /// Build one `ChatChoice` from generated text: a tool-call object is
    /// surfaced as OpenAI `tool_calls`; everything else as `content`.
    /// Shared by the sync `/v1/chat/completions` handler and the queued
    /// `conversation` executor so both emit the identical OpenAI shape.
    /// `finish` is the generator's stop reason (M31.2): a real tool call
    /// always reports `tool_calls`; otherwise the reason passes through
    /// (`stop` natural end, `length` max_tokens truncation).
    /// C2 — map the module's `[TokenLogprob]` into the OpenAI response object
    /// (`choices[].logprobs.content`). nil ⇒ nil (omitted from JSON).
    static func chatLogprobs(_ lps: [TokenLogprob]?) -> ChatLogprobs? {
        guard let lps else { return nil }
        return ChatLogprobs(
            content: lps.map { t in
                ChatCompletionTokenLogprob(
                    token: t.token, logprob: Double(t.logprob),
                    bytes: t.bytes,
                    top_logprobs: t.top.map {
                        ChatTopLogprob(
                            token: $0.token, logprob: Double($0.logprob),
                            bytes: $0.bytes)
                    })
            })
    }

    /// Build a `tool_calls` choice from a resolved (name, stringified-args)
    /// pair. Shared by the Guide-forced parse and the ADR-034 substrate-detected
    /// path so both emit the identical OpenAI shape.
    private static func toolCallChoice(
        name: String, argsJSON: String, reasoning: String?,
        logprobs: [TokenLogprob]?
    ) -> ChatChoice {
        ChatChoice(
            index: 0,
            message: ChatMessage(
                role: "assistant", content: nil,
                reasoning_content: reasoning,
                tool_calls: [
                    ToolCallOut(
                        id: "call_\(UUID().uuidString.prefix(8))",
                        type: "function",
                        function: FunctionCallOut(
                            name: name, arguments: argsJSON))
                ]),
            finish_reason: "tool_calls",
            logprobs: Self.chatLogprobs(logprobs))
    }

    private static func chatChoice(
        text: String, reasoning: String? = nil, isToolCall: Bool,
        detectedToolCall: (name: String, argsJSON: String)? = nil,
        finish: FinishReason,
        logprobs: [TokenLogprob]? = nil
    ) -> ChatChoice {
        // WP7 — the one shared tool-call precedence algebra (ADR 034): a
        // substrate-detected free call wins (already parsed); else a Guide-forced
        // call is the decoded JSON text; else plain content.
        switch resolveToolCallOutcome(
            detected: detectedToolCall, text: text, isToolCall: isToolCall)
        {
        case .detected(let n, let a), .forced(let n, let a):
            return toolCallChoice(
                name: n, argsJSON: a, reasoning: reasoning, logprobs: logprobs)
        case .none:
            return ChatChoice(
                index: 0,
                message: ChatMessage(
                    role: "assistant", content: text,
                    reasoning_content: reasoning),
                finish_reason: finish.rawValue,
                logprobs: Self.chatLogprobs(logprobs))
        }
    }

    /// Assemble a full OpenAI `ChatCompletionResponse` around one choice.
    private static func chatCompletionResponse(
        id: String, model: String, created: Int, text: String,
        reasoning: String? = nil,
        isToolCall: Bool,
        detectedToolCall: (name: String, argsJSON: String)? = nil,
        usage: TokenUsage,
        finish: FinishReason = .stop,
        logprobs: [TokenLogprob]? = nil
    ) -> ChatCompletionResponse {
        ChatCompletionResponse(
            id: id, object: "chat.completion", created: created,
            model: model,
            choices: [
                chatChoice(
                    text: text, reasoning: reasoning, isToolCall: isToolCall,
                    detectedToolCall: detectedToolCall, finish: finish,
                    logprobs: logprobs)
            ],
            usage: Usage(
                prompt_tokens: usage.promptTokens,
                completion_tokens: usage.completionTokens,
                total_tokens: usage.totalTokens,
                cachedTokens: usage.cachedTokens))
    }

    /// Accumulated result of draining a metered generation: the full
    /// text, the true token usage, and the finish reason.
    struct GenCollected: Sendable {
        var text = ""
        var usage = TokenUsage.zero
        var finish: FinishReason = .stop
        // C2 — per-token logprobs when the request asked for them.
        var logprobs: [TokenLogprob]?
        // ADR 034 — a freely-chosen tool call (tool_choice:auto) detected by
        // the substrate. nil ⇒ plain text completion (or a Guide-forced call,
        // which arrives as `text` and is parsed via `isToolCall`).
        var toolCall: (name: String, argsJSON: String)?
    }

    /// NSLock-isolated state for the M46.7 heartbeat: the event-drain
    /// loop increments `tokens` as text chunks arrive, while the
    /// detached heartbeat timer reads (tokens, lastLoggedTokens,
    /// lastLoggedAt) to compute "tokens/sec since the last heartbeat
    /// line". Cross-task touch ⇒ the lock makes the read-modify-write
    /// sound; `@unchecked Sendable` because all access is lock-mediated.
    /// Snapshot avoids holding the lock across the Logger call.
    ///
    /// M49.3 — heartbeat helpers: compact byte formatter + per-module
    /// memory tail. Free functions so the `Task.detached` closure can
    /// call them without capturing `self`.
    fileprivate static func formatBytes(_ n: Int) -> String {
        let gb = Double(n) / (1024.0 * 1024.0 * 1024.0)
        if gb >= 1.0 {
            return String(format: "%.1fGB", gb)
        }
        let mb = Double(n) / (1024.0 * 1024.0)
        return String(format: "%.0fMB", mb)
    }
    /// M56 — whole-ms elapsed since `start`, for per-request summary lines.
    fileprivate static func elapsedMs(_ start: Date) -> Int {
        Int(Date().timeIntervalSince(start) * 1000)
    }
    fileprivate static func formatModuleMemory(
        _ snap: GovernorSnapshot
    ) -> String {
        // Compact `id:GB` tail, only including loaded modules. Order
        // follows the snapshot's natural order so it stays stable
        // across heartbeats.
        let parts = snap.modules.compactMap { m -> String? in
            guard m.state == .loaded else { return nil }
            return "\(m.id.rawValue):\(formatBytes(m.residentBytes))"
        }
        return parts.isEmpty ? "" : " modules=" + parts.joined(separator: ",")
    }

    /// M46.8 — also conforms to `AthenaCore.DecodeProgressCounter`, so
    /// the synchronous decode loops (GuidedGreedy / GuidedSubstrate /
    /// SpeculativeGeneration / SpeculativeSampling) can increment the
    /// same counter via the `DecodeProgress.counter` TaskLocal. That
    /// gets a structured-output decode's per-iteration progress into
    /// the heartbeat without threading a callback through 5 layers of
    /// protocol/signature; without it the heartbeat sees `tokens=0`
    /// for the entire structured decode (the Guide path emits one
    /// `.text` event at completion, not per-token).
    final class HeartbeatCounter: @unchecked Sendable, DecodeProgressCounter {
        struct Snapshot {
            let tokens: Int
            let lastLoggedTokens: Int
            let lastLoggedAt: TimeInterval
            /// M48.4 — last-submitted prefill chunk index (1-based).
            /// 0 ⇒ prefill not started (or 1-token prompt — no chunks).
            let prefillCompleted: Int
            /// M48.4 — total prefill chunks for THIS request, or 0 if
            /// the decode loop never published a prefill state.
            let prefillTotal: Int
            /// M49.3 — current setup sub-stage (e.g. "compile-dfa",
            /// "build-vocab"). nil ⇒ either not in setup OR setup
            /// stage not annotated by the decode path.
            let setupStage: String?
        }
        private let lock = NSLock()
        private var tokens = 0
        private var lastLoggedTokens = 0
        private var lastLoggedAt: TimeInterval = 0
        private var prefillCompleted = 0
        private var prefillTotal = 0
        private var setupStage: String? = nil
        /// M60.5 — set by the serve path's task-cancellation handler (client
        /// disconnect or deadline); polled by the decode loops to stop early.
        private var cancelled = false

        func cancelGeneration() {
            lock.lock()
            defer { lock.unlock() }
            cancelled = true
        }

        var isCancelled: Bool {
            lock.lock()
            defer { lock.unlock() }
            return cancelled
        }

        func incrementToken() {
            lock.lock()
            defer { lock.unlock() }
            tokens += 1
        }

        func recordPrefillChunk(completed: Int, total: Int) {
            lock.lock()
            defer { lock.unlock() }
            self.prefillCompleted = completed
            self.prefillTotal = total
        }

        func setSetupStage(_ stage: String?) {
            lock.lock()
            defer { lock.unlock() }
            self.setupStage = stage
        }

        func snapshot() -> Snapshot {
            lock.lock()
            defer { lock.unlock() }
            return Snapshot(
                tokens: tokens,
                lastLoggedTokens: lastLoggedTokens,
                lastLoggedAt: lastLoggedAt,
                prefillCompleted: prefillCompleted,
                prefillTotal: prefillTotal,
                setupStage: setupStage)
        }

        func markLogged(elapsedAt: TimeInterval, tokens: Int) {
            lock.lock()
            defer { lock.unlock() }
            self.lastLoggedAt = elapsedAt
            self.lastLoggedTokens = tokens
        }
    }

    /// Drain a metered generation under the per-request deadline (M33.1)
    /// and return its text + usage + finish reason. `seconds` = 0 ⇒
    /// unbounded. On overrun it throws `AthenaError.requestTimedOut`
    /// (the caller maps it to a 504) and the generation is cancelled
    /// so it stops consuming the worker/budget. Shared by the sync
    /// `/v1/chat/completions`, native `/api/chat`, and queued
    /// `conversation` paths so all three honor the same timeout.
    ///
    /// M46.1 / M46.7 — long-generation heartbeat. A sync decode that
    /// runs past `heartbeatAfter` emits a `.notice`-level progress log
    /// every `heartbeatEverySec` seconds with elapsed time, tokens
    /// emitted, and current tokens/sec. Closes the silent-while-
    /// decoding gap: a 10-minute extraction-shape call previously left
    /// nothing in `log show` between request-start and request-complete,
    /// making "actively decoding" look identical to "process hung."
    ///
    /// M46.7 fix: the heartbeat now runs on an INDEPENDENT timer task,
    /// not piggy-backed on the event for-await. The original
    /// M46.1 implementation only fired when a token event arrived —
    /// so a stalled decode (model spinning on prompt-processing, KV
    /// warm-up, or genuinely hung) suspended the for-await on `await`,
    /// emitted no events, and emitted no heartbeats either, defeating
    /// the whole point. The timer task fires regardless of event
    /// throughput; "no events for >K seconds while elapsed > threshold"
    /// is exactly the alive-but-slow signal an operator needs to see.
    /// Notice-level so it persists to `log show`.
    ///
    /// M46.3 — `seconds` is the EFFECTIVE deadline for THIS call: the
    /// per-request `timeout` field on ChatCompletionRequest overrides
    /// the daemon-wide `request_timeout_secs` when present. Callers
    /// resolve this; `collectMetered` just honors what it's given.
    ///
    /// M48.1 — `eventsBuilder` is a thunk that constructs the metered
    /// `AsyncStream` (typically `llm.generateMetered(...)`). It MUST
    /// be called from inside the `DecodeProgress.$counter.withValue(...)`
    /// scope so the decode Task spawned inside the AsyncStream's
    /// initializer inherits the TaskLocal binding. The original M46.8
    /// shape (events constructed BEFORE the withValue) silently broke
    /// every structured-path heartbeat — the decode Task captured an
    /// empty TaskLocal table and `incrementToken()` was a no-op, so
    /// `tokens=0 tokens_per_sec=0.0` showed up forever even on a
    /// progressing decode. Diagnosed via process sample of a wedged
    /// daemon: worker thread was actively iterating in
    /// `GuidedGreedy.generate` while the heartbeat reported nothing.
    private func collectMetered(
        seconds: Int,
        _ eventsBuilder: @escaping @Sendable () -> AsyncStream<GenChunk>
    ) async throws -> GenCollected {
        try await withInferenceDeadline(seconds: seconds) {
            let started = Date()
            // Shared counter the timer task reads + the for-await loop
            // writes. NSLock-isolated so the cross-task touch is sound.
            let counter = HeartbeatCounter()
            // Locked defaults — quiet on a healthy short workload (no
            // log emission until 10 s in), informative on a long one
            // (one line every 5 s tells "alive, N tok/s" vs "alive,
            // 0 tok/s in the last 5 s ⇒ stalled / hung").
            let heartbeatAfterNanos: UInt64 = 10_000_000_000
            let heartbeatIntervalNanos: UInt64 = 5_000_000_000
            let governor = self.governor
            let metrics = self.metrics
            let heartbeatTask = Task.detached(
                priority: .utility
            ) { [counter, metrics] in
                // Wait out the silent threshold; if the whole call
                // finished before then, the outer defer cancels us
                // and Task.sleep throws — try? swallows it.
                try? await Task.sleep(
                    nanoseconds: heartbeatAfterNanos)
                while !Task.isCancelled {
                    let snap = counter.snapshot()
                    let elapsed = Date().timeIntervalSince(started)
                    let dt = max(elapsed - snap.lastLoggedAt, 0.001)
                    let tps =
                        Double(snap.tokens - snap.lastLoggedTokens)
                        / dt
                    // M48.4 — include prefill state when known so the
                    // operator can tell "stuck in prefill" (e.g.
                    // prefill=14/38 tokens=0) apart from "decoding"
                    // (e.g. prefill=38/38 tokens=124). The field is
                    // dropped entirely when the decode path doesn't
                    // publish prefill state (substrate-streamed
                    // unstructured requests).
                    let prefillField: String
                    if snap.prefillTotal > 0 {
                        prefillField =
                            " prefill=\(snap.prefillCompleted)/"
                            + "\(snap.prefillTotal)"
                    } else {
                        prefillField = ""
                    }
                    // M49.2 — phase label (setup / prefill / decode)
                    // derived from the counter snapshot. Lets the
                    // operator read the heartbeat line and answer
                    // "is it hung in setup or just running long?"
                    // without inferring from missing prefill fields.
                    // M49.3 — append the setup sub-stage when set so
                    // a setup-bound heartbeat says e.g.
                    // `phase=setup:compile-dfa` instead of bare
                    // `phase=setup`. Decode paths annotate via
                    // `DecodeProgress.counter?.setSetupStage(...)`.
                    let phase = DecodePhase.from(
                        tokens: snap.tokens,
                        prefillCompleted: snap.prefillCompleted,
                        prefillTotal: snap.prefillTotal)
                    let phaseField: String
                    if phase == .setup, let stage = snap.setupStage {
                        phaseField = "setup:\(stage)"
                    } else {
                        phaseField = phase.rawValue
                    }
                    // M60.1 — publish the live decode rate to the metrics
                    // actor so /healthz can report it (only while actually
                    // decoding, so the surfaced value is real throughput
                    // rather than a setup/prefill zero). Lets a client read
                    // tok/s + thermalState and back off before submitting a
                    // call that would cross its deadline.
                    if phase == .decode {
                        await metrics.recordDecodeRate(tps)
                    }
                    // M49.3 — per-module residentBytes appended so a
                    // memory-pressure regression is visible at-a-glance
                    // in the heartbeat instead of needing an out-of-band
                    // /healthz scrape during the wedge. Only modules in
                    // the .loaded state contribute (an evicted slot is
                    // 0 anyway). Compact `id:GB` form keeps the line
                    // ≤ ~200 chars even with all five modules loaded.
                    let gov = await governor.snapshot()
                    let modulesField = AthenaServer.formatModuleMemory(gov)
                    let residentField = AthenaServer.formatBytes(
                        gov.residentBytes)
                    // M49.4 — also emit the OS-level process RSS plus
                    // MLX's own active/cache memory. The 0.10.81
                    // operator report showed Activity Monitor at 142 GB
                    // while the heartbeat's per-module sum was 62 GB —
                    // an 80 GB gap that lives OUTSIDE the governor's
                    // per-module accounting (rust-shim DFA hashbrowns,
                    // unattributed Metal/heap allocations, etc.).
                    // Exposing `rss` and `mlx_active`+`mlx_cache`
                    // separately lets an operator localize that gap
                    // from one heartbeat line.
                    // M55 — `phys_footprint` (the Activity Monitor "Memory"
                    // number) counts the Metal/GPU KV-cache + prompt-cache
                    // + activation buffers that `rss` misses, so the gap
                    // (e.g. a long-context prompt cache) is now explicit on
                    // one line instead of only visible in Activity Monitor.
                    let mem = ProcessMemory.sample()
                    let rss = mem.resident
                    let physFootprint = mem.physFootprint
                    let mlxActive = MLX.Memory.activeMemory
                    let mlxCache = MLX.Memory.cacheMemory
                    Self.log.notice(
                        """
                        decode heartbeat elapsed=\(Int(elapsed))s \
                        phase=\(phaseField)\
                        \(prefillField) tokens=\(snap.tokens) \
                        tokens_per_sec=\
                        \(String(format: "%.1f", tps)) \
                        resident=\(residentField) \
                        rss=\(AthenaServer.formatBytes(rss)) \
                        phys_footprint=\
                        \(AthenaServer.formatBytes(physFootprint)) \
                        mlx_active=\(AthenaServer.formatBytes(mlxActive)) \
                        mlx_cache=\(AthenaServer.formatBytes(mlxCache))\
                        \(modulesField)
                        """)
                    counter.markLogged(
                        elapsedAt: elapsed, tokens: snap.tokens)
                    try? await Task.sleep(
                        nanoseconds: heartbeatIntervalNanos)
                }
            }
            defer { heartbeatTask.cancel() }
            var c = GenCollected()
            // M46.8 / M48.1 — bind the heartbeat counter on a TaskLocal
            // so the synchronous decode loops (GuidedGreedy /
            // GuidedSubstrate / SpeculativeGeneration /
            // SpeculativeSampling) can increment it per internal commit.
            // The events thunk is invoked INSIDE this scope so the Task
            // it spawns (inside `AsyncStream { continuation in Task {} }`)
            // inherits the TaskLocal binding — without that, the
            // structured paths' `incrementToken()` calls hit a nil
            // counter and the heartbeat reports `tokens=0` forever. The
            // event-drain increment below remains the source of truth
            // for the substrate-streamed (non-Guide) path, which emits
            // per-token `.text` events; those paths leave the TaskLocal
            // untouched so no double-counting happens.
            // M60.5 — bridge task cancellation (client disconnect or the M33
            // deadline) to the synchronous decode loops: the handler flips the
            // shared counter's cancel flag, which the loops poll and `break`
            // on, so an abandoned generation stops burning the GPU instead of
            // decoding to maxTokens. The flag is on the SAME counter object the
            // generation Task reads via the TaskLocal below, so it bridges the
            // consuming task and the (unstructured) generation task.
            try await withTaskCancellationHandler {
                try await DecodeProgress.$counter.withValue(counter) {
                    let events = eventsBuilder()
                    for await event in events {
                        switch event {
                        case .text(let chunk):
                            c.text += chunk
                            counter.incrementToken()
                        case .usage(let u): c.usage = u
                        case .finish(let r): c.finish = r
                        case .logprobs(let l): c.logprobs = l
                        case .toolCall(let name, let argsJSON):
                            // ADR 034 — substrate-detected free tool call.
                            c.toolCall = (name, argsJSON)
                        case .error(let athenaErr):
                            // M49.5.2 — re-throw the classified error so the
                            // HTTP layer's `do { ... } catch let e as AthenaError`
                            // returns the right status/code instead of a 200
                            // with the error text in the chat content.
                            throw athenaErr
                        }
                    }
                }
            } onCancel: {
                counter.cancelGeneration()
            }
            // M60.6 — shed the prompt-prefix KV pool if this request pushed the
            // process footprint over the high-water mark, so a pool that grew
            // under sustained decode is reclaimed now instead of staying pinned
            // over budget until the next model load. No-op (a cheap phys probe)
            // when there's headroom.
            await governor.relievePromptCachePressureIfNeeded()
            return c
        }
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

    /// ADR 036 S2 — the Anthropic Messages adapter (`POST /v1/messages`). Decode
    /// → `NativeChatRequest`, run the shared `prepareChat` seam, drain the one
    /// `GenChunk` stream, encode the Anthropic response. No orchestration or
    /// engine logic here — it reuses the exact path the OpenAI adapter does.
    /// Increment 1: non-streaming. Streaming (the SSE event protocol) lands in
    /// the next slice; a `stream:true` request is refused with a clean 400 until
    /// then.
    private func handleAnthropicMessages(_ request: Request) async -> Response {
        let body: AnthropicMessagesRequest
        do {
            let buffer = try await request.body.collect(upTo: maxRequestBodyBytes)
            body = try JSONDecoder().decode(
                AnthropicMessagesRequest.self, from: Data(buffer: buffer))
        } catch {
            return Self.anthropicError(
                status: .badRequest, message: "Invalid request body: \(error)")
        }
        let principal = await usagePrincipal(request)
        let lowered: AnthropicMessagesRequest.Lowered
        do {
            lowered = try body.lower(principal: principal)
        } catch let e as AnthropicDecodeError {
            return Self.anthropicError(status: .badRequest, message: e.message)
        } catch {
            return Self.anthropicError(
                status: .badRequest, message: "\(error)")
        }
        // The shared orchestration seam (ADR 036 S1b) — identical to the OpenAI
        // path. `wantStream:false` ⇒ block-until-ready (ADR 015) inline, never
        // `.deferToStream`, so a streamed Anthropic request to a cold model
        // blocks then streams (the `: loading` keep-alive is deferred for this
        // dialect). A pre-commitment fault surfaces Athena's canonical envelope
        // (accepted honesty boundary, ADR 036).
        let model: String
        switch await prepareChat(
            request: request, requestedModel: body.model,
            messages: lowered.native.messages, tools: lowered.native.tools,
            chatTemplateKwargs: nil, wantStream: false)
        {
        case .failed(let response): return response
        case .ready(let resolved): model = resolved
        case .deferToStream:
            return Self.anthropicError(
                status: .serviceUnavailable, message: "model is loading")
        }

        // WP7 — resolve the per-request deadline the SAME way the OpenAI path
        // does (`timeout` override, else the daemon default), so both dialects
        // honor it uniformly.
        let deadlineSecs =
            lowered.timeout.map { $0 > 0 ? $0 : 0 } ?? requestTimeoutSecs

        // Streaming: forward the one GenChunk stream as the Anthropic event
        // sequence. Mirrors the OpenAI warm-stream cancel/deadline wiring
        // (A8/E3) so a client disconnect or deadline stops the decode.
        if lowered.wantStream {
            let cancelCounter = HeartbeatCounter()
            let msgID = "msg_\(UUID().uuidString)"
            return DecodeProgress.$counter.withValue(cancelCounter) {
                Self.streamAnthropic(
                    id: msgID, model: model,
                    events: deadlineBounded(
                        seconds: deadlineSecs,
                        llm.generateMetered(lowered.native),
                        onTimerFired: {
                            cancelCounter.cancelGeneration()
                            Self.log.warning(
                                "streamed request truncated by deadline path=/v1/messages")
                        }),
                    isToolCall: lowered.isToolCall, stops: lowered.stops,
                    onConsumerCancel: { cancelCounter.cancelGeneration() },
                    record: { usage in
                        await meter(principal: principal, usage: usage)
                    })
            }
        }

        let collected: GenCollected
        do {
            collected = try await collectMetered(seconds: deadlineSecs) {
                llm.generateMetered(lowered.native)
            }
        } catch let e as AthenaError {
            return Self.anthropicError(
                status: HTTPResponse.Status(code: e.httpStatus),
                message: e.message)
        } catch {
            let c = AthenaError.classify(error, module: .llm)
            return Self.anthropicError(
                status: HTTPResponse.Status(code: c.httpStatus),
                message: c.message)
        }
        // ADR 035 — strip channel-reasoning so it never leaks into the text.
        var text = splitReasoningChannel(collected.text).content
        var stopHit: String?
        if !lowered.stops.isEmpty {
            let cut = StopStreamFilter.truncate(text, stops: lowered.stops)
            if cut.stopped {
                stopHit = lowered.stops.first { text.contains($0) }
                text = cut.text
            }
        }
        await meter(principal: principal, usage: collected.usage)
        // WP7 — the one shared tool-call precedence algebra (ADR 034/036): a
        // substrate-detected free call keeps the text as content; a Guide-forced
        // call IS the text (drop it); else plain content.
        let finalText: String
        let toolCall: (name: String, argsJSON: String)?
        switch resolveToolCallOutcome(
            detected: collected.toolCall, text: text,
            isToolCall: lowered.isToolCall)
        {
        case .detected(let n, let a):
            finalText = text
            toolCall = (n, a)
        case .forced(let n, let a):
            finalText = ""
            toolCall = (n, a)
        case .none:
            finalText = text
            toolCall = nil
        }
        return Self.json(
            AnthropicMessagesResponse.make(
                id: "msg_\(UUID().uuidString)", model: model, text: finalText,
                toolCall: toolCall, promptTokens: collected.usage.promptTokens,
                completionTokens: collected.usage.completionTokens,
                finishIsLength: collected.finish == .length,
                stopSequenceHit: stopHit))
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

        do {
            // M43.2: never block the request thread on a multi-GB
            // cold-load. `.loading` ⇒ 503+Retry-After so the client
            // paces its retries instead of hitting its own timeout.
            switch try await governor.awaitLoad(.textEmbedding, within: coldLoadWaitSecs) {
            case .loaded: break
            case .loading: return Self.coldLoadResponse(.textEmbedding)
            }
            // M41.4: a real resident-id change from per-request `model`
            // is audited (the embedder also self-rebinds inside embed(),
            // which becomes a no-op once the slot already matches).
            if let m = body.model, !m.isEmpty {
                try await auditedRebind(
                    request, module: .textEmbedding, target: m)
            }
        } catch {
            // issue #6: route every failure through the classification seam
            // (correct 4xx + `type`, OOM→503, no substrate-detail leak to the
            // client body) instead of a catch-all 500 `internal_error`.
            return Self.classified(error, module: .textEmbedding)
        }

        // M39: `body.model` selects among the configured set. The module
        // rebinds its slot (or 400s on an unknown id) and reports the id
        // actually served, which we echo back — no longer the unused
        // request string.
        let batch: EmbeddingBatch
        do {
            batch = try await embedding.embed(body.input, model: body.model)
        } catch {
            return Self.classified(error, module: .textEmbedding)
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

    /// WP9 — the shared multipart upload preamble for the media routes
    /// (transcription, video, diarization, speaker-embedding): content-type +
    /// boundary → ADR-017 cap fast-fail (declared `Content-Length`) →
    /// `collect(upTo:cap)` with a streamed 413 backstop → parse → the required
    /// non-empty `file` part. `cap` is the modality's byte ceiling (audio vs
    /// video), so one helper closes the drift between the four hand-copied
    /// blocks (the audio/video cap had already diverged).
    private func extractUploadFile(
        _ request: Request, cap: Int
    ) async -> Outcome<(form: MultipartForm, file: MultipartForm.Part)> {
        guard
            let ct = request.headers[.contentType],
            let boundary = MultipartForm.boundary(fromContentType: ct)
        else {
            return .fail(
                Self.error(
                    status: .badRequest,
                    message: "expected multipart/form-data with a boundary",
                    type: "invalid_request_error", code: "invalid_content_type"))
        }
        if let tooLarge = Self.payloadTooLarge(request, cap: cap) {
            return .fail(tooLarge)
        }
        let body: Data
        do {
            let buffer = try await request.body.collect(upTo: cap)
            body = Data(buffer: buffer)
        } catch is NIOTooManyBytesError {
            return .fail(Self.tooLargeResponse(cap: cap))
        } catch {
            return .fail(
                Self.error(
                    status: .badRequest, message: "invalid request body",
                    type: "invalid_request_error", code: "invalid_body"))
        }
        guard
            let form = MultipartForm(body: body, boundary: boundary),
            let file = form.first("file"), !file.data.isEmpty
        else {
            return .fail(
                Self.error(
                    status: .badRequest,
                    message: "missing required 'file' part",
                    type: "invalid_request_error", code: "missing_file"))
        }
        return .ok((form, file))
    }

    private func handleTranscriptions(_ request: Request) async -> Response
    {
        let t0 = Date()
        let upload = await extractUploadFile(request, cap: maxAudioUploadBytes)
        guard case .ok(let (form, file)) = upload else { return upload.orFail }

        do {
            // M43.2: non-blocking cold-load (see /v1/embeddings).
            switch try await governor.awaitLoad(.transcription, within: coldLoadWaitSecs) {
            case .loaded: break
            case .loading: return Self.coldLoadResponse(.transcription)
            }
            // M41.3: a `model` form field selects among the operator-
            // declared whisper allowlist; an unknown id ⇒ 400
            // model_not_available via the classified path. M41.4: an
            // actual rebind is audited (`model.rebind` trigger=inference).
            if let m = form.text("model"), !m.isEmpty {
                try await auditedRebind(
                    request, module: .transcription, target: m)
            }
        } catch {
            // issue #6: classify (see textEmbedding handler) — no leak, right code.
            return Self.classified(error, module: .transcription)
        }

        // Word timestamps are an opt-in of verbose_json only
        // (`timestamp_granularities[]=word`); every other format is
        // byte-unchanged and never triggers the alignment pass.
        let wantWords =
            form.text("response_format") == "verbose_json"
            && form.texts("timestamp_granularities[]").contains("word")

        let result: TranscriptionResult
        do {
            result = try await transcription.transcribe(
                audio: file.data, filename: file.filename,
                language: form.text("language"),
                wordTimestamps: wantWords)
        } catch {
            return Self.classified(error, module: .transcription)
        }
        // M56 — per-request summary (the format serialization below is
        // cheap; this captures the transcription work).
        Logger(label: AthenaLogLabel.model(.transcription)).notice(
            """
            transcription done segments=\(result.segments.count) \
            audio_secs=\(String(format: "%.1f", result.duration)) \
            lang=\(result.language ?? "auto") words=\(wantWords) \
            elapsed_ms=\(Self.elapsedMs(t0))
            """)

        func plain(_ s: String, _ type: String) -> Response {
            var headers = HTTPFields()
            headers[.contentType] = type
            var buf = ByteBuffer()
            buf.writeString(s)
            return Response(
                status: .ok, headers: headers,
                body: ResponseBody(byteBuffer: buf))
        }

        switch form.text("response_format") {
        case "text":
            return plain(result.text, "text/plain; charset=utf-8")
        case "srt":
            return plain(
                TranscriptionFormat.srt(result.segments),
                "text/plain; charset=utf-8")
        case "vtt":
            return plain(
                TranscriptionFormat.vtt(result.segments),
                "text/vtt; charset=utf-8")
        case "verbose_json":
            // Opt-in diarization: tag each Whisper segment with the
            // best-overlapping Sortformer speaker turn (M4.3c).
            var turns: [DiarizationTurn] = []
            if form.text("diarize") == "true" {
                switch await diarizeTurns(
                    audio: file.data, filename: file.filename)
                {
                case .fail(let r): return r
                case .ok(let t): turns = t
                }
            }
            func words(_ ws: [WordTiming]?) -> [WordTimestamp]? {
                guard let ws else { return nil }
                return ws.map {
                    WordTimestamp(
                        word: $0.word, start: $0.start, end: $0.end,
                        probability: $0.probability)
                }
            }
            return Self.json(
                VerboseTranscriptionResponse(
                    task: "transcribe", language: result.language,
                    duration: result.duration, text: result.text,
                    segments: result.segments.enumerated().map {
                        VerboseSegment(
                            id: $0.offset, start: $0.element.start,
                            end: $0.element.end, text: $0.element.text,
                            avg_logprob: $0.element.avgLogprob,
                            speaker: turns.isEmpty
                                ? nil
                                : Self.speaker(
                                    start: $0.element.start,
                                    end: $0.element.end, turns: turns),
                            words: words($0.element.words))
                    },
                    words: wantWords ? words(result.words) : nil))
        case "diarized_json":
            // ADR 013 #3 (#4a): OpenAI's diarized transcription format. Unlike
            // verbose_json's opt-in `diarize=true`, diarized_json IMPLIES
            // diarization — always run it and label every segment. Marked
            // **native-flavored** per the `/v1` compatibility rule: it reuses
            // the verbose envelope for consumer convenience, but `speaker` is
            // Athena's Int id (not OpenAI's string label) and word timings are
            // not emitted. The standalone /v1/audio/diarizations route stays
            // the canonical diarization surface.
            let dturns: [DiarizationTurn]
            switch await diarizeTurns(
                audio: file.data, filename: file.filename)
            {
            case .fail(let r): return r
            case .ok(let t): dturns = t
            }
            return Self.json(
                VerboseTranscriptionResponse(
                    task: "transcribe", language: result.language,
                    duration: result.duration, text: result.text,
                    segments: result.segments.enumerated().map {
                        VerboseSegment(
                            id: $0.offset, start: $0.element.start,
                            end: $0.element.end, text: $0.element.text,
                            avg_logprob: $0.element.avgLogprob,
                            speaker: Self.speaker(
                                start: $0.element.start,
                                end: $0.element.end, turns: dturns),
                            words: nil)
                    },
                    words: nil))
        default:  // "json" / nil
            return Self.json(TranscriptionResponse(text: result.text))
        }
    }

    /// Run end-to-end (Sortformer) diarization to tag transcription segments
    /// with speaker turns — shared by `verbose_json` (opt-in `diarize=true`)
    /// and `diarized_json` (implicit). The diarization slot is a single global
    /// tenant (ADR 011); if a pyannote *segmentation* model is resident (e.g. a
    /// prior `method=pyannote` request rebound it), this can't run and returns a
    /// clear 409 rather than a misleading message (the canonical diarization
    /// surface is the standalone /v1/audio/diarizations route, ADR 013).
    private func diarizeTurns(
        audio: Data, filename: String?
    ) async -> Outcome<[DiarizationTurn]> {
        do {
            // M43.2: non-blocking cold-load.
            switch try await governor.awaitLoad(
                .diarization, within: coldLoadWaitSecs)
            {
            case .loaded: break
            case .loading: return .fail(Self.coldLoadResponse(.diarization))
            }
            guard await diarization.residentBackend() == .sortformer else {
                return .fail(
                    Self.error(
                        status: .conflict,
                        message: "diarization needs an end-to-end "
                            + "(Sortformer) model, but a segmentation model is "
                            + "currently resident in the single diarization "
                            + "slot. Diarize separately via POST "
                            + "/v1/audio/diarizations, or select a Sortformer "
                            + "diarization model.",
                        type: "invalid_request_error",
                        code: "diarization_backend_conflict"))
            }
            return .ok(
                try await diarization.diarize(
                    audio: audio, filename: filename
                ).turns)
        } catch {
            return .fail(Self.classified(error, module: .diarization))
        }
    }

    /// The speaker whose turn most overlaps `[start,end]`, or nil if
    /// none overlap (M4.3c Sortformer↔Whisper alignment).
    private static func speaker(
        start: Double, end: Double, turns: [DiarizationTurn]
    ) -> Int? {
        var best: (speaker: Int, overlap: Double)?
        for t in turns {
            let ov = min(end, t.end) - max(start, t.start)
            if ov > 0, best == nil || ov > best!.overlap {
                best = (t.speaker, ov)
            }
        }
        return best?.speaker
    }

    /// ADR 022 M78.1 — `POST /v1/video/transcriptions` (Athena-native, NOT
    /// OpenAI). Demux the audio track out of the uploaded video and transcribe
    /// it via the same Whisper/Parakeet tenant; the response shapes mirror
    /// `/v1/audio/transcriptions` so an existing transcription consumer reuses
    /// its parser. Bounded by `maxVideoUploadBytes`. (`diarize=true` on video is
    /// not yet wired — a 501; transcribe, then diarize the extracted audio via
    /// `/v1/audio/diarizations`.)
    private func handleVideoTranscriptions(_ request: Request) async -> Response
    {
        let t0 = Date()
        let upload = await extractUploadFile(request, cap: maxVideoUploadBytes)
        guard case .ok(let (form, file)) = upload else { return upload.orFail }

        // diarization on video is a 501 (not yet wired) — fail fast before any
        // load/decode so the caller gets the clear answer immediately. Both the
        // `diarize=true` flag and the `diarized_json` format (which implies
        // diarization, ADR 013 #3) take this path.
        if form.text("diarize") == "true"
            || form.text("response_format") == "diarized_json"
        {
            return Self.classified(
                AthenaError.notImplemented(
                    feature: "diarization on /v1/video/transcriptions "
                        + "(diarize=true or response_format=diarized_json) — "
                        + "transcribe, then POST the extracted audio to "
                        + "/v1/audio/diarizations"),
                module: .transcription)
        }

        do {
            switch try await governor.awaitLoad(
                .transcription, within: coldLoadWaitSecs)
            {
            case .loaded: break
            case .loading: return Self.coldLoadResponse(.transcription)
            }
            if let m = form.text("model"), !m.isEmpty {
                try await auditedRebind(
                    request, module: .transcription, target: m)
            }
        } catch {
            return Self.classified(error, module: .transcription)
        }

        let wantWords =
            form.text("response_format") == "verbose_json"
            && form.texts("timestamp_granularities[]").contains("word")

        // Demux the audio track to PCM straight from the in-memory upload bytes
        // (Option D, ADR 025 S5 — no temp file), then reuse the shared
        // transcribePCM seam (S2). The extracted PCM funnels through the same
        // floor/ceiling — a degenerate video is a 4xx here.
        let result: TranscriptionResult
        do {
            var pcm = try await VideoAudioTrack.extractPCM(
                from: file.data, filename: file.filename, module: .transcription)
            defer { ProcessHardening.secureZero(&pcm) }  // ADR 024 T2
            result = try await transcription.transcribePCM(
                pcm, language: form.text("language"),
                wordTimestamps: wantWords)
        } catch {
            return Self.classified(error, module: .transcription)
        }
        Logger(label: AthenaLogLabel.model(.transcription)).notice(
            """
            video transcription done segments=\(result.segments.count) \
            audio_secs=\(String(format: "%.1f", result.duration)) \
            lang=\(result.language ?? "auto") words=\(wantWords) \
            elapsed_ms=\(Self.elapsedMs(t0))
            """)

        func plain(_ s: String, _ type: String) -> Response {
            var headers = HTTPFields()
            headers[.contentType] = type
            var buf = ByteBuffer()
            buf.writeString(s)
            return Response(
                status: .ok, headers: headers,
                body: ResponseBody(byteBuffer: buf))
        }
        func words(_ ws: [WordTiming]?) -> [WordTimestamp]? {
            ws.map {
                $0.map {
                    WordTimestamp(
                        word: $0.word, start: $0.start, end: $0.end,
                        probability: $0.probability)
                }
            }
        }
        switch form.text("response_format") {
        case "text":
            return plain(result.text, "text/plain; charset=utf-8")
        case "srt":
            return plain(
                TranscriptionFormat.srt(result.segments),
                "text/plain; charset=utf-8")
        case "vtt":
            return plain(
                TranscriptionFormat.vtt(result.segments),
                "text/vtt; charset=utf-8")
        case "verbose_json":
            return Self.json(
                VerboseTranscriptionResponse(
                    task: "transcribe", language: result.language,
                    duration: result.duration, text: result.text,
                    segments: result.segments.enumerated().map {
                        VerboseSegment(
                            id: $0.offset, start: $0.element.start,
                            end: $0.element.end, text: $0.element.text,
                            avg_logprob: $0.element.avgLogprob,
                            speaker: nil, words: words($0.element.words))
                    },
                    words: wantWords ? words(result.words) : nil))
        default:  // "json" / nil
            return Self.json(TranscriptionResponse(text: result.text))
        }
    }

    private func handleDiarizations(_ request: Request) async -> Response
    {
        let t0 = Date()
        let upload = await extractUploadFile(request, cap: maxAudioUploadBytes)
        guard case .ok(let (form, file)) = upload else { return upload.orFail }

        // Method select (ADR 018). Default `sortformer` (fast, end-to-end,
        // ≤4 speakers); `cluster` = naive-window embedding cluster (M25.3,
        // >4, no overlap); `pyannote` = learned segmentation + embed + global
        // cluster (overlap-aware, arbitrary speakers). `model` selects weights
        // within the chosen method's family.
        switch (form.text("method") ?? "").lowercased() {
        case "", "sortformer":
            break  // fall through to the Sortformer path below
        case "cluster":
            return await handleClusterDiarization(file: file, form: form)
        case "pyannote":
            return await handlePyannoteDiarization(
                request: request, file: file, form: form)
        case let other:
            return Self.classified(
                AthenaError.diarizationMethodInvalid(
                    method: other,
                    reason: "unknown method — use sortformer, cluster, "
                        + "or pyannote"),
                module: .diarization)
        }

        do {
            // M43.2: non-blocking cold-load.
            switch try await governor.awaitLoad(.diarization, within: coldLoadWaitSecs) {
            case .loaded: break
            case .loading: return Self.coldLoadResponse(.diarization)
            }
            // M41.3 per-request diarization model selection;
            // M41.4 audited on a real resident-id change.
            if let m = form.text("model"), !m.isEmpty {
                try await auditedRebind(
                    request, module: .diarization, target: m)
            }
        } catch {
            // issue #6: classify (see textEmbedding handler) — no leak, right code.
            return Self.classified(error, module: .diarization)
        }

        let r: DiarizationResult
        do {
            r = try await diarization.diarize(
                audio: file.data, filename: file.filename)
        } catch {
            return Self.classified(error, module: .diarization)
        }
        // M56 — per-request summary.
        Logger(label: AthenaLogLabel.model(.diarization)).notice(
            """
            diarization done method=sortformer \
            speakers=\(r.numSpeakers) turns=\(r.turns.count) \
            elapsed_ms=\(Self.elapsedMs(t0))
            """)
        return Self.json(
            DiarizationResponse(
                num_speakers: r.numSpeakers,
                segments: r.turns.map {
                    DiarizationSegmentDTO(
                        start: $0.start, end: $0.end,
                        speaker: $0.speaker)
                }))
    }

    /// ADR 018 — pyannote pipeline: learned PyanNet segmentation → WeSpeaker
    /// embed each locally-active region → GLOBAL agglomerative cluster
    /// (same-window cannot-link) → overlap-aware turns with file-stable speaker
    /// ids. The overlap-aware path the naive `cluster` method cannot produce.
    /// The resident diarization model must be a pyannote segmentation
    /// checkpoint (select via `model=`); `num_speakers`/`min_speakers`/
    /// `max_speakers`/`threshold` tune the global clustering.
    private func handlePyannoteDiarization(
        request: Request, file: MultipartForm.Part, form: MultipartForm
    ) async -> Response {
        let t0 = Date()
        let numSpeakers = form.text("num_speakers").flatMap(Int.init)
        let minSpeakers = form.text("min_speakers").flatMap(Int.init)
        let maxSpeakers = form.text("max_speakers").flatMap(Int.init)
        let threshold = form.text("threshold").flatMap(Float.init) ?? 0.75

        do {
            // Cold-load the segmentation slot + optional per-request model
            // selection, same as the Sortformer path (shared 100 MiB cap +
            // cold-load 503 behavior).
            switch try await governor.awaitLoad(
                .diarization, within: coldLoadWaitSecs)
            {
            case .loaded: break
            case .loading: return Self.coldLoadResponse(.diarization)
            }
            if let m = form.text("model"), !m.isEmpty {
                try await auditedRebind(
                    request, module: .diarization, target: m)
            }
            // The pyannote pipeline also needs the speaker-embedding model.
            switch try await governor.awaitLoad(
                .speakerEmbedding, within: coldLoadWaitSecs)
            {
            case .loaded: break
            case .loading: return Self.coldLoadResponse(.speakerEmbedding)
            }
        } catch {
            return Self.classified(error, module: .diarization)
        }

        // Method/model match: pyannote requires a segmentation-backed model.
        let backend = await diarization.residentBackend()
        guard backend == .pyannoteSegmentation else {
            return Self.classified(
                AthenaError.diarizationMethodInvalid(
                    method: "pyannote",
                    reason: "the resident diarization model is not a "
                        + "segmentation model — select one with `model=` "
                        + "(operator must pull a pyannote-segmentation model "
                        + "into the diarization allowlist)"),
                module: .diarization)
        }

        // 1. Learned segmentation → per-window locally-tagged regions.
        let regions: [SpeakerActivityRegion]
        do {
            regions = try await diarization.segment(
                audio: file.data, filename: file.filename)
        } catch {
            return Self.classified(error, module: .diarization)
        }
        if regions.isEmpty {
            return Self.json(
                DiarizationResponse(num_speakers: 0, segments: []))
        }

        // 2. Embed each region with WeSpeaker (256-d, stable model — ADR 018).
        let embResult: SpeakerEmbeddingResult
        do {
            embResult = try await speakerEmbedding.embed(
                audio: file.data, filename: file.filename,
                segments: regions.map {
                    SpeakerSegmentRequest(start: $0.start, end: $0.end)
                })
        } catch {
            return Self.classified(error, module: .speakerEmbedding)
        }
        guard embResult.segments.count == regions.count else {
            return Self.classified(
                AthenaError.moduleLoadFailed(
                    .speakerEmbedding,
                    reason: "embedding count \(embResult.segments.count) "
                        + "≠ region count \(regions.count)"),
                module: .speakerEmbedding)
        }

        // 3. GLOBAL cluster with same-window cannot-link → file-stable ids.
        let embeddings = embResult.segments.map { $0.embedding }
        var labels = AgglomerativeClustering.cluster(
            embeddings,
            numClusters: numSpeakers, threshold: threshold,
            maxClusters: maxSpeakers, minClusters: minSpeakers ?? 1,
            cannotLink: PyannoteSegmentationDecode.sameWindowCannotLink(regions))

        let regionDurations = regions.map { $0.end - $0.start }
        if let target = numSpeakers {
            // Exact count: the agglomerative cut can stick *above* the target
            // because same-window cannot-link forbids the final merges, so
            // force exactly N (override the constraint — the user asked for N).
            labels = PyannoteSegmentationDecode.reduceToTargetClusters(
                embeddings: embeddings, labels: labels,
                durations: regionDurations, target: target)
        } else {
            // Auto mode: dissolve tiny clusters into real speakers (pyannote
            // min_cluster_size) so noisy short/overlap embeddings on long messy
            // audio don't inflate the count.
            let minClusterSeconds =
                form.text("min_cluster_seconds").flatMap(Double.init) ?? 6.0
            labels = PyannoteSegmentationDecode.reassignSmallClusters(
                embeddings: embeddings, labels: labels,
                durations: regionDurations, minDuration: minClusterSeconds)
        }

        // 4. Overlap-aware turns: one turn per region at its global id, then
        //    merge each speaker's overlapping/adjacent turns (cross-speaker
        //    overlap is preserved).
        let rawTurns = zip(regions, labels).map {
            DiarizationTurn(start: $0.start, end: $0.end, speaker: $1)
        }
        let turns = PyannoteSegmentationDecode.mergeSameSpeakerTurns(rawTurns)
        let speakers = Set(labels).count

        Logger(label: AthenaLogLabel.model(.diarization)).notice(
            """
            diarization done method=pyannote \
            speakers=\(speakers) regions=\(regions.count) \
            turns=\(turns.count) elapsed_ms=\(Self.elapsedMs(t0))
            """)
        return Self.json(
            DiarizationResponse(
                num_speakers: speakers,
                segments: turns.map {
                    DiarizationSegmentDTO(
                        start: $0.start, end: $0.end, speaker: $0.speaker)
                }))
    }

    /// M25.3 embedding+clustering diarizer: window → WeSpeaker embed →
    /// agglomerative cluster → merge same-speaker windows into turns.
    /// Recovers >4 speakers, which the offline Sortformer cannot.
    private func handleClusterDiarization(
        file: MultipartForm.Part, form: MultipartForm
    ) async -> Response {
        let t0 = Date()
        let numSpeakers = form.text("num_speakers").flatMap(Int.init)
        let maxSpeakers = form.text("max_speakers").flatMap(Int.init)
        let threshold = form.text("threshold").flatMap(Float.init) ?? 0.75

        do {
            // M43.2: non-blocking cold-load.
            switch try await governor.awaitLoad(.speakerEmbedding, within: coldLoadWaitSecs) {
            case .loaded: break
            case .loading: return Self.coldLoadResponse(.speakerEmbedding)
            }
        } catch {
            // issue #6: classify (see textEmbedding handler) — no leak, right code.
            return Self.classified(error, module: .speakerEmbedding)
        }

        let we: SpeakerEmbeddingResult
        do {
            we = try await speakerEmbedding.windowEmbeddings(
                audio: file.data, filename: file.filename,
                windowSeconds: 1.5, hopSeconds: 0.75)
        } catch {
            return Self.classified(error, module: .speakerEmbedding)
        }

        let labels = AgglomerativeClustering.cluster(
            we.segments.map { $0.embedding },
            numClusters: numSpeakers, threshold: threshold,
            maxClusters: maxSpeakers)
        let turns = Self.turnsFromWindows(we.segments, labels: labels)
        // M56 — per-request summary.
        Logger(label: AthenaLogLabel.model(.speakerEmbedding)).notice(
            """
            diarization done method=cluster \
            speakers=\(Set(labels).count) windows=\(we.segments.count) \
            turns=\(turns.count) elapsed_ms=\(Self.elapsedMs(t0))
            """)
        return Self.json(
            DiarizationResponse(
                num_speakers: Set(labels).count,
                segments: turns.map {
                    DiarizationSegmentDTO(
                        start: $0.start, end: $0.end, speaker: $0.speaker)
                }))
    }

    /// Merge time-ordered, possibly-overlapping labeled windows into
    /// contiguous same-speaker turns.
    private static func turnsFromWindows(
        _ segs: [SpeakerSegmentEmbedding], labels: [Int]
    ) -> [DiarizationTurn] {
        guard !segs.isEmpty, segs.count == labels.count else { return [] }
        var turns: [DiarizationTurn] = []
        var curLabel = labels[0]
        var curStart = segs[0].start
        var curEnd = segs[0].end
        for i in 1..<segs.count {
            if labels[i] == curLabel {
                curEnd = max(curEnd, segs[i].end)
            } else {
                turns.append(
                    DiarizationTurn(
                        start: curStart, end: curEnd, speaker: curLabel))
                curLabel = labels[i]
                curStart = segs[i].start
                curEnd = segs[i].end
            }
        }
        turns.append(
            DiarizationTurn(
                start: curStart, end: curEnd, speaker: curLabel))
        return turns
    }

    /// M25.2 — voice/speaker embeddings. Multipart `file` (audio) + an
    /// optional `segments` JSON field (`[{start,end}]`, seconds); absent
    /// ⇒ the whole clip is embedded as one segment. Returns one 256-d
    /// L2-normalized vector per segment for cross-recording speaker ID.
    private func handleSpeakerEmbeddings(_ request: Request) async
        -> Response
    {
        let t0 = Date()
        let upload = await extractUploadFile(request, cap: maxAudioUploadBytes)
        guard case .ok(let (form, file)) = upload else { return upload.orFail }

        // Optional `segments` JSON; absent ⇒ embed the whole clip.
        var segments: [SpeakerSegmentRequest] = []
        if let segText = form.text("segments"), !segText.isEmpty {
            do {
                let specs = try JSONDecoder().decode(
                    [SpeakerSegmentSpec].self,
                    from: Data(segText.utf8))
                segments = specs.map {
                    SpeakerSegmentRequest(start: $0.start, end: $0.end)
                }
            } catch {
                return Self.error(
                    status: .badRequest,
                    message: "invalid 'segments' JSON: \(error)",
                    type: "invalid_request_error",
                    code: "invalid_segments")
            }
        }

        do {
            // M43.2: non-blocking cold-load.
            switch try await governor.awaitLoad(.speakerEmbedding, within: coldLoadWaitSecs) {
            case .loaded: break
            case .loading: return Self.coldLoadResponse(.speakerEmbedding)
            }
            // M41.3 per-request speaker-embedding model selection;
            // M41.4 audited on a real resident-id change.
            if let m = form.text("model"), !m.isEmpty {
                try await auditedRebind(
                    request, module: .speakerEmbedding, target: m)
            }
        } catch {
            // issue #6: classify (see textEmbedding handler) — no leak, right code.
            return Self.classified(error, module: .speakerEmbedding)
        }

        let result: SpeakerEmbeddingResult
        do {
            result = try await speakerEmbedding.embed(
                audio: file.data, filename: file.filename,
                segments: segments)
        } catch {
            return Self.classified(error, module: .speakerEmbedding)
        }

        // M41.3: response.model echoes the id ACTUALLY served (post-
        // rebind), not the form-field passthrough — same truthful-echo
        // discipline as the M39 embedding batch.
        let selSpeaker = selectable(.speakerEmbedding)
        let resSpeaker = await selSpeaker.residentModelId()
        let defSpeaker = await selSpeaker.defaultModelId()
        let servedSpeaker = resSpeaker ?? defSpeaker
        // M56 — per-request summary.
        Logger(label: AthenaLogLabel.model(.speakerEmbedding)).notice(
            """
            speaker-embeddings done model=\(servedSpeaker) \
            segments=\(result.segments.count) \
            elapsed_ms=\(Self.elapsedMs(t0))
            """)
        return Self.json(
            SpeakerEmbeddingResponse(
                object: "list",
                data: result.segments.enumerated().map { idx, s in
                    SpeakerEmbeddingObject(
                        object: "speaker_embedding", index: idx,
                        segment: SpeakerSegmentSpec(
                            start: s.start, end: s.end),
                        embedding: s.embedding,
                        duration_seconds: s.durationSeconds)
                },
                model: servedSpeaker,
                dimension: result.dimension))
    }

    // MARK: - Principal resolution & metering (M12.6 / M27)

    /// Resolve the caller's principal + admin/enforced flags from the
    /// AuthMiddleware-published resolution. `enforced` is auth being on;
    /// `principal` identifies the bearer's owning subject (`u:<user>` for a
    /// managed token, `t:<hash>` for a bootstrap key). `isAdmin` = the
    /// caller holds the full permission set (the `admin` role). Used for
    /// per-principal usage metering + the `/api/usage` admin/own scoping.
    private func queuePrincipal(_ request: Request) async -> (
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

    /// The principal a (non-queue) request should be metered against:
    /// the authenticated subject, or nil when auth is off (mapped to
    /// `xenos` by `meter`). Inference handlers only run after the auth
    /// middleware admits the request, so an enabled-auth request always
    /// resolves a real principal here.
    private func usagePrincipal(_ request: Request) async -> String? {
        await queuePrincipal(request).principal
    }

    /// Record one request's token usage (M27): bump the global metrics
    /// counter and the persisted per-principal counter. nil principal ⇒
    /// auth off ⇒ the `xenos` sentinel. Persistence failures are
    /// non-fatal — metering must never break inference.
    private func meter(principal: String?, usage: TokenUsage) async {
        await metrics.addTokens(usage.totalTokens)
        try? await store.addUsage(
            principal: principal ?? Self.xenos,
            promptTokens: usage.promptTokens,
            completionTokens: usage.completionTokens)
    }

    /// `GET /api/usage` (M27.3). An admin (or the single-tenant
    /// loopback operator when auth is off) sees every principal's
    /// counters; any other authenticated caller sees only its own row.
    private func handleUsage(_ request: Request) async -> Response {
        let who = await queuePrincipal(request)
        let rows: [UsageRow]
        if !who.enforced || who.isAdmin {
            rows = await store.allUsage()
        } else if let p = who.principal, let r = await store.usage(
            principal: p)
        {
            rows = [r]
        } else {
            rows = []
        }
        return Self.json(
            UsageReportResponse(
                usage: rows.map {
                    UsageEntryDTO(
                        principal: $0.principal, requests: $0.requests,
                        prompt_tokens: $0.promptTokens,
                        completion_tokens: $0.completionTokens,
                        total_tokens: $0.totalTokens,
                        updated: $0.updated)
                }))
    }

    /// `GET /api/audit` (M30.2). Admin-only (AuthPolicy →
    /// `daemon.admin`), so no owner-scoping is needed — the trail is a
    /// privileged oversight view. Filterable by `principal`, `action`,
    /// `since` (epoch seconds) and `limit` (default 100, capped 1000);
    /// rows come back most-recent-first. Pull only — the passive
    /// oracle never pushes the audit trail out.
    private func handleAudit(_ request: Request) async -> Response {
        var principal: String?
        var action: String?
        var since: Double?
        var limit = 100
        if let q = request.uri.query {
            for kv in q.split(separator: "&") {
                let p = kv.split(separator: "=", maxSplits: 1)
                guard let k = p.first.map(String.init) else {
                    continue
                }
                let raw = p.count == 2 ? String(p[1]) : ""
                let v = raw.removingPercentEncoding ?? raw
                switch k {
                case "principal": principal = v.isEmpty ? nil : v
                case "action": action = v.isEmpty ? nil : v
                case "since": since = Double(v)
                case "limit":
                    limit = min(1000, max(1, Int(v) ?? 100))
                default: break
                }
            }
        }
        let rows = await store.listAudit(
            principal: principal, action: action, since: since,
            limit: limit)
        return Self.json(
            AuditReportResponse(
                audit: rows.map {
                    AuditEntryDTO(
                        id: $0.id, ts: $0.ts,
                        principal: $0.principal, action: $0.action,
                        target: $0.target, result: $0.result,
                        detail: $0.detail)
                }))
    }

    /// `GET /api/cache/prompt` (M59.4). Admin-only stats for the
    /// cross-request prompt-prefix KV pool: whether it's enabled, the live
    /// entry/byte occupancy + caps, and cumulative hit/miss/eviction
    /// counters. Read-only ⇒ not audited (mirrors `/api/audit` itself).
    private func handlePromptCacheStats() -> Response {
        guard let prefixCache else {
            return Self.json(PromptCacheStatsResponse.disabled)
        }
        let s = prefixCache.stats()
        return Self.json(
            PromptCacheStatsResponse(
                enabled: true, entries: s.entries, bytes: s.bytes,
                hits: s.hits, misses: s.misses, evictions: s.evictions,
                max_entries: s.maxEntries, max_bytes: s.maxBytes))
    }

    /// `DELETE /api/cache/prompt` (M59.4). Admin-only flush of the pool —
    /// drops every entry NOT currently held by an in-flight generation
    /// (those are freed when their request releases them). Audited at the
    /// shared chokepoint (M30). Returns how many entries were freed plus the
    /// post-flush occupancy.
    private func handlePromptCacheFlush(_ request: Request) async -> Response {
        guard let prefixCache else {
            await audit(
                request, action: "prompt_cache.flush", target: nil,
                result: "ok", detail: "disabled")
            return Self.json(PromptCacheFlushResponse(flushed: 0, entries: 0, bytes: 0))
        }
        let freed = prefixCache.flushIdle()
        let s = prefixCache.stats()
        await audit(
            request, action: "prompt_cache.flush", target: nil,
            result: "ok", detail: "freed=\(freed) remaining=\(s.entries)")
        return Self.json(
            PromptCacheFlushResponse(
                flushed: freed, entries: s.entries, bytes: s.bytes))
    }

    /// `GET /api/logs` (M45.5). Admin-only daemon-log oversight,
    /// projected from `/usr/bin/log show --style ndjson` filtered to
    /// `subsystem == "athena"`. Query params:
    ///   - `since=<dur>` (default 1h; passed to `log show --last`)
    ///   - `category=<X>` (repeatable; AND'd into the predicate)
    ///   - `debug=1` (include info+debug entries; default notice+)
    ///   - `limit=<n>` (default 200, capped 5000)
    /// Pull-only JSON; the SSE sibling at `/api/logs/stream` follows
    /// new entries live.
    private func handleLogs(_ request: Request) async -> Response {
        var since = "1h"
        var categories: [String] = []
        var debug = false
        var limit = 200
        if let q = request.uri.query {
            for kv in q.split(separator: "&") {
                let p = kv.split(separator: "=", maxSplits: 1)
                guard let k = p.first.map(String.init) else { continue }
                let raw = p.count == 2 ? String(p[1]) : ""
                let v = raw.removingPercentEncoding ?? raw
                switch k {
                case "since": if !v.isEmpty { since = v }
                case "category": if !v.isEmpty { categories.append(v) }
                case "debug": debug = (v == "1" || v == "true")
                case "limit":
                    limit = min(5000, max(1, Int(v) ?? 200))
                default: break
                }
            }
        }
        // Build predicate the same way `athena logs` does — pre-baked
        // subsystem, optional categories.
        guard let predicate = Self.logPredicate(categories: categories)
        else {
            return Self.error(
                status: .badRequest,
                message: "invalid category (must be alphanum + dot)",
                type: "invalid_request_error",
                code: "invalid_category")
        }
        var args = [
            "show", "--style", "ndjson", "--last", since,
            "--predicate", predicate,
        ]
        if debug { args += ["--info", "--debug"] }
        let entries = (try? await Self.collectLogEntries(
            args: args, limit: limit)) ?? []
        return Self.json(LogsReportResponse(logs: entries))
    }

    /// `GET /api/logs/stream` (M45.5). Admin-only SSE following
    /// `log stream --style ndjson` filtered to subsystem athena.
    /// Each event line is `data: {<LogEntryDTO JSON>}\n\n`. Capped at
    /// ~10 min so a dropped client doesn't pin a process forever; the
    /// subprocess is terminated on `continuation.onTermination`.
    private func handleLogsStream(_ request: Request) async -> Response {
        var categories: [String] = []
        var debug = false
        if let q = request.uri.query {
            for kv in q.split(separator: "&") {
                let p = kv.split(separator: "=", maxSplits: 1)
                guard let k = p.first.map(String.init) else { continue }
                let raw = p.count == 2 ? String(p[1]) : ""
                let v = raw.removingPercentEncoding ?? raw
                switch k {
                case "category": if !v.isEmpty { categories.append(v) }
                case "debug": debug = (v == "1" || v == "true")
                default: break
                }
            }
        }
        guard let predicate = Self.logPredicate(categories: categories)
        else {
            return Self.error(
                status: .badRequest,
                message: "invalid category (must be alphanum + dot)",
                type: "invalid_request_error",
                code: "invalid_category")
        }
        var args = [
            "stream", "--style", "ndjson", "--predicate", predicate,
        ]
        if debug { args += ["--info", "--debug"] }
        let stream = AsyncStream<ByteBuffer> { continuation in
            let proc = Process()
            proc.executableURL = URL(
                fileURLWithPath: "/usr/bin/log")
            proc.arguments = args
            let pipe = Pipe()
            proc.standardOutput = pipe
            proc.standardError = FileHandle.nullDevice
            let task = Task.detached {
                let h = pipe.fileHandleForReading
                var buffer = Data()
                let deadline = Date().addingTimeInterval(600)
                while Date() < deadline {
                    if Task.isCancelled { break }
                    let chunk = h.availableData
                    if chunk.isEmpty {
                        // Process exited — bail.
                        break
                    }
                    buffer.append(chunk)
                    while let nl = buffer.firstIndex(of: 0x0A) {
                        let line = buffer.subdata(
                            in: buffer.startIndex..<nl)
                        buffer = buffer.subdata(
                            in: (nl + 1)..<buffer.endIndex)
                        if let entry = Self.parseNDJSONLogLine(line),
                            let json = try? JSONEncoder().encode(entry)
                        {
                            var b = ByteBuffer()
                            b.writeString("data: ")
                            b.writeBytes(json)
                            b.writeString("\n\n")
                            continuation.yield(b)
                        }
                    }
                }
                if proc.isRunning { proc.terminate() }
                continuation.finish()
            }
            do { try proc.run() } catch {
                task.cancel()
                continuation.finish()
                return
            }
            continuation.onTermination = { _ in
                task.cancel()
                if proc.isRunning { proc.terminate() }
            }
        }
        var headers = HTTPFields()
        headers[.contentType] = "text/event-stream"
        headers[.cacheControl] = "no-cache"
        return Response(
            status: .ok, headers: headers,
            body: ResponseBody(asyncSequence: stream))
    }

    /// Build the `log` predicate: subsystem pin + optional category
    /// AND-filter. Returns nil if any category contains characters
    /// outside `[A-Za-z0-9._-]` (defensive — predicate-injection
    /// guard since the value is interpolated into a quoted string).
    private static func logPredicate(categories: [String]) -> String? {
        let safe = CharacterSet(
            charactersIn:
                "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
        )
        for c in categories {
            guard !c.isEmpty,
                c.unicodeScalars.allSatisfy({ safe.contains($0) })
            else { return nil }
        }
        var p = "subsystem == \"athena\""
        if !categories.isEmpty {
            let list = categories
                .map { "\"\($0)\"" }
                .joined(separator: ", ")
            p += " AND category IN { \(list) }"
        }
        return p
    }

    /// Run `/usr/bin/log` with the supplied args, parse each ndjson
    /// line into a LogEntryDTO, return up to `limit` entries. The
    /// process is killed on timeout (5s) so a hung `log` invocation
    /// can't pin a request thread.
    private static func collectLogEntries(
        args: [String], limit: Int
    ) async throws -> [LogEntryDTO] {
        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: "/usr/bin/log")
        proc.arguments = args
        let pipe = Pipe()
        proc.standardOutput = pipe
        proc.standardError = FileHandle.nullDevice
        try proc.run()
        let deadline = Date().addingTimeInterval(5)
        // Read all available data; bound by deadline + limit.
        var buffer = Data()
        var entries: [LogEntryDTO] = []
        let h = pipe.fileHandleForReading
        while Date() < deadline && entries.count < limit {
            let chunk = h.availableData
            if chunk.isEmpty {
                if !proc.isRunning { break }
                try await Task.sleep(nanoseconds: 50_000_000)
                continue
            }
            buffer.append(chunk)
            while let nl = buffer.firstIndex(of: 0x0A),
                entries.count < limit
            {
                let line = buffer.subdata(
                    in: buffer.startIndex..<nl)
                buffer = buffer.subdata(
                    in: (nl + 1)..<buffer.endIndex)
                if let entry = parseNDJSONLogLine(line) {
                    entries.append(entry)
                }
            }
        }
        if proc.isRunning { proc.terminate() }
        return entries
    }

    /// Decode one `log show --style ndjson` line into our compact
    /// LogEntryDTO. Returns nil for header rows / parse failures —
    /// best-effort projection, not strict decoding.
    private static func parseNDJSONLogLine(_ data: Data) -> LogEntryDTO?
    {
        guard !data.isEmpty,
            let obj = try? JSONSerialization.jsonObject(with: data)
                as? [String: Any]
        else { return nil }
        // `log show` emits a header row first; skip if there's no
        // eventMessage.
        guard let msg = obj["eventMessage"] as? String else {
            return nil
        }
        let ts = obj["timestamp"] as? String ?? ""
        let levelRaw = obj["messageType"] as? String ?? "default"
        let category = obj["category"] as? String ?? "daemon"
        return LogEntryDTO(
            ts: ts, level: levelRaw, category: category, message: msg)
    }

    // MARK: - Model lifecycle ops (ADR 025 S2 — synchronous + SSE)

    /// One SSE `data:` frame around an encoded JSON payload.
    private static func sseFrame(_ bytes: Data) -> ByteBuffer {
        var b = ByteBuffer()
        b.writeString("data: ")
        b.writeBytes(bytes)
        b.writeString("\n\n")
        return b
    }

    /// Run a model store op to completion. `progress` (0…1) reports the
    /// download fraction where the op has one (pull/convert); prune has
    /// none. Returns the op-specific result JSON, or an error envelope
    /// detail (`{message,type,code}`). ADR 025 S2: model ops are
    /// synchronous now — no async queue, no job id, no persistence. Shared
    /// by the SSE CLI handlers and the blocking `/ui` console.
    private func performModelOp(
        kind: String, body: Data,
        progress: @escaping @Sendable (Double) -> Void = { _ in }
    ) async -> (result: Data?, error: APIErrorBody.ErrorDetail?) {
        switch kind {
        case "model_pull":
            // Egress (proxy + HF token) was exported process-wide at daemon
            // startup (Load); the #hubDownloader reads it. Only sanctioned
            // model-fetch egress — passive oracle intact.
            guard
                let req = try? JSONDecoder().decode(
                    ModelPullRequest.self, from: body), !req.id.isEmpty
            else {
                return (
                    nil,
                    .init(
                        message: "model_pull requires non-empty 'id'",
                        type: "invalid_request_error", code: "invalid_body"))
            }
            do {
                let dest = try await ModelPull.pull(
                    id: req.id, revision: req.revision,
                    into: modelStoreRoot, progress: progress)
                return (
                    try? JSONEncoder().encode(
                        ModelPullResult(
                            name: dest.lastPathComponent, path: dest.path)),
                    nil)
            } catch {
                return (
                    nil,
                    .init(
                        message: "pull failed: \(ModelPull.friendlyError(error))",
                        type: "server_error", code: "pull_failed"))
            }
        case "model_convert":
            guard
                let req = try? JSONDecoder().decode(
                    ModelConvertRequest.self, from: body), !req.id.isEmpty
            else {
                return (
                    nil,
                    .init(
                        message: "model_convert requires non-empty 'id'",
                        type: "invalid_request_error", code: "invalid_body"))
            }
            do {
                // M-conv: `bits` is opt-in (mlx_lm-style). Omit ⇒ no
                // quantization; explicit N ⇒ quantize to N-bit.
                // ADR 029 WP3 — convert loads weights + runs quantize kernels on
                // the shared Metal pool; gate it so it can't co-execute with a
                // decode (the "two uncoordinated allocators on one pool" hazard
                // ADR 011/029 forbid). A convert holds the gate for the whole
                // quantize (minutes) — deliberate: an operator convert blocks
                // inference for its duration rather than racing it on the device.
                let r = try await InferenceGate.shared.withExclusiveExecution {
                    try await ModelConvert.convert(
                        id: req.id, revision: req.revision, bits: req.bits,
                        groupSize: req.group_size ?? 64,
                        into: modelStoreRoot, name: req.name,
                        progress: progress)
                }
                return (
                    try? JSONEncoder().encode(
                        ModelConvertResult(path: r.path.path, bytes: r.bytes)),
                    nil)
            } catch let e as AthenaError {
                // Cause-naming convert errors (ADR 016 redirect / unsupported
                // class) carry an actionable message + stable code — surface
                // them verbatim instead of a raw substrate dump.
                return (
                    nil,
                    .init(message: e.message, type: e.type, code: e.code))
            } catch {
                return (
                    nil,
                    .init(
                        message: "convert failed: \(error)",
                        type: "server_error", code: "convert_failed"))
            }
        case "model_prune":
            let req =
                (try? JSONDecoder().decode(
                    ModelPruneRequest.self, from: body))
                ?? ModelPruneRequest(dry_run: false)
            do {
                let pr = try ModelStoreOps.prune(
                    root: modelStoreRoot, dryRun: req.dry_run ?? false)
                return (
                    try? JSONEncoder().encode(
                        ModelPruneResult(
                            candidates: pr.victims.map { $0.name },
                            removed: pr.removed, dry_run: pr.dryRun)),
                    nil)
            } catch {
                return (
                    nil,
                    .init(
                        message: "prune failed: \(error)",
                        type: "server_error", code: "prune_failed"))
            }
        default:
            return (
                nil,
                .init(
                    message: "unknown model op '\(kind)'",
                    type: "invalid_request_error", code: "invalid_kind"))
        }
    }

    /// Stream a model op's progress over SSE on the current request (ADR
    /// 025 S2). Emits `data: {"event":"progress","fraction":F}` frames
    /// during the download, `: keep-alive` comments during silent tails
    /// (e.g. the convert quantization phase, which has no HF progress), and
    /// a terminal `data: {"event":"done","result":…}` or
    /// `data: {"event":"error","error":…}` frame before `[DONE]`.
    private func streamModelOp(kind: String, body: Data) -> Response {
        let stream = AsyncStream<ByteBuffer> { continuation in
            // Keep-alive ticker so a long silent op doesn't look hung to a
            // client/proxy idle timeout.
            let keepAlive = Task {
                while !Task.isCancelled {
                    try? await Task.sleep(nanoseconds: 15_000_000_000)
                    if Task.isCancelled { return }
                    var b = ByteBuffer()
                    b.writeString(": keep-alive\n\n")
                    continuation.yield(b)
                }
            }
            let work = Task {
                let progress: @Sendable (Double) -> Void = { f in
                    if let d = try? JSONEncoder().encode(
                        ModelOpProgressEvent(event: "progress", fraction: f))
                    {
                        continuation.yield(Self.sseFrame(d))
                    }
                }
                let (result, error) = await performModelOp(
                    kind: kind, body: body, progress: progress)
                keepAlive.cancel()
                if let result {
                    var b = ByteBuffer()
                    b.writeString("data: {\"event\":\"done\",\"result\":")
                    b.writeBytes(result)
                    b.writeString("}\n\n")
                    continuation.yield(b)
                } else {
                    let detail =
                        error
                        ?? .init(
                            message: "model op failed",
                            type: "server_error", code: "model_op_failed")
                    if let d = try? JSONEncoder().encode(
                        ModelOpErrorEvent(event: "error", error: detail))
                    {
                        continuation.yield(Self.sseFrame(d))
                    }
                }
                var done = ByteBuffer()
                done.writeString("data: [DONE]\n\n")
                continuation.yield(done)
                continuation.finish()
            }
            continuation.onTermination = { _ in
                work.cancel()
                keepAlive.cancel()
            }
        }
        var headers = HTTPFields()
        headers[.contentType] = "text/event-stream"
        headers[.cacheControl] = "no-cache"
        return Response(
            status: .ok, headers: headers,
            body: ResponseBody(asyncSequence: stream))
    }

    /// Collect + validate a model-op body, then stream it. A bad body is a
    /// plain `400` (the stream never opens); a healthy submit returns the
    /// SSE stream (200). The `.modelWrite` gate is applied by the auth
    /// middleware (route → permission); audit it here at initiation.
    private func handleModelPull(_ request: Request) async -> Response {
        await beginModelOp(
            request, kind: "model_pull", action: "model.pull"
        ) { body in
            guard
                let r = try? JSONDecoder().decode(
                    ModelPullRequest.self, from: body), !r.id.isEmpty
            else { return ("model_pull requires non-empty 'id'", nil) }
            return (nil, r.id)
        }
    }

    private func handleModelConvert(_ request: Request) async -> Response {
        await beginModelOp(
            request, kind: "model_convert", action: "model.convert"
        ) { body in
            guard
                let r = try? JSONDecoder().decode(
                    ModelConvertRequest.self, from: body), !r.id.isEmpty
            else { return ("model_convert requires non-empty 'id'", nil) }
            return (nil, r.id)
        }
    }

    private func handleModelPrune(_ request: Request) async -> Response {
        await beginModelOp(
            request, kind: "model_prune", action: "model.prune"
        ) { _ in (nil, nil) }
    }

    /// Shared model-op entry: collect the body (≤ 1 MiB), run `validate`
    /// (returns an error message for a 400, or the audit target id), record
    /// the M30 audit row, and hand off to the SSE stream.
    private func beginModelOp(
        _ request: Request, kind: String, action: String,
        validate: (Data) -> (error: String?, target: String?)
    ) async -> Response {
        let body: Data
        do {
            let buf = try await request.body.collect(upTo: 1 * 1024 * 1024)
            body = Data(buffer: buf)
        } catch {
            return Self.error(
                status: .badRequest,
                message: "Invalid request body: \(error)",
                type: "invalid_request_error", code: "invalid_body")
        }
        let (problem, target) = validate(body)
        if let problem {
            return Self.error(
                status: .badRequest, message: problem,
                type: "invalid_request_error", code: "invalid_body")
        }
        await audit(
            request, action: action, target: target, result: "ok")
        // Operator legibility: a notice-level line per model op (replacing the
        // old `queue submit/running` lines). Emitted inside the request's
        // LogScope so it carries req=/principal=/function= (M45.3).
        Self.log.notice(
            "model op \(action) target=\(target ?? "-")")
        return streamModelOp(kind: kind, body: body)
    }

    /// `/ui` console model op (ADR 025 S2). EventSource is GET-only, so the
    /// browser POST BLOCKS until the op completes and returns the terminal
    /// result JSON (or the error envelope) — no streaming, no job poll.
    /// `.modelWrite` + CSRF are checked by `uiModelMutate`; audited here.
    private func uiModelOp(kind: String, _ r: Request) async -> Response {
        let body: Data
        do {
            let buf = try await r.body.collect(upTo: 1 * 1024 * 1024)
            body = Data(buffer: buf)
        } catch {
            return Self.error(
                status: .badRequest,
                message: "Invalid request body: \(error)",
                type: "invalid_request_error", code: "invalid_body")
        }
        let action = "model." + kind.replacingOccurrences(
            of: "model_", with: "")
        await audit(r, action: action, target: nil, result: "ok")
        let (result, error) = await performModelOp(kind: kind, body: body)
        if let result {
            return Self.jsonString(
                String(data: result, encoding: .utf8) ?? "{}")
        }
        let detail =
            error
            ?? .init(
                message: "model op failed", type: "server_error",
                code: "model_op_failed")
        return Self.error(
            status: HTTPResponse.Status(
                code: detail.type == "invalid_request_error" ? 400 : 500),
            message: detail.message, type: detail.type, code: detail.code)
    }

    // MARK: - Native /api inference (M16)

    /// `Response` isn't `Error`, so a plain success-or-error-response.
    private enum Outcome<T> {
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

    private func decodeJSON<T: Decodable>(
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

    /// Governed embedding helper shared by `/api/embed` and `/v1/embeddings`.
    /// Returns the
    /// whole batch so callers can echo the model ACTUALLY served (M39).
    /// `model` selects among the configured set (nil ⇒ default); an
    /// unknown id surfaces as a classified 400 `model_not_available`.
    private func governedEmbed(
        _ inputs: [String], module: ModuleID, model: String? = nil
    ) async -> Outcome<EmbeddingBatch> {
        do {
            // M43.2: non-blocking cold-load — surfaces as
            // `503 module_loading` to the native /api/embed caller too.
            switch try await governor.awaitLoad(.textEmbedding, within: coldLoadWaitSecs) {
            case .loaded: break
            case .loading:
                return .fail(Self.coldLoadResponse(.textEmbedding))
            }
        } catch let e as AthenaError {
            return .fail(
                Self.error(
                    status: HTTPResponse.Status(code: e.httpStatus),
                    message: e.message, type: "server_error",
                    code: e.code))
        } catch {
            return .fail(Self.classified(error, module: module))
        }
        do {
            return .ok(try await embedding.embed(inputs, model: model))
        } catch {
            return .fail(Self.classified(error, module: module))
        }
    }

    /// `POST /api/embed` — Athena-native embeddings. `input` is a
    /// string or `[string]`; reply is `{model, embeddings:[[Float]]}`.
    private func handleNativeEmbed(_ request: Request) async -> Response {
        let decoded = await decodeJSON(request, AthenaEmbedRequest.self)
        guard case .ok(let body) = decoded else {
            return decoded.orFail
        }
        guard !body.input.isEmpty else {
            return Self.error(
                status: .badRequest, message: "'input' is required",
                type: "invalid_request_error", code: "invalid_input")
        }
        // M39: `body.model` selects among the configured embedding set;
        // the response reports the model actually served (not the request
        // echo). An unknown id ⇒ 400 model_not_available via governedEmbed.
        switch await governedEmbed(
            body.input, module: .textEmbedding, model: body.model)
        {
        case .fail(let r): return r
        case .ok(let batch):
            // Native /api metering (ADR 007 #8): mirror /v1/embeddings —
            // embeddings have no completion, so prompt == total. Closes the
            // gap where `handleNativeChat` metered but the embed twin dropped
            // usage, so `/api/embed` traffic was invisible to usage_counters.
            await meter(
                principal: usagePrincipal(request),
                usage: TokenUsage(
                    promptTokens: batch.promptTokens, completionTokens: 0))
            return Self.json(
                AthenaEmbedResponse(
                    model: batch.model, embeddings: batch.vectors))
        }
    }

    // MARK: - Model store (M16.2)

    /// WP9 — one shared formatter instead of allocating an
    /// `ISO8601DateFormatter` per call on request paths (they're expensive to
    /// construct). `nonisolated(unsafe)`: `Foundation` date formatters are
    /// documented thread-safe for read-only formatting (`.string(from:)`), which
    /// is the only use here, so the shared instance is safe across request tasks.
    private nonisolated(unsafe) static let isoFormatter = ISO8601DateFormatter()
    private static func iso(_ d: Date) -> String {
        isoFormatter.string(from: d)
    }

    /// Conservative model-name guard for NETWORK input: a bare store
    /// child over `[A-Za-z0-9._-]`. Blocks path traversal and — for
    /// the `default` setter — TOML injection via quotes/newlines
    /// (`setScalarThrowing` does not escape string values).
    private static func safeModelName(_ s: String?) -> String? {
        guard let s, !s.isEmpty, s.count <= 128 else { return nil }
        let ok = s.allSatisfy {
            $0.isLetter || $0.isNumber || $0 == "." || $0 == "_"
                || $0 == "-"
        }
        guard ok, s != "..", s != "." else { return nil }
        return s
    }

    private func handleModelsList() -> Response {
        Self.json(
            ModelListResponse(
                models: ModelStoreOps.list(root: modelStoreRoot)
                    .map {
                        ModelEntryDTO(
                            name: $0.name, bytes: $0.bytes,
                            modified: Self.iso($0.modified))
                    }))
    }

    /// `GET /v1/models` (M31.1) — OpenAI list shape over the model store.
    private func handleOpenAIModelsList() -> Response {
        Self.json(
            OpenAIModelList(
                object: "list",
                data: ModelStoreOps.list(root: modelStoreRoot)
                    .map { Self.openAIModel($0.name, $0.modified) }))
    }

    /// `GET /v1/models/:id` (M31.1) — OpenAI retrieve shape; 404 in the
    /// OpenAI error envelope when the model is not in the store.
    private func handleOpenAIModelRetrieve(_ id: String?) -> Response {
        guard let name = Self.safeModelName(id) else {
            return Self.error(
                status: .badRequest, message: "invalid model name",
                type: "invalid_request_error", code: "invalid_name")
        }
        guard
            let d = ModelStoreOps.show(root: modelStoreRoot, name: name)
        else {
            return Self.error(
                status: .notFound, message: "no model '\(name)'",
                type: "invalid_request_error", code: "not_found")
        }
        // A16 — `show` now carries this entry's own mtime, so a single-model
        // retrieve no longer stat-walks the entire store via `list` just to
        // find one `created` time.
        return Self.json(Self.openAIModel(d.name, d.modified))
    }

    private static func openAIModel(_ name: String, _ modified: Date)
        -> OpenAIModel
    {
        OpenAIModel(
            id: name, object: "model",
            created: Int(modified.timeIntervalSince1970),
            owned_by: "athena")
    }

    private func handleModelShow(_ name: String?) -> Response {
        guard let name = Self.safeModelName(name) else {
            return Self.error(
                status: .badRequest, message: "invalid model name",
                type: "invalid_request_error", code: "invalid_name")
        }
        guard
            let d = ModelStoreOps.show(
                root: modelStoreRoot, name: name)
        else {
            return Self.error(
                status: .notFound, message: "no model '\(name)'",
                type: "invalid_request_error", code: "not_found")
        }
        return Self.json(
            ModelDetailResponse(
                name: d.name, path: d.path, bytes: d.bytes,
                config: try? JSONDecoder().decode(
                    JSONValue.self, from: d.configJSON)))
    }

    private func handleModelRemove(
        _ name: String?, _ request: Request
    ) async -> Response {
        guard let name = Self.safeModelName(name) else {
            return Self.error(
                status: .badRequest, message: "invalid model name",
                type: "invalid_request_error", code: "invalid_name")
        }
        do {
            try ModelStoreOps.remove(
                root: modelStoreRoot, name: name)
        } catch ModelStoreOps.OpError.notFound {
            return Self.error(
                status: .notFound, message: "no model '\(name)'",
                type: "invalid_request_error", code: "not_found")
        } catch {
            return Self.error(
                status: .internalServerError, message: "\(error)",
                type: "server_error", code: "remove_failed")
        }
        await audit(
            request, action: "model.remove", target: name,
            result: "ok")
        return Self.json(
            ModelRemovedResponse(name: name, removed: true))
    }

    private func handleModelCopy(_ request: Request) async -> Response {
        let decoded = await decodeJSON(request, ModelCopyRequest.self)
        guard case .ok(let body) = decoded else {
            return decoded.orFail
        }
        // Network copies are store-name → store-name only (no
        // absolute path / traversal source).
        guard
            let src = Self.safeModelName(body.src),
            let dst = Self.safeModelName(body.dst)
        else {
            return Self.error(
                status: .badRequest,
                message: "src and dst must be bare model names",
                type: "invalid_request_error", code: "invalid_name")
        }
        let deep = body.copy ?? false
        do {
            let dest = try ModelStoreOps.copy(
                root: modelStoreRoot, src: src, dst: dst,
                deepCopy: deep, force: body.force ?? false)
            return Self.json(
                ModelCopyResponse(
                    src: src, dst: dst, path: dest.path,
                    aliased: !deep))
        } catch ModelStoreOps.OpError.notFound {
            return Self.error(
                status: .notFound, message: "no model '\(src)'",
                type: "invalid_request_error", code: "not_found")
        } catch ModelStoreOps.OpError.exists {
            return Self.error(
                status: .conflict, message: "'\(dst)' exists",
                type: "invalid_request_error", code: "exists")
        } catch {
            return Self.error(
                status: .internalServerError, message: "\(error)",
                type: "server_error", code: "copy_failed")
        }
    }

    private func handleDefaultModelGet() -> Response {
        let url = ConfigEditor.resolvePath(nil)
        if let cfg = try? AthenaConfig.parse(file: url),
            let model = cfg.model, !model.isEmpty
        {
            return Self.json(
                DefaultModelResponse(model: model, source: "config"))
        }
        return Self.json(
            DefaultModelResponse(
                model: ModelStore.defaultModelName,
                source: "builtin"))
    }

    private func handleDefaultModelSet(_ request: Request) async
        -> Response
    {
        let decoded = await decodeJSON(
            request, SetDefaultModelRequest.self)
        guard case .ok(let body) = decoded else {
            return decoded.orFail
        }
        guard let name = Self.safeModelName(body.name) else {
            return Self.error(
                status: .badRequest,
                message:
                    "name must be a bare model name [A-Za-z0-9._-]",
                type: "invalid_request_error", code: "invalid_name")
        }
        let url = ConfigEditor.resolvePath(nil)
        do {
            try ConfigEditor.setScalarThrowing(
                key: "model", value: name, in: url)
        } catch {
            return Self.error(
                status: .badRequest, message: "\(error)",
                type: "invalid_request_error", code: "config_error")
        }
        await audit(
            request, action: "model.default_set", target: name,
            result: "ok")
        return Self.json(
            DefaultModelResponse(model: name, source: "config"))
    }

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
            // non-default model silently got the default (the consuming application's
            // 4bit→8bit). Validated here so an unknown id is a 400 before a
            // doomed multi-GB load starts; nil/empty ⇒ the default.
            try await llm.selectColdLoadModel(requestedModel)
            // ADR 015: block-until-ready — wait up to `coldLoadWaitSecs` for an
            // on-disk cold-load, then serve (peer-runner behavior); only a
            // timeout or an in-flight download (pull) still 503s. Covers
            // /v1/chat/completions AND /api/chat. (The streaming path layers
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
    private func auditedRebind(
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
    private func servedLLMModel() async -> String {
        let sel = selectable(.llm)
        if let r = await sel.residentModelId() { return r }
        return await sel.defaultModelId()
    }


    /// The `any ModelSelectable` corresponding to `id`. Every concrete
    /// module conforms (the stubs too), so this is total — no `nil`.
    private func selectable(_ id: ModuleID) -> any ModelSelectable {
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

    private func handleModelsResident() async -> Response {
        var slots: [ModelSlotDTO] = []
        for id in ModuleID.allCases {
            let sel = selectable(id)
            let allowed = await sel.allowedModelIds()
            let def = await sel.defaultModelId()
            let resident = await sel.residentModelId()
            slots.append(
                ModelSlotDTO(
                    module: id.rawValue, allowed: allowed,
                    default: def, resident: resident))
        }
        return Self.json(ModelResidentResponse(slots: slots))
    }

    /// `POST /api/models/load` (M41.1) — rebind `body.module`'s slot to
    /// `body.id` (omit ⇒ the module's default). The slot is governor-
    /// loaded first (so the rebind has its fixed reservation), then
    /// the module rebinds in place. An id outside the module's
    /// allowlist surfaces as a classified 400 (`model_not_available`);
    /// a substrate load failure becomes a classified
    /// 500/503 (Metal OOM). Gated `model.write` by AuthPolicy. Audited
    /// (M30): caller + module + id + result.
    private func handleModelsLoad(_ request: Request) async -> Response {
        let decoded = await decodeJSON(request, ModelLoadRequest.self)
        guard case .ok(let body) = decoded else {
            return decoded.orFail
        }
        guard
            let moduleId = ModuleID(rawValue: body.module)
        else {
            await audit(
                request, action: "model.load", target: body.module,
                result: "denied", detail: "unknown module")
            return Self.error(
                status: .badRequest,
                message: "unknown module '\(body.module)'",
                type: "invalid_request_error", code: "invalid_module")
        }
        let sel = selectable(moduleId)
        let allowed = await sel.allowedModelIds()
        let def = await sel.defaultModelId()
        let requested = body.id ?? def
        // NE5 — store-identity lookup (bare name OR full HF id), uniform
        // with the LLM rebind/select path and the embedding/audio modules;
        // the canonical stored id drives the load.
        guard let target =
            allowed.canonicalByStoreIdentity(requested)
        else {
            await audit(
                request, action: "model.load",
                target: "\(moduleId.rawValue):\(requested)",
                result: "denied", detail: "id outside allowlist")
            return Self.error(
                status: .badRequest,
                message:
                    "Model '\(requested)' is not available. Configured "
                    + "models for \(moduleId.rawValue): "
                    + "\(allowed.dedupedCaseInsensitive().joined(separator: ", ")).",
                type: "invalid_request_error",
                code: "model_not_available")
        }
        do {
            try await governor.ensureLoaded(moduleId)
            // ADR 029 WP3 — gate the rebind exactly as the request-path
            // `auditedRebind` (:3761) does: a warm swap drops+loads weights on
            // the Metal pool and must not run while a decode holds the slot
            // (transient double-residency → OOM) or another tenant executes.
            // This control-plane path previously rebound off-gate (the H3
            // hazard reachable via /api/models/load).
            try await InferenceGate.shared.withExclusiveExecution {
                try await sel.rebind(to: target)
            }
        } catch let e as AthenaError {
            await audit(
                request, action: "model.load",
                target: "\(moduleId.rawValue):\(target)",
                result: "denied", detail: e.code)
            return Self.error(
                status: HTTPResponse.Status(code: e.httpStatus),
                message: e.message, type: "server_error", code: e.code)
        } catch {
            await audit(
                request, action: "model.load",
                target: "\(moduleId.rawValue):\(target)",
                result: "denied", detail: String(describing: error))
            return Self.classified(error, module: moduleId)
        }
        await audit(
            request, action: "model.load",
            target: "\(moduleId.rawValue):\(target)", result: "ok")
        return Self.json(
            ModelLoadResponse(
                module: moduleId.rawValue, id: target,
                status: "loaded"))
    }

    /// `POST /api/models/unload` (M41.1) — release a single module's
    /// slot, or all of them when `module` is absent / `"all"`. Daemon
    /// keeps running; the next inference reloads the module's default
    /// lazily. Gated `model.write` by AuthPolicy. Audited (M30).
    private func handleModelsUnload(_ request: Request) async
        -> Response
    {
        let decoded = await decodeJSON(request, ModelUnloadRequest.self)
        guard case .ok(let body) = decoded else {
            return decoded.orFail
        }
        let targets: [ModuleID]
        if let raw = body.module, raw != "all", !raw.isEmpty {
            guard let m = ModuleID(rawValue: raw) else {
                await audit(
                    request, action: "model.unload", target: raw,
                    result: "denied", detail: "unknown module")
                return Self.error(
                    status: .badRequest,
                    message: "unknown module '\(raw)'",
                    type: "invalid_request_error",
                    code: "invalid_module")
            }
            targets = [m]
        } else {
            targets = ModuleID.allCases
        }
        var unloaded: [String] = []
        for m in targets {
            await governor.unload(m)
            unloaded.append(m.rawValue)
        }
        await audit(
            request, action: "model.unload",
            target: unloaded.joined(separator: ","),
            result: "ok")
        return Self.json(
            ModelUnloadResponse(
                modules: unloaded, status: "unloaded"))
    }

    // MARK: - WebUI model console reuse (M18.2)

    /// 403 JSON for a /ui/api/* action the logged-in user's perms
    /// (or CSRF) reject. AuthPolicy already gated /ui* on
    /// daemonAdmin; this is the per-action defense-in-depth
    /// re-check (the page is never trusted). Lives here (not in
    /// WebUI.swift) so it can reuse the file-private M16 op methods.
    private static func uiDeny(_ msg: String) -> Response {
        Self.error(
            status: .forbidden, message: msg,
            type: "auth_error", code: "forbidden")
    }

    /// First raw value of `key` in the query string. Our model
    /// names are `[A-Za-z0-9._-]` (validated downstream by
    /// `safeModelName`) so no percent-decode is needed.
    private static func uiQuery(
        _ r: Request, _ key: String
    ) -> String? {
        guard let q = r.uri.query else { return nil }
        for kv in q.split(separator: "&") {
            let p = kv.split(separator: "=", maxSplits: 1)
            if p.first.map(String.init) == key {
                return p.count == 2 ? String(p[1]) : ""
            }
        }
        return nil
    }

    private func uiModelsList(_ r: Request) async -> Response {
        guard await uiCaller(r).perms.contains(.modelRead) else {
            return Self.uiDeny("need model.read")
        }
        return handleModelsList()
    }

    private func uiModelShow(_ r: Request) async -> Response {
        guard await uiCaller(r).perms.contains(.modelRead) else {
            return Self.uiDeny("need model.read")
        }
        return handleModelShow(Self.uiQuery(r, "name"))
    }

    private func uiDefaultGet(_ r: Request) async -> Response {
        guard await uiCaller(r).perms.contains(.modelRead) else {
            return Self.uiDeny("need model.read")
        }
        return handleDefaultModelGet()
    }

    /// CSRF + per-action `.modelWrite` re-check, THEN delegate to
    /// the shared M16 op (ModelStoreOps / performModelOp). Mutations
    /// only. `op` is non-escaping (invoked here, never stored).
    private func uiModelMutate(
        _ r: Request, _ op: (Request) async -> Response
    ) async -> Response {
        guard csrfOK(r) else {
            return Self.uiDeny("csrf token missing or invalid")
        }
        guard await uiCaller(r).perms.contains(.modelWrite) else {
            return Self.uiDeny("need model.write")
        }
        return await op(r)
    }

    private func uiModelRemove(_ r: Request) async -> Response {
        await uiModelMutate(r) { req in
            let decoded = await self.decodeJSON(
                req, SetDefaultModelRequest.self)
            guard case .ok(let body) = decoded else {
                return decoded.orFail
            }
            return await self.handleModelRemove(body.name, req)
        }
    }

    // MARK: - Daemon control ops (M16 admin + M18.3 reuse)

    /// Unload the LLM module (frees its memory). Shared by
    /// `POST /api/admin/stop` (bearer) and the M18.3 `/ui/api/admin/
    /// stop` (cookie) — one implementation, no duplication.
    private func adminUnloadLLM(_ request: Request) async -> Response {
        await governor.unload(.llm)
        await audit(
            request, action: "daemon.unload", target: modelName,
            result: "ok")
        return Self.json(
            AthenaStopResponse(status: "unloaded", model: modelName))
    }

    /// Pre-warm the LLM module so the next inference is hot. New
    /// daemon-control verb (M18.3); exposed only on the cookie+CSRF
    /// /ui surface (the public /api admin surface is frozen at M16).
    private func adminLoadLLM(_ request: Request) async -> Response {
        do {
            try await governor.ensureLoaded(.llm)
            await audit(
                request, action: "daemon.load", target: modelName,
                result: "ok")
            return Self.json(
                AthenaStopResponse(
                    status: "loaded", model: modelName))
        } catch {
            return Self.classified(error, module: .llm)
        }
    }

    /// Daemon posture snapshot. Shared by `GET /api/admin/status`
    /// (bearer) and `/ui/api/admin/status` (cookie).
    private func adminStatus() async -> Response {
        Self.json(
            AdminStatusResponse(
                model: modelName,
                listen:
                    "\(config.listenHost):\(config.listenPort)",
                auth_enabled: auth.isEnabled,
                users: await store.userCount(),
                tokens: await store.tokenCount(),
                admins:
                    (try? await store.usersWithRole("admin"))?.count ?? 0))
    }

    // MARK: - WebUI daemon console reuse (M18.3)

    private func uiAdminStatus(_ r: Request) async -> Response {
        guard await uiCaller(r).perms.contains(.daemonAdmin) else {
            return Self.uiDeny("need daemon.admin")
        }
        return await adminStatus()
    }

    /// Destructive: CSRF + per-action `.daemonAdmin` re-check on the
    /// logged-in user, then the shared unload op. The UI also gates
    /// this behind an explicit confirm step (the page is not
    /// trusted; the server decides).
    private func uiAdminStop(_ r: Request) async -> Response {
        guard csrfOK(r) else {
            return Self.uiDeny("csrf token missing or invalid")
        }
        guard await uiCaller(r).perms.contains(.daemonAdmin) else {
            return Self.uiDeny("need daemon.admin")
        }
        return await adminUnloadLLM(r)
    }

    private func uiAdminLoad(_ r: Request) async -> Response {
        guard csrfOK(r) else {
            return Self.uiDeny("csrf token missing or invalid")
        }
        guard await uiCaller(r).perms.contains(.daemonAdmin) else {
            return Self.uiDeny("need daemon.admin")
        }
        return await adminLoadLLM(r)
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
    private func callerPermissions(_ request: Request) async
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
    private func auditPrincipal(_ request: Request) async -> String {
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
    private func audit(
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

    /// Bare identifier guard for network input (username/role names):
    /// `[A-Za-z0-9._-]`, ≤64, not `.`/`..`.
    private static func safeIdent(_ s: String?) -> String? {
        guard let s, !s.isEmpty, s.count <= 64 else { return nil }
        let ok = s.allSatisfy {
            $0.isLetter || $0.isNumber || $0 == "." || $0 == "_"
                || $0 == "-"
        }
        guard ok, s != "..", s != "." else { return nil }
        return s
    }

    /// Caller may assign every role in `roles` only if it holds every
    /// permission each confers (fail-closed on an unknown role).
    private static func canGrantAll(
        _ roles: some Sequence<String>, _ caller: Set<Permission>
    ) -> Bool {
        roles.allSatisfy {
            RBAC.canGrant(role: $0, grantorPermissions: caller)
        }
    }

    private func handleUsersList() async -> Response {
        var out: [UserSummaryDTO] = []
        for u in await store.listUsers() {
            out.append(
                UserSummaryDTO(
                    username: u,
                    roles: await store.rolesForUser(username: u)))
        }
        return Self.json(UserListResponse(users: out))
    }

    private func handleUserCreate(_ request: Request) async
        -> Response
    {
        let decoded = await decodeJSON(request, CreateUserRequest.self)
        guard case .ok(let body) = decoded else {
            return decoded.orFail
        }
        guard let username = Self.safeIdent(body.username) else {
            return Self.error(
                status: .badRequest,
                message: "invalid username [A-Za-z0-9._-]",
                type: "invalid_request_error", code: "invalid_name")
        }
        guard body.password.count >= 8 else {
            return Self.error(
                status: .badRequest,
                message: "password must be >= 8 chars",
                type: "invalid_request_error", code: "weak_password")
        }
        let role = body.role ?? "member"
        guard RBAC.isValidRole(role) else {
            return Self.error(
                status: .badRequest,
                message: "unknown role '\(role)'",
                type: "invalid_request_error", code: "unknown_role")
        }
        let caller = await callerPermissions(request)
        guard
            RBAC.canGrant(role: role, grantorPermissions: caller)
        else {
            await audit(
                request, action: "user.create", target: username,
                result: "denied", detail: "role '\(role)' exceeds "
                    + "grantor permissions")
            return Self.deny403(
                "cannot grant a role conferring permissions you do "
                    + "not hold")
        }
        // Replacing an existing account = a password reset; refuse if
        // the target currently outranks the caller (account takeover).
        if await store.getUser(username: username) != nil {
            let cur = await store.rolesForUser(username: username)
            guard
                RBAC.permissions(forRoles: cur).isSubset(of: caller)
            else {
                await audit(
                    request, action: "user.create",
                    target: username, result: "denied",
                    detail: "target outranks caller")
                return Self.deny403(
                    "refusing to replace a user whose roles exceed "
                        + "your permissions")
            }
        }
        let salt = Passwords.randomSalt()
        let hash = Passwords.derive(
            password: body.password, salt: salt,
            iters: Passwords.defaultIterations)
        do {
            try await store.putUser(
                username: username, salt: salt, hash: hash,
                iters: Passwords.defaultIterations)
            try await store.grantRole(
                username: username, role: role)
        } catch {
            return Self.storeError(error)
        }
        await audit(
            request, action: "user.create", target: username,
            result: "ok", detail: "role=\(role)")
        return Self.json(
            UserSummaryDTO(
                username: username,
                roles: await store.rolesForUser(
                    username: username)))
    }

    private func handleUserDelete(
        _ name: String?, _ request: Request
    ) async -> Response {
        guard let username = Self.safeIdent(name) else {
            return Self.error(
                status: .badRequest, message: "invalid username",
                type: "invalid_request_error", code: "invalid_name")
        }
        guard await store.getUser(username: username) != nil else {
            return Self.error(
                status: .notFound, message: "no user '\(username)'",
                type: "invalid_request_error", code: "not_found")
        }
        let onlyAdmin: Bool
        do {
            onlyAdmin = try await store.usersWithRole("admin") == [username]
        } catch {
            // NB11: fail closed — refuse the delete if the admin set can't
            // be read, rather than risk stripping the last admin.
            await audit(
                request, action: "user.delete", target: username,
                result: "denied", detail: "admin-set check failed")
            return Self.error(
                status: .internalServerError,
                message: "could not verify the admin set",
                type: "server_error", code: "store_error")
        }
        if onlyAdmin {
            await audit(
                request, action: "user.delete", target: username,
                result: "denied", detail: "only admin")
            return Self.deny403(
                "'\(username)' is the only admin — refusing "
                    + "(grant admin to another user first)")
        }
        let ok: Bool
        do {
            ok = try await store.deleteUser(username: username)
        } catch {
            // H11 (M66.1): the cascade rolled back as a unit — surface the
            // failure instead of reporting a partial delete as success.
            await audit(
                request, action: "user.delete", target: username,
                result: "denied", detail: "store error")
            return Self.error(
                status: .internalServerError,
                message: "failed to delete user",
                type: "server_error", code: "store_error")
        }
        await audit(
            request, action: "user.delete", target: username,
            result: ok ? "ok" : "denied")
        return Self.json(
            UserRemovedResponse(username: username, removed: ok))
    }

    private func handleRoleGrant(
        _ name: String?, _ role: String?, _ request: Request
    ) async -> Response {
        guard let username = Self.safeIdent(name),
            let role = Self.safeIdent(role)
        else {
            return Self.error(
                status: .badRequest, message: "invalid name/role",
                type: "invalid_request_error", code: "invalid_name")
        }
        guard RBAC.isValidRole(role) else {
            return Self.error(
                status: .badRequest,
                message: "unknown role '\(role)'",
                type: "invalid_request_error", code: "unknown_role")
        }
        guard await store.getUser(username: username) != nil else {
            return Self.error(
                status: .notFound, message: "no user '\(username)'",
                type: "invalid_request_error", code: "not_found")
        }
        let caller = await callerPermissions(request)
        guard
            RBAC.canGrant(role: role, grantorPermissions: caller)
        else {
            await audit(
                request, action: "role.grant",
                target: "\(username):\(role)", result: "denied",
                detail: "role exceeds grantor permissions")
            return Self.deny403(
                "cannot grant a role conferring permissions you do "
                    + "not hold")
        }
        do {
            try await store.grantRole(
                username: username, role: role)
        } catch {
            return Self.storeError(error)
        }
        await audit(
            request, action: "role.grant",
            target: "\(username):\(role)", result: "ok")
        return Self.json(OkResponse(ok: true))
    }

    private func handleRoleRevoke(
        _ name: String?, _ role: String?, _ request: Request
    ) async -> Response {
        guard let username = Self.safeIdent(name),
            let role = Self.safeIdent(role)
        else {
            return Self.error(
                status: .badRequest, message: "invalid name/role",
                type: "invalid_request_error", code: "invalid_name")
        }
        guard await store.getUser(username: username) != nil else {
            return Self.error(
                status: .notFound, message: "no user '\(username)'",
                type: "invalid_request_error", code: "not_found")
        }
        if role == "admin" {
            let onlyAdmin: Bool
            do {
                onlyAdmin =
                    try await store.usersWithRole("admin") == [username]
            } catch {
                // NB11: can't read the admin set ⇒ fail closed (refuse),
                // never assume "not the last admin" on a query error.
                await audit(
                    request, action: "role.revoke",
                    target: "\(username):\(role)", result: "denied",
                    detail: "admin-set check failed")
                return Self.error(
                    status: .internalServerError,
                    message: "could not verify the admin set",
                    type: "server_error", code: "store_error")
            }
            if onlyAdmin {
                await audit(
                    request, action: "role.revoke",
                    target: "\(username):\(role)", result: "denied",
                    detail: "only admin")
                return Self.deny403(
                    "'\(username)' is the only admin — refusing to "
                        + "revoke admin")
            }
        }
        let ok = await store.revokeRole(
            username: username, role: role)
        await audit(
            request, action: "role.revoke",
            target: "\(username):\(role)", result: ok ? "ok" : "denied")
        return Self.json(OkResponse(ok: ok))
    }

    private func handleTokensList() async -> Response {
        let toks = await store.listTokens()
        return Self.json(
            TokenListResponse(
                tokens: toks.map {
                    TokenSummaryDTO(
                        username: $0.username, scope: $0.scoped,
                        hash_prefix: $0.hashPrefix,
                        label: $0.label, expires: $0.expires)
                }))
    }

    private func handleTokenCreate(_ request: Request) async
        -> Response
    {
        let decoded = await decodeJSON(
            request, CreateTokenRequest.self)
        guard case .ok(let body) = decoded else {
            return decoded.orFail
        }
        guard let user = Self.safeIdent(body.user) else {
            return Self.error(
                status: .badRequest, message: "invalid user",
                type: "invalid_request_error", code: "invalid_name")
        }
        guard await store.getUser(username: user) != nil else {
            return Self.error(
                status: .notFound, message: "no user '\(user)'",
                type: "invalid_request_error", code: "not_found")
        }
        let caller = await callerPermissions(request)
        let scoped =
            (body.role?.isEmpty == false) ? body.role : nil
        if let scoped {
            for r in scoped where !RBAC.isValidRole(r) {
                return Self.error(
                    status: .badRequest,
                    message: "unknown role '\(r)'",
                    type: "invalid_request_error",
                    code: "unknown_role")
            }
            guard Self.canGrantAll(scoped, caller) else {
                await audit(
                    request, action: "token.create", target: user,
                    result: "denied", detail: "scope exceeds caller")
                return Self.deny403(
                    "token scope exceeds your permissions")
            }
        } else {
            // Unscoped ⇒ inherits the user's full roles; refuse to
            // mint a token more powerful than the caller.
            let inherited = RBAC.permissions(
                forRoles: await store.rolesForUser(username: user))
            guard inherited.isSubset(of: caller) else {
                await audit(
                    request, action: "token.create", target: user,
                    result: "denied",
                    detail: "unscoped token exceeds caller")
                return Self.deny403(
                    "an unscoped token for '\(user)' would exceed "
                        + "your permissions — pass an explicit role "
                        + "scope you can grant")
            }
        }
        let (key, hash) = AuthConfig.mintToken()
        let expires = body.ttl_secs.flatMap {
            $0 > 0 ? Date().timeIntervalSince1970 + Double($0) : nil
        }
        do {
            try await store.putToken(
                hash: hash, username: user, scopedRoles: scoped,
                label: body.label, expires: expires)
        } catch {
            return Self.storeError(error)
        }
        let hashPrefix = String(
            AuthConfig.hex(Array(hash)).prefix(12))
        await audit(
            request, action: "token.create", target: user,
            result: "ok",
            detail: "hash=\(hashPrefix) scope="
                + (scoped?.joined(separator: ",") ?? "inherit"))
        return Self.json(
            CreateTokenResponse(
                user: user, scope: scoped, token: key,
                hash_prefix: hashPrefix),
            status: .accepted)
    }

    /// 64-char hex → 32 bytes (token-hash reconstruction).
    private static func hexData(_ s: String) -> Data? {
        guard s.count == 64 else { return nil }
        var out = [UInt8]()
        out.reserveCapacity(32)
        var i = s.startIndex
        while i < s.endIndex {
            let j = s.index(i, offsetBy: 2)
            guard let b = UInt8(s[i..<j], radix: 16) else {
                return nil
            }
            out.append(b)
            i = j
        }
        return Data(out)
    }

    private func handleTokenDelete(
        _ prefix: String?, _ request: Request
    ) async -> Response {
        guard let prefix, prefix.count >= 6,
            prefix.allSatisfy({ $0.isHexDigit })
        else {
            return Self.error(
                status: .badRequest,
                message: "prefix must be >= 6 hex chars",
                type: "invalid_request_error", code: "invalid_prefix")
        }
        let matches = await store.tokensMatchingHashPrefix(prefix)
        if matches.isEmpty {
            return Self.error(
                status: .notFound,
                message: "no token matched \(prefix)",
                type: "invalid_request_error", code: "not_found")
        }
        var removed = 0
        for m in matches {
            if let d = Self.hexData(m.hex),
                await store.deleteToken(hash: d)
            {
                removed += 1
            }
        }
        await audit(
            request, action: "token.delete", target: prefix,
            result: "ok", detail: "removed=\(removed)")
        return Self.json(TokensRemovedResponse(removed: removed))
    }

    /// `POST /api/tokens/{prefix}/rotate` (M36.2) — revoke + reissue.
    /// The prefix must match EXACTLY one token; its owner/scope/label
    /// carry to a fresh secret (returned once), the old hash is deleted.
    /// `ttl_secs` sets the new token's lifetime (absent ⇒ no expiry).
    /// Same scope guard as create: the caller may not re-mint a token
    /// more powerful than they can grant.
    private func handleTokenRotate(
        _ prefix: String?, _ request: Request
    ) async -> Response {
        guard let prefix, prefix.count >= 6,
            prefix.allSatisfy({ $0.isHexDigit })
        else {
            return Self.error(
                status: .badRequest,
                message: "prefix must be >= 6 hex chars",
                type: "invalid_request_error", code: "invalid_prefix")
        }
        let decoded = await decodeJSON(request, RotateTokenRequest.self)
        guard case .ok(let body) = decoded else {
            return decoded.orFail
        }
        let matches = await store.tokensMatchingHashPrefix(prefix)
        guard matches.count == 1, let m = matches.first,
            let oldHash = Self.hexData(m.hex)
        else {
            return Self.error(
                status: matches.count > 1 ? .conflict : .notFound,
                message: matches.count > 1
                    ? "prefix \(prefix) matched \(matches.count) "
                        + "tokens — use a longer, unambiguous prefix"
                    : "no token matched \(prefix)",
                type: "invalid_request_error",
                code: matches.count > 1 ? "ambiguous_prefix"
                    : "not_found")
        }
        // Same scope guard as create — rotation must not re-mint a token
        // more powerful than the caller can grant.
        let caller = await callerPermissions(request)
        if let scoped = m.scoped {
            guard Self.canGrantAll(scoped, caller) else {
                await audit(
                    request, action: "token.rotate", target: m.username,
                    result: "denied", detail: "scope exceeds caller")
                return Self.deny403("token scope exceeds your permissions")
            }
        } else {
            let inherited = RBAC.permissions(
                forRoles: await store.rolesForUser(username: m.username))
            guard inherited.isSubset(of: caller) else {
                await audit(
                    request, action: "token.rotate", target: m.username,
                    result: "denied",
                    detail: "unscoped token exceeds caller")
                return Self.deny403(
                    "rotating this unscoped token would exceed your "
                        + "permissions")
            }
        }
        let expires = body.ttl_secs.flatMap {
            $0 > 0 ? Date().timeIntervalSince1970 + Double($0) : nil
        }
        // NA4 (M66.2): persist the NEW token BEFORE revoking the old one.
        // The previous order (delete-then-put) left the principal with no
        // working token if putToken threw — a lockout on partial failure.
        // New + old briefly coexist (distinct hashes); we revoke the old
        // only once the new is durably stored.
        let (key, hash) = AuthConfig.mintToken()
        do {
            try await store.putToken(
                hash: hash, username: m.username, scopedRoles: m.scoped,
                label: m.label, expires: expires)
        } catch {
            return Self.storeError(error)
        }
        _ = await store.deleteToken(hash: oldHash)
        let newPrefix = String(AuthConfig.hex(Array(hash)).prefix(12))
        await audit(
            request, action: "token.rotate", target: m.username,
            result: "ok",
            detail: "old=\(prefix) new=\(newPrefix)")
        return Self.json(
            CreateTokenResponse(
                user: m.username, scope: m.scoped, token: key,
                hash_prefix: newPrefix),
            status: .accepted)
    }

    /// Shared remediation for the in-handler RBAC/privilege guards
    /// (#12 / M43.4). Every `deny403` is either an "insufficient
    /// privilege" or a "safety guard" refusal, so one hint covers them;
    /// surfaced as `error.hint` and rendered by the CLI client. Mirrors
    /// the auth-middleware forbidden hint for envelope parity.
    static let privilegedActionHint =
        "Refused: this action needs elevated privileges or trips a "
        + "safety guard (e.g. removing the last admin). Use an admin "
        + "token, have an admin perform it, or adjust roles with "
        + "`athena auth role grant <user> <role>`."

    private static func deny403(
        _ msg: String, hint: String? = nil
    ) -> Response {
        Self.error(
            status: .forbidden, message: msg,
            type: "auth_error", code: "forbidden",
            hint: hint ?? privilegedActionHint)
    }

    /// The compiled-in RBAC role→perms catalog. Shared by
    /// `GET /api/roles` (bearer) and `/ui/api/roles` (cookie).
    private static func rolesCatalogResponse() -> Response {
        Self.json(
            RolesResponse(
                roles: RBAC.roleNames.map { r in
                    RoleCatalogEntry(
                        role: r,
                        permissions: (RBAC.catalog[r] ?? [])
                            .map(\.rawValue).sorted())
                }))
    }

    // MARK: - WebUI RBAC admin reuse (M18.4)

    /// Small `{key:value}` JSON body → map (for the name/role/prefix
    /// the M16.4 handlers take as path params over `/api/*`; the
    /// cookie /ui passes them in the body instead). Only consumed
    /// for delete/grant/revoke — handlers that DON'T re-read the
    /// body; create/token-create are passed the Request untouched so
    /// they decode their own typed DTO.
    private func uiJSONMap(_ r: Request) async -> [String: String] {
        guard
            let buf = try? await r.body.collect(upTo: 64 * 1024),
            let m = try? JSONDecoder().decode(
                [String: String].self, from: Data(buffer: buf))
        else { return [:] }
        return m
    }

    /// Cookie + per-action perm re-check (+ CSRF when `mutating`),
    /// then run `op`. Reuses the M16.4 handlers as-is: their inner
    /// `callerPermissions` is now cookie-aware (M18.4) so canGrant /
    /// last-admin / cross-rank guards bind to the LOGGED-IN user.
    private func uiRBAC(
        _ r: Request, _ need: Permission, mutating: Bool,
        _ op: (Request) async -> Response
    ) async -> Response {
        if mutating, !csrfOK(r) {
            return Self.uiDeny("csrf token missing or invalid")
        }
        guard await uiCaller(r).perms.contains(need) else {
            return Self.uiDeny("need \(need.rawValue)")
        }
        return await op(r)
    }

    private func uiUsersList(_ r: Request) async -> Response {
        await uiRBAC(r, .usersRead, mutating: false) { _ in
            await self.handleUsersList()
        }
    }
    private func uiRolesList(_ r: Request) async -> Response {
        await uiRBAC(r, .usersRead, mutating: false) { _ in
            Self.rolesCatalogResponse()
        }
    }
    private func uiUserCreate(_ r: Request) async -> Response {
        await uiRBAC(r, .usersAdmin, mutating: true) { req in
            await self.handleUserCreate(req)
        }
    }
    private func uiUserDelete(_ r: Request) async -> Response {
        await uiRBAC(r, .usersAdmin, mutating: true) { req in
            let f = await self.uiJSONMap(req)
            return await self.handleUserDelete(f["name"], req)
        }
    }
    private func uiRoleGrant(_ r: Request) async -> Response {
        await uiRBAC(r, .usersAdmin, mutating: true) { req in
            let f = await self.uiJSONMap(req)
            return await self.handleRoleGrant(
                f["name"], f["role"], req)
        }
    }
    private func uiRoleRevoke(_ r: Request) async -> Response {
        await uiRBAC(r, .usersAdmin, mutating: true) { req in
            let f = await self.uiJSONMap(req)
            return await self.handleRoleRevoke(
                f["name"], f["role"], req)
        }
    }
    private func uiTokensList(_ r: Request) async -> Response {
        await uiRBAC(r, .tokensAdmin, mutating: false) { _ in
            await self.handleTokensList()
        }
    }
    private func uiTokenCreate(_ r: Request) async -> Response {
        await uiRBAC(r, .tokensAdmin, mutating: true) { req in
            await self.handleTokenCreate(req)
        }
    }
    private func uiTokenDelete(_ r: Request) async -> Response {
        await uiRBAC(r, .tokensAdmin, mutating: true) { req in
            let f = await self.uiJSONMap(req)
            return await self.handleTokenDelete(f["prefix"], req)
        }
    }

    // MARK: - Response helpers

    /// Stream `/v1/chat/completions` over SSE (M27.4). Consumes the
    /// metered stream so it can (a) emit a terminal usage chunk when the
    /// client set `stream_options.include_usage` and (b) always meter the
    /// request via `record` once generation finishes — closing the
    /// streaming metering gap from M27.1. `record` runs inside the
    /// streaming task (the body is produced lazily).
    private static func streamSSE(
        id: String, model: String, created: Int,
        events: AsyncStream<GenChunk>, includeUsage: Bool,
        isToolCall: Bool = false,
        stops: [String] = [],
        onConsumerCancel: (@Sendable () -> Void)? = nil,
        record: @escaping @Sendable (TokenUsage) async -> Void
    ) -> Response {
        let stream = AsyncStream<ByteBuffer> { continuation in
            let task = Task {
                await pumpTokens(
                    into: continuation, id: id, model: model,
                    created: created, events: events,
                    includeUsage: includeUsage, isToolCall: isToolCall,
                    stops: stops, record: record)
            }
            // A8 (M68.4) — a client disconnect terminates THIS byte stream;
            // bridge it to the generation's cancel flag so the synchronous
            // decode loops (which poll `DecodeProgress.counter?.isCancelled`,
            // not `Task.isCancelled`) stop instead of decoding to maxTokens
            // for a request no one is reading.
            continuation.onTermination = { _ in
                task.cancel()
                onConsumerCancel?()
            }
        }
        var headers = HTTPFields()
        headers[.contentType] = "text/event-stream"
        headers[.cacheControl] = "no-cache"
        return Response(
            status: .ok, headers: headers,
            body: ResponseBody(asyncSequence: stream))
    }

    /// ADR 015 — outcome of the in-SSE cold-load wait. `ready` ⇒ proceed to
    /// post-load validation + token streaming; `timedOut`/`failed` ⇒ emit an
    /// in-stream OpenAI-style error event then `[DONE]`.
    // ADR 036 — internal (not private) so the protocol-agnostic `ChatPrep`
    // seam can carry it and a future dialect adapter in a sibling file can too.
    enum ColdStreamLoad: Sendable {
        case ready
        case timedOut
        case failed(message: String, type: String, code: String)
    }

    /// ADR 015 — the streaming counterpart of the block-until-ready gate: open
    /// the SSE `200` immediately, emit `: loading` keep-alive comments while the
    /// model loads (so a reverse proxy doesn't idle-time-out and the client sees
    /// liveness), then run the MODEL-DEPENDENT validations (`prepareAfterLoad`:
    /// rebind / vision / prompt-cap) — surfacing any failure as an in-stream
    /// error event, OpenAI-consistent — and finally stream tokens. A load
    /// timeout or failure becomes an in-stream error, not a dropped connection.
    /// Used only when `peekLoad` said `.needsLoad`; the warm path uses
    /// `streamSSE` and never emits keep-alives.
    private static func streamSSEAwaitingLoad(
        id: String, created: Int,
        modelName: @escaping @Sendable () async -> String,
        load: @escaping @Sendable () async -> ColdStreamLoad,
        prepareAfterLoad:
            @escaping @Sendable () async -> (
                message: String, type: String, code: String
            )?,
        eventsBuilder: @escaping @Sendable () -> AsyncStream<GenChunk>,
        includeUsage: Bool, isToolCall: Bool = false, stops: [String] = [],
        onConsumerCancel: (@Sendable () -> Void)? = nil,
        record: @escaping @Sendable (TokenUsage) async -> Void
    ) -> Response {
        let stream = AsyncStream<ByteBuffer> { continuation in
            let task = Task {
                func emitError(
                    _ message: String, _ type: String, _ code: String
                ) {
                    let body = APIErrorBody(
                        error: .init(
                            message: message, type: type, code: code))
                    if let data = try? JSONEncoder().encode(body) {
                        var buf = ByteBuffer()
                        buf.writeString("data: ")
                        buf.writeBytes(data)
                        buf.writeString("\n\n")
                        continuation.yield(buf)
                    }
                    var done = ByteBuffer()
                    done.writeString("data: [DONE]\n\n")
                    continuation.yield(done)
                }
                // Emit `: loading` SSE comments on a timer (a child of the group
                // so a consumer disconnect — which cancels `task` — stops it),
                // while the bounded load runs. Comment lines keep the byte
                // stream alive without being surfaced to the client as content.
                let outcome: ColdStreamLoad = await withTaskGroup(
                    of: Void.self
                ) { group in
                    group.addTask {
                        while !Task.isCancelled {
                            do {
                                try await Task.sleep(
                                    nanoseconds: 10_000_000_000)
                            } catch { break }
                            var b = ByteBuffer()
                            b.writeString(": loading\n\n")
                            continuation.yield(b)
                        }
                    }
                    let o = await load()
                    group.cancelAll()  // stop the keep-alive ticker
                    return o
                }
                switch outcome {
                case .timedOut:
                    emitError(
                        "model is loading; retry shortly", "server_error",
                        "module_loading")
                    continuation.finish()
                    return
                case .failed(let m, let t, let c):
                    emitError(m, t, c)
                    continuation.finish()
                    return
                case .ready:
                    if let err = await prepareAfterLoad() {
                        emitError(err.message, err.type, err.code)
                        continuation.finish()
                        return
                    }
                    let model = await modelName()
                    await pumpTokens(
                        into: continuation, id: id, model: model,
                        created: created, events: eventsBuilder(),
                        includeUsage: includeUsage, isToolCall: isToolCall,
                        stops: stops, record: record)
                }
            }
            continuation.onTermination = { _ in
                task.cancel()
                onConsumerCancel?()
            }
        }
        var headers = HTTPFields()
        headers[.contentType] = "text/event-stream"
        headers[.cacheControl] = "no-cache"
        return Response(
            status: .ok, headers: headers,
            body: ResponseBody(asyncSequence: stream))
    }

    /// ADR 036 WP7 — the per-dialect sink `foldGenChunks` drives. A dialect
    /// supplies only how to *encode* a chunk; the fold owns the shared decode
    /// logic (stop-filter latching, ADR-035 reasoning peel, tool buffering,
    /// finish-reason + stop-sequence attribution). Collapses the two SSE pumps
    /// into one traversal so detection can't drift between OpenAI and Anthropic.
    struct ProtocolEncoder {
        /// One content delta (already stop-filtered; may be the tool-parse-fail
        /// fallback text). Empty pieces are the encoder's to drop.
        var emitText: (String) -> Void
        /// One reasoning delta (ADR 035). Anthropic passes a no-op — dropping
        /// reasoning is an *encoder* decision here, not a pump fork.
        var emitReasoning: (String) -> Void
        /// The resolved terminal tool call (free-detected or Guide-forced).
        var emitToolCall: (_ name: String, _ argsJSON: String) -> Void
        /// C2 terminal logprobs list — OpenAI only (Anthropic passes nil).
        var emitLogprobs: (([TokenLogprob]) -> Void)?
        /// In-stream error event (HTTP 200 already sent).
        var emitError: ((AthenaError) -> Void)?
        /// Terminal: `reason` = generator/stop-latched finish, `toolCalled` = a
        /// tool block was emitted, `stopHit` = the stop sequence that actually
        /// matched (nil if none), `usage` = final counts.
        var finish:
            (_ reason: FinishReason, _ toolCalled: Bool, _ stopHit: String?,
                _ usage: TokenUsage) -> Void
    }

    /// ADR 036 WP7 — the single `GenChunk` traversal both dialects' streaming
    /// pumps share. Owns stop-sequence latching, reasoning peel, tool-call
    /// buffering, and finish/stop attribution; the dialect-specific wire shape
    /// is the injected `ProtocolEncoder`. The terminal tool precedence goes
    /// through the shared `resolveToolCallOutcome` (the same algebra the two
    /// non-streaming encoders switch on). Returns final usage so the caller
    /// records + closes the transport.
    @discardableResult
    static func foldGenChunks(
        events: AsyncStream<GenChunk>, stops: [String], isToolCall: Bool,
        into sink: ProtocolEncoder
    ) async -> TokenUsage {
        var usage = TokenUsage.zero
        var finish: FinishReason = .stop
        var stopHit: String?
        var toolBuffer = ""
        var freeToolCall: (name: String, argsJSON: String)?
        var stopFilter = StopStreamFilter(stops: stops)
        var reasoningFilter = ReasoningChannelFilter()
        // Route a content piece through the stop filter (latching `stop` + the
        // matched sequence) or emit it directly when no stops are set.
        func pushContent(_ piece: String) {
            if stopFilter.isActive {
                let wasStopped = stopFilter.stopped
                sink.emitText(stopFilter.push(piece))
                if stopFilter.stopped && !wasStopped {
                    finish = .stop
                    stopHit = stopFilter.matchedStop
                }
            } else {
                sink.emitText(piece)
            }
        }
        for await event in events {
            switch event {
            case .text(let piece):
                if isToolCall {
                    // Guide-constrained to one JSON object — buffer + parse at
                    // the end so no raw tool JSON leaks into content.
                    toolBuffer += piece
                } else {
                    let s = reasoningFilter.push(piece)
                    sink.emitReasoning(s.reasoning)
                    pushContent(s.content)
                }
            case .usage(let u):
                usage = u
            case .toolCall(let name, let argsJSON):
                // ADR 034 — free tool call. Buffer it; `resolveToolCallOutcome`
                // emits it at the terminal (uniform with the non-stream paths).
                freeToolCall = (name, argsJSON)
            case .finish(let r):
                // A stop-sequence hit wins over the generator's own reason.
                if !stopFilter.stopped { finish = r }
            case .logprobs(let l):
                sink.emitLogprobs?(l)
            case .error(let e):
                sink.emitError?(e)
                finish = .stop
            }
        }
        // ADR 035 — flush any held reasoning/content tail, then the stop tail.
        if !isToolCall {
            let s = reasoningFilter.flush()
            sink.emitReasoning(s.reasoning)
            pushContent(s.content)
        }
        if stopFilter.isActive && !stopFilter.stopped {
            sink.emitText(stopFilter.flush())
        }
        var toolCalled = false
        switch resolveToolCallOutcome(
            detected: freeToolCall, text: toolBuffer, isToolCall: isToolCall)
        {
        case .detected(let n, let a), .forced(let n, let a):
            sink.emitToolCall(n, a)
            toolCalled = true
        case .none:
            // Guide-forced output that didn't parse (e.g. truncated by
            // max_tokens): surface the raw buffer as text rather than drop it.
            if isToolCall && !toolBuffer.isEmpty { sink.emitText(toolBuffer) }
        }
        sink.finish(finish, toolCalled, stopHit, usage)
        return usage
    }

    /// Shared GenChunk→SSE pump: emit the assistant role chunk, stream content
    /// deltas (stop-sequence filtered), then the terminal finish/usage chunks
    /// and `[DONE]`, recording usage. Factored out of `streamSSE` so the
    /// load-awaiting variant reuses the exact wire shape (ADR 015). The
    /// OpenAI-shaped encoding is a `ProtocolEncoder` over the shared
    /// `foldGenChunks` (ADR 036 WP7).
    private static func pumpTokens(
        into continuation: AsyncStream<ByteBuffer>.Continuation,
        id: String, model: String, created: Int,
        events: AsyncStream<GenChunk>, includeUsage: Bool,
        isToolCall: Bool = false,
        stops: [String],
        record: @escaping @Sendable (TokenUsage) async -> Void
    ) async {
        func emit(_ chunk: ChatCompletionChunk) {
            if let data = try? JSONEncoder().encode(chunk) {
                var buf = ByteBuffer()
                buf.writeString("data: ")
                buf.writeBytes(data)
                buf.writeString("\n\n")
                continuation.yield(buf)
            }
        }
        func chunk(_ delta: ChatDelta, logprobs: ChatLogprobs? = nil)
            -> ChatCompletionChunk
        {
            ChatCompletionChunk(
                id: id, object: "chat.completion.chunk", created: created,
                model: model,
                choices: [
                    ChatChunkChoice(
                        index: 0, delta: delta, finish_reason: nil,
                        logprobs: logprobs)
                ])
        }
        // Assistant role preamble.
        emit(chunk(ChatDelta(role: "assistant", content: "")))

        let encoder = ProtocolEncoder(
            emitText: { piece in
                guard !piece.isEmpty else { return }
                emit(chunk(ChatDelta(role: nil, content: piece)))
            },
            emitReasoning: { piece in
                guard !piece.isEmpty else { return }
                emit(
                    chunk(
                        ChatDelta(
                            role: nil, content: nil, reasoning_content: piece)))
            },
            // One `delta.tool_calls` chunk (v0.10.230 shape).
            emitToolCall: { name, argsJSON in
                emit(
                    chunk(
                        ChatDelta(
                            role: nil, content: nil,
                            tool_calls: [
                                ToolCallDelta(
                                    index: 0,
                                    id: "call_\(UUID().uuidString.prefix(8))",
                                    type: "function",
                                    function: FunctionCallOut(
                                        name: name, arguments: argsJSON))
                            ])))
            },
            // C2 — one chunk carrying the OpenAI logprobs object (empty delta).
            emitLogprobs: { l in
                emit(
                    chunk(
                        ChatDelta(role: nil, content: nil),
                        logprobs: Self.chatLogprobs(l)))
            },
            // M49.5.2 — an in-stream OpenAI-style error event (status already sent).
            emitError: { athenaErr in
                let body = APIErrorBody(
                    error: .init(
                        message: athenaErr.message, type: "server_error",
                        code: athenaErr.code))
                if let data = try? JSONEncoder().encode(body) {
                    var buf = ByteBuffer()
                    buf.writeString("data: ")
                    buf.writeBytes(data)
                    buf.writeString("\n\n")
                    continuation.yield(buf)
                }
            },
            finish: { reason, toolCalled, _, usage in
                // M31.2 — `length` at max_tokens, `stop`/`tool_calls` otherwise.
                emit(
                    ChatCompletionChunk(
                        id: id, object: "chat.completion.chunk",
                        created: created, model: model,
                        choices: [
                            ChatChunkChoice(
                                index: 0,
                                delta: ChatDelta(role: nil, content: nil),
                                finish_reason: toolCalled
                                    ? "tool_calls" : reason.rawValue)
                        ]))
                // OpenAI emits usage in a final empty-choices chunk, opt-in only.
                if includeUsage {
                    emit(
                        ChatCompletionChunk(
                            id: id, object: "chat.completion.chunk",
                            created: created, model: model, choices: [],
                            usage: Usage(
                                prompt_tokens: usage.promptTokens,
                                completion_tokens: usage.completionTokens,
                                total_tokens: usage.totalTokens,
                                cachedTokens: usage.cachedTokens)))
                }
                var done = ByteBuffer()
                done.writeString("data: [DONE]\n\n")
                continuation.yield(done)
            })

        let usage = await foldGenChunks(
            events: events, stops: stops, isToolCall: isToolCall, into: encoder)
        await record(usage)
        continuation.finish()
    }

    /// ADR 036 S2 — the Anthropic streaming counterpart of `streamSSE`: a
    /// `text/event-stream` Response whose producer is `pumpAnthropic`. Mirrors
    /// the warm-stream cancel wiring (a client disconnect cancels the decode).
    private static func streamAnthropic(
        id: String, model: String,
        events: AsyncStream<GenChunk>, isToolCall: Bool, stops: [String],
        onConsumerCancel: (@Sendable () -> Void)? = nil,
        record: @escaping @Sendable (TokenUsage) async -> Void
    ) -> Response {
        let stream = AsyncStream<ByteBuffer> { continuation in
            let task = Task {
                await pumpAnthropic(
                    into: continuation, id: id, model: model, events: events,
                    isToolCall: isToolCall, stops: stops, record: record)
            }
            continuation.onTermination = { _ in
                task.cancel()
                onConsumerCancel?()
            }
        }
        var headers = HTTPFields()
        headers[.contentType] = "text/event-stream"
        headers[.cacheControl] = "no-cache"
        return Response(
            status: .ok, headers: headers,
            body: ResponseBody(asyncSequence: stream))
    }

    /// ADR 036 S2 — the Anthropic event-stream pump: consume the SAME
    /// `GenChunk` stream the OpenAI pump does, emit the Anthropic event
    /// sequence (message_start → content_block_start/delta/stop → message_delta
    /// → message_stop). Text streams as `text_delta`; a tool call (free-detected
    /// or Guide-forced) is one `tool_use` block with a single `input_json_delta`
    /// carrying the args. Reasoning (ADR 035 `<|channel>`) is peeled and dropped
    /// (first cut: no `thinking` blocks). Stop-sequence + reasoning filtering
    /// reuse the exact `StopStreamFilter`/`ReasoningChannelFilter` the OpenAI
    /// pump uses, so detection can't drift between the two dialects.
    private static func pumpAnthropic(
        into continuation: AsyncStream<ByteBuffer>.Continuation,
        id: String, model: String,
        events: AsyncStream<GenChunk>, isToolCall: Bool, stops: [String],
        record: @escaping @Sendable (TokenUsage) async -> Void
    ) async {
        func emit<T: Encodable>(_ eventName: String, _ payload: T) {
            guard let data = try? JSONEncoder().encode(payload) else { return }
            var buf = ByteBuffer()
            buf.writeString("event: \(eventName)\ndata: ")
            buf.writeBytes(data)
            buf.writeString("\n\n")
            continuation.yield(buf)
        }
        emit(
            "message_start",
            AnthropicStreamStart(
                message: .init(
                    id: id, model: model,
                    usage: AnthropicUsage(input_tokens: 0, output_tokens: 0))))

        // Content-block bookkeeping — the one piece of dialect state the encoder
        // closures share (text block opened lazily on first delta, closed before
        // a tool_use block or the terminal).
        var index = 0
        var textOpen = false
        func openText() {
            guard !textOpen else { return }
            emit(
                "content_block_start",
                AnthropicBlockStart(
                    index: index,
                    content_block: AnthropicResponseBlock(
                        type: "text", text: "")))
            textOpen = true
        }
        func closeText() {
            guard textOpen else { return }
            emit("content_block_stop", AnthropicBlockStop(index: index))
            textOpen = false
            index += 1
        }

        let encoder = ProtocolEncoder(
            emitText: { s in
                guard !s.isEmpty else { return }
                openText()
                emit(
                    "content_block_delta",
                    AnthropicTextDelta(index: index, delta: .init(text: s)))
            },
            // ADR 036 first cut: reasoning is dropped — an ENCODER decision, not
            // a pump fork (surface as thinking-blocks here if a consumer asks).
            emitReasoning: { _ in },
            emitToolCall: { name, argsJSON in
                closeText()
                let input: JSONValue =
                    (argsJSON.data(using: .utf8).flatMap {
                        try? JSONDecoder().decode(JSONValue.self, from: $0)
                    }) ?? .object([:])
                emit(
                    "content_block_start",
                    AnthropicBlockStart(
                        index: index,
                        content_block: AnthropicResponseBlock(
                            type: "tool_use", id: "toolu_\(UUID().uuidString)",
                            name: name, input: .object([:]))))
                emit(
                    "content_block_delta",
                    AnthropicInputJSONDelta(
                        index: index,
                        delta: .init(partial_json: input.jsonString() ?? "{}")))
                emit("content_block_stop", AnthropicBlockStop(index: index))
                index += 1
            },
            emitLogprobs: nil,  // no Anthropic equivalent
            emitError: { e in
                emit(
                    "error",
                    AnthropicErrorBody(
                        error: .init(type: "api_error", message: e.message)))
            },
            finish: { reason, toolCalled, stopHit, usage in
                closeText()
                let stopReason: String
                if toolCalled {
                    stopReason = "tool_use"
                } else if stopHit != nil {
                    stopReason = "stop_sequence"
                } else if reason == .length {
                    stopReason = "max_tokens"
                } else {
                    stopReason = "end_turn"
                }
                emit(
                    "message_delta",
                    AnthropicMessageDelta(
                        delta: .init(
                            stop_reason: stopReason, stop_sequence: stopHit),
                        usage: AnthropicUsage(
                            input_tokens: usage.promptTokens,
                            output_tokens: usage.completionTokens)))
                emit("message_stop", AnthropicMessageStop())
            })

        let usage = await foldGenChunks(
            events: events, stops: stops, isToolCall: isToolCall, into: encoder)
        await record(usage)
        continuation.finish()
    }

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

    private static func error(
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
    private static func tooLargeResponse(cap: Int) -> Response {
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
    private static func payloadTooLarge(_ request: Request, cap: Int)
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
    private static func coldLoadResponse(_ id: ModuleID) -> Response {
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
    private static let log = Logger(label: AthenaLog.daemonLabel)
    private static let auditLog = Logger(label: AthenaLogLabel.audit)

    /// WP5 (audit P3) — an RBAC-admin store operation (putUser / grantRole /
    /// putToken) failed. Log the raw detail (SQLite message / constraint) to
    /// os_log but return only a stable, detail-free message to the client,
    /// mirroring the `classified` suppression boundary the inference paths use.
    /// The four admin sites previously returned `"\(error)"` verbatim.
    private static func storeError(_ err: any Error) -> Response {
        log.warning("admin store operation failed: \(err)")
        return error(
            status: .internalServerError,
            message: "internal store error",
            type: "server_error", code: "store_error")
    }

    private static func classified(
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
