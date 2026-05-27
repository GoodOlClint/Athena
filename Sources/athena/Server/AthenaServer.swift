import AthenaCore
import AthenaDeploy
import AthenaEmbedding
import AthenaLLM
import AthenaStore
import AthenaStructured
import AthenaTranscription
import Foundation
import HTTPTypes
import Hummingbird
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
    let vectorStore: VectorStore
    let queue: RequestQueue
    /// Shared SQLite store (vectors + queue) — backs `/v1/store/*`.
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
    /// Warm the LLM at startup instead of lazily on first request
    /// (M33.3). `var = false` so it's a memberwise-init param. Best-
    /// effort: the warm runs concurrently with serving (the HTTP surface
    /// is up immediately) and a failure falls back to the lazy path.
    var preload: Bool = false
    /// Queue-result retention (M34.1). `queueResultTtlSecs` > 0 ⇒ prune
    /// terminal (done/error/canceled) results older than that window;
    /// `queueMaxRows` > 0 ⇒ cap total job rows (oldest terminal first).
    /// Both 0 ⇒ keep forever (opt-in). Swept on the worker idle path so
    /// inference outputs don't accumulate on disk unbounded. `var = 0`
    /// so they're memberwise-init params.
    var queueResultTtlSecs: Int = 0
    var queueMaxRows: Int = 0
    /// Vector-store retention (M34.2). > 0 ⇒ on each upsert, prune
    /// persisted vectors whose last-write time is older than the window
    /// (prune-on-write, like the M30 audit prune — a daemon actively
    /// writing vectors keeps itself bounded; an idle one accrues
    /// nothing). 0 ⇒ keep forever (opt-in). `var = 0` so it's a
    /// memberwise-init param.
    var vectorTtlSecs: Int = 0
    /// Content opt-out (M34.2). When true, the queue clears a job's
    /// persisted `request` (prompt) blob once it reaches a terminal
    /// state, so inference INPUTS don't sit on disk after the answer is
    /// produced (the result the client polls for stays, bounded by the
    /// queue TTL). `var = false` so it's a memberwise-init param.
    var dropRequestContent: Bool = false
    /// WebUI session signer (M12.2). Per-process random secret —
    /// sessions invalidate on restart (acceptable for an appliance).
    let session = Session()

    func run() async throws {
        let router = Router()
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

        router.get("/healthz") { [metrics, queue] _, _ -> Response in
            let snapshot = await governor.snapshot()
            let (inflight, lastAt) = await metrics.healthFields()
            let depth = await queue.depth()
            return Self.json(
                HealthResponse(
                    snapshot: snapshot,
                    inflight: inflight,
                    queueDepth: depth,
                    lastRequestAt: lastAt))
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
        // enqueueModelOp) — no self-HTTP, no duplication. All static
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
        router.post("/ui/api/models/pull") { request, _
            -> Response in
            await uiModelMutate(request) { r in
                await self.enqueueModelOp(
                    kind: "model_pull", r
                ) { d in
                    guard
                        let x = try? JSONDecoder().decode(
                            ModelPullRequest.self, from: d),
                        !x.id.isEmpty
                    else {
                        return "model_pull requires non-empty 'id'"
                    }
                    return nil
                }
            }
        }
        router.post("/ui/api/models/convert") { request, _
            -> Response in
            await uiModelMutate(request) { r in
                await self.enqueueModelOp(
                    kind: "model_convert", r
                ) { d in
                    guard
                        let x = try? JSONDecoder().decode(
                            ModelConvertRequest.self, from: d),
                        !x.id.isEmpty
                    else {
                        return
                            "model_convert requires non-empty 'id'"
                    }
                    return nil
                }
            }
        }
        router.post("/ui/api/models/prune") { request, _
            -> Response in
            await uiModelMutate(request) { r in
                await self.enqueueModelOp(
                    kind: "model_prune", r) { _ in nil }
            }
        }
        router.get("/ui/api/job") { request, _ -> Response in
            await uiJobStatus(request)
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

        // Allowlist console (M44.1). Cookie + per-action model.read
        // (list) / model.write (mutations) + CSRF, then REUSE the
        // M42.2 `/api/models/allow` handlers (handleAllowlist*) so the
        // server-side refresh-on-mutate path is unchanged. All static
        // literals (module/id ride the JSON body) ⇒ no trie hazard.
        router.get("/ui/allowlist") { request, _ -> Response in
            await handleUIAllowlistPage(request)
        }
        router.get("/ui/api/allowlist") { request, _ -> Response in
            await uiAllowlistList(request)
        }
        router.post("/ui/api/allowlist") { request, _ -> Response in
            await uiAllowlistAdd(request)
        }
        router.post("/ui/api/allowlist/rm") { request, _
            -> Response in
            await uiAllowlistRm(request)
        }
        router.post("/ui/api/allowlist/default") { request, _
            -> Response in
            await uiAllowlistDefault(request)
        }

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
        router.post("/ui/login") { request, _ -> Response in
            await handleUILoginPost(request)
        }
        router.get("/ui/logout") { _, _ -> Response in
            Self.logoutResponse()
        }
        router.post("/ui/logout") { _, _ -> Response in
            Self.logoutResponse()
        }

        router.post("/v1/chat/completions") { request, _ -> Response in
            await handleChatCompletions(request)
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

        // Built-in vector DB (M7.2).
        router.post("/v1/vectors") { request, _ -> Response in
            await handleVectorUpsert(request)
        }
        router.post("/v1/vectors/query") { request, _ -> Response in
            await handleVectorQuery(request)
        }
        router.get("/v1/vectors/stats") { _, _ -> Response in
            let st = await vectorStore.stats()
            return Self.json(
                VectorStatsResponse(
                    count: st.count, dim: st.dim, bytes: st.bytes,
                    cap_bytes: st.capBytes))
        }
        router.delete("/v1/vectors/:id") { _, context -> Response in
            await handleVectorDelete(context.parameters.get("id"))
        }

        // Shared SQLite store admin (M9.3). Export = a live, consistent
        // snapshot via VACUUM INTO (safe while serving).
        router.post("/v1/store/export") { request, _ -> Response in
            await handleStoreExport(request)
        }
        router.get("/v1/store/stats") { _, _ -> Response in
            await handleStoreStats()
        }

        // Async request queue (M8.1).
        // Same path node ⇒ one shared param name across methods
        // (Hummingbird's trie binds the name per position).
        router.post("/v1/queue/:arg") { request, context -> Response in
            await handleQueueSubmit(
                context.parameters.get("arg"), request)
        }
        router.get("/v1/queue") { request, _ -> Response in
            await handleQueueList(request)
        }
        router.get("/v1/queue/:arg") { request, context -> Response in
            await handleQueueStatus(
                context.parameters.get("arg"), request)
        }
        router.delete("/v1/queue/:arg") { request, context -> Response in
            await handleQueueRemove(
                context.parameters.get("arg"), request)
        }
        // SSE: stream status transitions until terminal (inbound only —
        // the passive-oracle thesis forbids outbound webhooks).
        router.get("/v1/queue/:arg/events") { request, context
            -> Response in
            await handleQueueEvents(
                context.parameters.get("arg"), request)
        }

        // Athena-native API (M16). `/v1/*` stays OpenAI-compatible;
        // `/api/*` is Athena's OWN dialect (clean minimal JSON, NOT
        // Ollama, NOT OpenAI), routed through the SAME governed module
        // paths. Ollama was deleted entirely (no /api/version, /tags,
        // /generate, /embeddings).
        router.post("/api/chat") { request, _ -> Response in
            await handleNativeChat(request)
        }
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
        // Long-running ops → enqueued (model.write-gated route);
        // returns {job_id}, poll GET /v1/queue/:job_id.
        router.post("/api/models/pull") { request, _ -> Response in
            await enqueueModelOp(
                kind: "model_pull", request
            ) { d in
                guard
                    let r = try? JSONDecoder().decode(
                        ModelPullRequest.self, from: d),
                    !r.id.isEmpty
                else { return "model_pull requires non-empty 'id'" }
                return nil
            }
        }
        router.post("/api/models/convert") { request, _ -> Response in
            await enqueueModelOp(
                kind: "model_convert", request
            ) { d in
                guard
                    let r = try? JSONDecoder().decode(
                        ModelConvertRequest.self, from: d),
                    !r.id.isEmpty
                else {
                    return "model_convert requires non-empty 'id'"
                }
                return nil
            }
        }
        router.post("/api/models/prune") { request, _ -> Response in
            await enqueueModelOp(kind: "model_prune", request) {
                _ in nil
            }
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
        // Persistent operator-declared allowlist (M42.2). Backs every
        // module's selectable set; survives daemon restarts. GET =
        // model.read, mutations = model.write (per AuthPolicy mapping
        // of /api/models/*). Default sub-path PUT clears the prior
        // default + marks the named row.
        router.get("/api/models/allow") { request, _ -> Response in
            await handleAllowlistList(request)
        }
        router.post("/api/models/allow") { request, _ -> Response in
            await handleAllowlistAdd(request)
        }
        router.delete("/api/models/allow") { request, _ -> Response in
            await handleAllowlistRemove(request)
        }
        router.put("/api/models/allow/default") {
            request, _ -> Response in
            await handleAllowlistSetDefault(request)
        }
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

        // Wire the queue executor to the governed module paths (M8.1).
        // The serial worker runs as a managed Service (below) so it
        // drains on graceful shutdown rather than being cancelled mid-job.
        await queue.setExecutor { kind, data, owner in
            await self.queuedExecute(
                kind: kind, request: data, owner: owner)
        }

        // M34.1: bound retained queue results so inference outputs don't
        // persist forever. Swept on the worker idle path (startup + each
        // time it drains empty). 0/0 ⇒ keep forever (opt-in).
        await queue.setRetention(
            ttlSecs: queueResultTtlSecs, maxRows: queueMaxRows,
            dropRequestContent: dropRequestContent)

        // M33.3: optionally warm the LLM at startup so the first request
        // doesn't pay the load latency. Best-effort and concurrent — the
        // HTTP surface (below) still comes up immediately; a failed warm
        // logs and leaves the lazy path intact. `governor.ensureLoaded`
        // is idempotent, so a request racing the warm is safe.
        if preload {
            let governor = self.governor
            Task {
                let log = Logger(label: AthenaLogLabel.daemon)
                log.notice("preload: warming LLM at startup")
                do {
                    try await governor.ensureLoaded(.llm)
                    log.notice("preload: LLM warm")
                } catch {
                    log.warning(
                        "preload: LLM warm failed, will load lazily: \(error)"
                    )
                }
            }
        }

        // M33.2: register the queue worker in the application's
        // ServiceGroup. `runService` installs SIGTERM/SIGINT graceful
        // shutdown; on signal the HTTP server drains its in-flight
        // requests and the worker finishes its in-flight job, both within
        // the stop window — no abrupt mid-request/mid-job teardown.
        let app = Application(
            router: router,
            server: try Self.serverBuilder(
                tlsCertPath: tlsCertPath, tlsKeyPath: tlsKeyPath),
            configuration: .init(
                address: .hostname(
                    config.listenHost, port: config.listenPort),
                serverName: "athena"
            ),
            services: [QueueWorkerService(queue: queue)]
        )
        try await app.runService()
    }

    /// Build the HTTP(S) listener. Both cert+key ⇒ TLS; neither ⇒
    /// plaintext HTTP; exactly one ⇒ a hard error (fail-closed — never
    /// silently fall back to plaintext when TLS was half-configured).
    /// PEM load failures (missing/unreadable/malformed) propagate from
    /// NIOSSL and abort daemon start with a clear error.
    static func serverBuilder(
        tlsCertPath: String?, tlsKeyPath: String?
    ) throws -> HTTPServerBuilder {
        switch (tlsCertPath, tlsKeyPath) {
        case (nil, nil):
            return .http1()
        case (let cert?, let key?):
            let chain = try NIOSSLCertificate.fromPEMFile(cert)
                .map { NIOSSLCertificateSource.certificate($0) }
            let pkey = try NIOSSLPrivateKey(file: key, format: .pem)
            let tls = TLSConfiguration.makeServerConfiguration(
                certificateChain: chain,
                privateKey: .privateKey(pkey))
            Logger(label: AthenaLogLabel.daemon).notice(
                "TLS: serving HTTPS (cert \(cert))")
            return try .tls(.http1(), tlsConfiguration: tls)
        case (.some, nil), (nil, .some):
            throw TLSConfigError.incomplete
        }
    }

    private func handleChatCompletions(_ request: Request) async -> Response {
        let body: ChatCompletionRequest
        do {
            let buffer = try await request.body.collect(upTo: 4 * 1024 * 1024)
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
        if let err = await governedLLM(
            request: request, requestedModel: body.model)
        {
            return err
        }
        let model = await servedLLMModel()
        // M24.1: carry the FULL conversation (system/user/assistant/tool)
        // into the model, not a user-only join — system instructions and
        // prior turns must reach the chat template. A message with no text
        // content (e.g. an assistant tool-call shell) carries nothing.
        let turns = body.messages.compactMap { m -> ChatTurn? in
            guard let c = m.content else { return nil }
            return ChatTurn(role: m.role, content: c)
        }
        // Brief 4b: refuse an over-cap prompt up front as a governed
        // 503, before any KV cache is allocated.
        do {
            try await llm.preflightPromptCache(messages: turns)
        } catch let e as AthenaError {
            return Self.error(
                status: HTTPResponse.Status(code: e.httpStatus),
                message: e.message, type: "server_error", code: e.code)
        } catch {
            return Self.classified(error, module: .llm)
        }

        let created = Int(Date().timeIntervalSince1970)
        let id = "chatcmpl-\(UUID().uuidString)"
        let effective = body.effectiveSchema()
        let schemaJSON = effective?.json
        let toolSpecs = body.toolSpecs()

        let stops = body.stopSequences()

        if body.stream == true {
            // M27.4: meter streamed requests too, and emit a terminal
            // usage chunk when the client opted in via stream_options.
            let principal = await usagePrincipal(request)
            let includeUsage = body.stream_options?.include_usage == true
            return Self.streamSSE(
                id: id, model: model, created: created,
                events: deadlineBounded(
                    seconds: requestTimeoutSecs,
                    llm.generateMetered(
                        messages: turns, schemaJSON: schemaJSON,
                        tools: toolSpecs, maxTokens: body.max_tokens,
                        temperature: body.temperature,
                        topP: body.top_p, seed: body.seed,
                        speculative: body.speculative)),
                includeUsage: includeUsage, stops: stops,
                record: { usage in
                    await meter(principal: principal, usage: usage)
                })
        }

        let collected: GenCollected
        do {
            collected = try await collectMetered(
                llm.generateMetered(
                    messages: turns, schemaJSON: schemaJSON,
                    tools: toolSpecs, maxTokens: body.max_tokens,
                    temperature: body.temperature,
                    topP: body.top_p, seed: body.seed,
                    speculative: body.speculative))
        } catch let e as AthenaError {
            // M33.1: the only AthenaError collectMetered raises is the
            // per-request timeout → classified 504.
            return Self.error(
                status: HTTPResponse.Status(code: e.httpStatus),
                message: e.message, type: "server_error", code: e.code)
        } catch {
            return Self.classified(error, module: .llm)
        }
        var text = collected.text
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
        await meter(
            principal: usagePrincipal(request), usage: usage)

        return Self.json(
            Self.chatCompletionResponse(
                id: id, model: model, created: created, text: text,
                isToolCall: effective?.isToolCall == true, usage: usage,
                finish: finish))
    }

    /// Build one `ChatChoice` from generated text: a tool-call object is
    /// surfaced as OpenAI `tool_calls`; everything else as `content`.
    /// Shared by the sync `/v1/chat/completions` handler and the queued
    /// `conversation` executor so both emit the identical OpenAI shape.
    /// `finish` is the generator's stop reason (M31.2): a real tool call
    /// always reports `tool_calls`; otherwise the reason passes through
    /// (`stop` natural end, `length` max_tokens truncation).
    private static func chatChoice(
        text: String, isToolCall: Bool, finish: FinishReason
    ) -> ChatChoice {
        if isToolCall,
            let data = text.data(using: .utf8),
            let obj = try? JSONSerialization.jsonObject(with: data)
                as? [String: Any],
            let name = obj["name"] as? String
        {
            let args = obj["arguments"] ?? [String: Any]()
            let argsJSON =
                (try? JSONSerialization.data(
                    withJSONObject: args, options: [.sortedKeys]))
                .flatMap { String(data: $0, encoding: .utf8) } ?? "{}"
            return ChatChoice(
                index: 0,
                message: ChatMessage(
                    role: "assistant", content: nil,
                    tool_calls: [
                        ToolCallOut(
                            id: "call_\(UUID().uuidString.prefix(8))",
                            type: "function",
                            function: FunctionCallOut(
                                name: name, arguments: argsJSON))
                    ]),
                finish_reason: "tool_calls")
        }
        return ChatChoice(
            index: 0,
            message: ChatMessage(role: "assistant", content: text),
            finish_reason: finish.rawValue)
    }

    /// Assemble a full OpenAI `ChatCompletionResponse` around one choice.
    private static func chatCompletionResponse(
        id: String, model: String, created: Int, text: String,
        isToolCall: Bool, usage: TokenUsage,
        finish: FinishReason = .stop
    ) -> ChatCompletionResponse {
        ChatCompletionResponse(
            id: id, object: "chat.completion", created: created,
            model: model,
            choices: [
                chatChoice(
                    text: text, isToolCall: isToolCall, finish: finish)
            ],
            usage: Usage(
                prompt_tokens: usage.promptTokens,
                completion_tokens: usage.completionTokens,
                total_tokens: usage.totalTokens))
    }

    /// Accumulated result of draining a metered generation: the full
    /// text, the true token usage, and the finish reason.
    struct GenCollected: Sendable {
        var text = ""
        var usage = TokenUsage.zero
        var finish: FinishReason = .stop
    }

    /// Drain a metered generation under the per-request deadline (M33.1)
    /// and return its text + usage + finish reason. `requestTimeoutSecs`
    /// = 0 ⇒ unbounded. On overrun it throws
    /// `AthenaError.requestTimedOut` (the caller maps it to a 504) and the
    /// generation is cancelled so it stops consuming the worker/budget.
    /// Shared by the sync `/v1/chat/completions`, native `/api/chat`, and
    /// queued `conversation` paths so all three honor the same timeout.
    private func collectMetered(
        _ events: AsyncStream<GenChunk>
    ) async throws -> GenCollected {
        try await withInferenceDeadline(seconds: requestTimeoutSecs) {
            var c = GenCollected()
            for await event in events {
                switch event {
                case .text(let chunk): c.text += chunk
                case .usage(let u): c.usage = u
                case .finish(let r): c.finish = r
                }
            }
            return c
        }
    }

    private func handleEmbeddings(_ request: Request) async -> Response {
        let body: EmbeddingRequest
        do {
            let buffer = try await request.body.collect(upTo: 4 * 1024 * 1024)
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
            switch try await governor.beginLoadIfNeeded(.textEmbedding) {
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
        } catch let e as AthenaError {
            return Self.error(
                status: HTTPResponse.Status(code: e.httpStatus),
                message: e.message, type: "server_error", code: e.code)
        } catch {
            return Self.error(
                status: .internalServerError,
                message: String(describing: error),
                type: "server_error", code: "internal_error")
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
        return Self.json(response)
    }

    private func handleTranscriptions(_ request: Request) async -> Response
    {
        guard
            let ct = request.headers[.contentType],
            let boundary = MultipartForm.boundary(fromContentType: ct)
        else {
            return Self.error(
                status: .badRequest,
                message: "expected multipart/form-data with a boundary",
                type: "invalid_request_error", code: "invalid_content_type")
        }

        let body: Data
        do {
            let buffer = try await request.body.collect(
                upTo: 25 * 1024 * 1024)  // OpenAI's 25 MB audio cap
            body = Data(buffer: buffer)
        } catch {
            return Self.error(
                status: .badRequest,
                message: "Invalid request body: \(error)",
                type: "invalid_request_error", code: "invalid_body")
        }

        guard
            let form = MultipartForm(body: body, boundary: boundary),
            let file = form.first("file"), !file.data.isEmpty
        else {
            return Self.error(
                status: .badRequest,
                message: "missing required 'file' part",
                type: "invalid_request_error", code: "missing_file")
        }

        do {
            // M43.2: non-blocking cold-load (see /v1/embeddings).
            switch try await governor.beginLoadIfNeeded(.transcription) {
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
        } catch let e as AthenaError {
            return Self.error(
                status: HTTPResponse.Status(code: e.httpStatus),
                message: e.message, type: "server_error", code: e.code)
        } catch {
            return Self.error(
                status: .internalServerError,
                message: String(describing: error),
                type: "server_error", code: "internal_error")
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
                do {
                    // M43.2: non-blocking cold-load.
                    switch try await governor.beginLoadIfNeeded(.diarization) {
                    case .loaded: break
                    case .loading: return Self.coldLoadResponse(.diarization)
                    }
                    turns = try await diarization.diarize(
                        audio: file.data, filename: file.filename
                    ).turns
                } catch {
                    return Self.classified(error, module: .diarization)
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
        default:  // "json" / nil
            return Self.json(TranscriptionResponse(text: result.text))
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

    private func handleDiarizations(_ request: Request) async -> Response
    {
        guard
            let ct = request.headers[.contentType],
            let boundary = MultipartForm.boundary(fromContentType: ct)
        else {
            return Self.error(
                status: .badRequest,
                message: "expected multipart/form-data with a boundary",
                type: "invalid_request_error",
                code: "invalid_content_type")
        }
        let body: Data
        do {
            let buffer = try await request.body.collect(
                upTo: 25 * 1024 * 1024)
            body = Data(buffer: buffer)
        } catch {
            return Self.error(
                status: .badRequest,
                message: "Invalid request body: \(error)",
                type: "invalid_request_error", code: "invalid_body")
        }
        guard
            let form = MultipartForm(body: body, boundary: boundary),
            let file = form.first("file"), !file.data.isEmpty
        else {
            return Self.error(
                status: .badRequest,
                message: "missing required 'file' part",
                type: "invalid_request_error", code: "missing_file")
        }

        // Method select: default Sortformer (fast, ≤4 speakers); opt into
        // the embedding+clustering diarizer (M25.3) for >4 speakers with
        // `method=cluster` (+ optional num_speakers / max_speakers /
        // threshold).
        if form.text("method") == "cluster" {
            return await handleClusterDiarization(file: file, form: form)
        }

        do {
            // M43.2: non-blocking cold-load.
            switch try await governor.beginLoadIfNeeded(.diarization) {
            case .loaded: break
            case .loading: return Self.coldLoadResponse(.diarization)
            }
            // M41.3 per-request diarization model selection;
            // M41.4 audited on a real resident-id change.
            if let m = form.text("model"), !m.isEmpty {
                try await auditedRebind(
                    request, module: .diarization, target: m)
            }
        } catch let e as AthenaError {
            return Self.error(
                status: HTTPResponse.Status(code: e.httpStatus),
                message: e.message, type: "server_error", code: e.code)
        } catch {
            return Self.error(
                status: .internalServerError,
                message: String(describing: error),
                type: "server_error", code: "internal_error")
        }

        let r: DiarizationResult
        do {
            r = try await diarization.diarize(
                audio: file.data, filename: file.filename)
        } catch {
            return Self.classified(error, module: .diarization)
        }
        return Self.json(
            DiarizationResponse(
                num_speakers: r.numSpeakers,
                segments: r.turns.map {
                    DiarizationSegmentDTO(
                        start: $0.start, end: $0.end,
                        speaker: $0.speaker)
                }))
    }

    /// M25.3 embedding+clustering diarizer: window → WeSpeaker embed →
    /// agglomerative cluster → merge same-speaker windows into turns.
    /// Recovers >4 speakers, which the offline Sortformer cannot.
    private func handleClusterDiarization(
        file: MultipartForm.Part, form: MultipartForm
    ) async -> Response {
        let numSpeakers = form.text("num_speakers").flatMap(Int.init)
        let maxSpeakers = form.text("max_speakers").flatMap(Int.init)
        let threshold = form.text("threshold").flatMap(Float.init) ?? 0.75

        do {
            // M43.2: non-blocking cold-load.
            switch try await governor.beginLoadIfNeeded(.speakerEmbedding) {
            case .loaded: break
            case .loading: return Self.coldLoadResponse(.speakerEmbedding)
            }
        } catch let e as AthenaError {
            return Self.error(
                status: HTTPResponse.Status(code: e.httpStatus),
                message: e.message, type: "server_error", code: e.code)
        } catch {
            return Self.error(
                status: .internalServerError,
                message: String(describing: error),
                type: "server_error", code: "internal_error")
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
        guard
            let ct = request.headers[.contentType],
            let boundary = MultipartForm.boundary(fromContentType: ct)
        else {
            return Self.error(
                status: .badRequest,
                message: "expected multipart/form-data with a boundary",
                type: "invalid_request_error",
                code: "invalid_content_type")
        }
        let body: Data
        do {
            let buffer = try await request.body.collect(
                upTo: 25 * 1024 * 1024)
            body = Data(buffer: buffer)
        } catch {
            return Self.error(
                status: .badRequest,
                message: "Invalid request body: \(error)",
                type: "invalid_request_error", code: "invalid_body")
        }
        guard
            let form = MultipartForm(body: body, boundary: boundary),
            let file = form.first("file"), !file.data.isEmpty
        else {
            return Self.error(
                status: .badRequest,
                message: "missing required 'file' part",
                type: "invalid_request_error", code: "missing_file")
        }

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
            switch try await governor.beginLoadIfNeeded(.speakerEmbedding) {
            case .loaded: break
            case .loading: return Self.coldLoadResponse(.speakerEmbedding)
            }
            // M41.3 per-request speaker-embedding model selection;
            // M41.4 audited on a real resident-id change.
            if let m = form.text("model"), !m.isEmpty {
                try await auditedRebind(
                    request, module: .speakerEmbedding, target: m)
            }
        } catch let e as AthenaError {
            return Self.error(
                status: HTTPResponse.Status(code: e.httpStatus),
                message: e.message, type: "server_error", code: e.code)
        } catch {
            return Self.error(
                status: .internalServerError,
                message: String(describing: error),
                type: "server_error", code: "internal_error")
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

    // MARK: - Built-in vector DB (M7.2)

    /// Resolve a request's vector: explicit `vector`, else embed
    /// `text` via the governed embedding module.
    private func resolveVector(
        _ vector: [Float]?, _ text: String?
    ) async -> Outcome<[Float]> {
        if let vector { return .ok(vector) }
        guard let text, !text.isEmpty else {
            return .fail(
                Self.error(
                    status: .badRequest,
                    message: "provide 'vector' or non-empty 'text'",
                    type: "invalid_request_error",
                    code: "missing_vector"))
        }
        switch await governedEmbed([text], module: .textEmbedding) {
        case .fail(let r): return .fail(r)
        case .ok(let batch):
            return .ok(batch.vectors.first ?? [])
        }
    }

    private static func vectorErrorResponse(
        _ error: any Error
    ) -> Response {
        if let e = error as? VectorStore.VectorError {
            switch e {
            case .capExceeded:
                return Self.error(
                    status: .serviceUnavailable, message: e.description,
                    type: "server_error",
                    code: "vector_store_cap_exceeded")
            case .dimMismatch:
                return Self.error(
                    status: .badRequest, message: e.description,
                    type: "invalid_request_error",
                    code: "dimension_mismatch")
            }
        }
        return Self.classified(error, module: .textEmbedding)
    }

    private func handleVectorUpsert(_ request: Request) async
        -> Response
    {
        let decoded = await decodeJSON(
            request, VectorUpsertRequest.self)
        guard case .ok(let body) = decoded else {
            if case .fail(let r) = decoded { return r }
            fatalError()
        }
        let vec: [Float]
        switch await resolveVector(body.vector, body.text) {
        case .fail(let r): return r
        case .ok(let v): vec = v
        }
        let meta = body.metadata.flatMap { try? JSONEncoder().encode($0) }
        do {
            try await vectorStore.upsert(
                id: body.id, vector: vec, metadata: meta)
        } catch {
            return Self.vectorErrorResponse(error)
        }
        // M34.2: opportunistic age-based retention (prune-on-write).
        // 0 ⇒ keep forever. Non-fatal — never fails the upsert.
        if vectorTtlSecs > 0 {
            let cutoff = Date().timeIntervalSince1970
                - Double(vectorTtlSecs)
            let removed = await vectorStore.sweepExpired(olderThan: cutoff)
            if removed > 0 {
                Logger(label: AthenaLogLabel.daemon).notice(
                    "vector retention: pruned \(removed) vector(s) older than \(vectorTtlSecs)s")
            }
        }
        return Self.json(VectorIdResponse(id: body.id))
    }

    private func handleVectorQuery(_ request: Request) async -> Response
    {
        let decoded = await decodeJSON(request, VectorQueryRequest.self)
        guard case .ok(let body) = decoded else {
            if case .fail(let r) = decoded { return r }
            fatalError()
        }
        let vec: [Float]
        switch await resolveVector(body.vector, body.text) {
        case .fail(let r): return r
        case .ok(let v): vec = v
        }
        let hits = await vectorStore.query(
            vector: vec, k: body.k ?? 5)
        return Self.json(
            VectorQueryResponse(
                matches: hits.map {
                    VectorMatch(
                        id: $0.id, score: $0.score,
                        metadata: $0.metadata.flatMap {
                            try? JSONDecoder().decode(
                                JSONValue.self, from: $0)
                        })
                }))
    }

    private func handleVectorDelete(_ id: String?) async -> Response {
        guard let id, !id.isEmpty else {
            return Self.error(
                status: .badRequest, message: "missing vector id",
                type: "invalid_request_error", code: "missing_id")
        }
        let ok = await vectorStore.delete(id: id)
        if !ok {
            return Self.error(
                status: .notFound, message: "no vector '\(id)'",
                type: "invalid_request_error", code: "not_found")
        }
        return Self.json(VectorIdResponse(id: id))
    }

    // MARK: - Shared store admin (M9.3)

    private static func fileBytes(_ url: URL) -> Int {
        let attrs = try? FileManager.default.attributesOfItem(
            atPath: url.path)
        return (attrs?[.size] as? Int) ?? 0
    }

    /// On-disk footprint = main DB + the WAL/SHM sidecars (a hot DB's
    /// recent writes live in `-wal` until checkpoint).
    func storeBytes() -> Int {
        let base = store.dbPath
        return Self.fileBytes(base)
            + Self.fileBytes(base.appendingPathExtension("wal"))
            + Self.fileBytes(base.appendingPathExtension("shm"))
    }

    private func handleStoreExport(_ request: Request) async -> Response {
        let decoded = await decodeJSON(request, StoreExportRequest.self)
        guard case .ok(let body) = decoded else {
            if case .fail(let r) = decoded { return r }
            fatalError()
        }
        let raw = (body.path as NSString).expandingTildeInPath
        guard !raw.isEmpty else {
            return Self.error(
                status: .badRequest, message: "missing export path",
                type: "invalid_request_error", code: "missing_path")
        }
        let dest = URL(fileURLWithPath: raw)
        do {
            try await store.backup(to: dest)
        } catch {
            return Self.error(
                status: .internalServerError,
                message: "export failed: \(error)",
                type: "server_error", code: "export_failed")
        }
        return Self.json(
            StoreExportResponse(
                path: dest.path, bytes: Self.fileBytes(dest)))
    }

    private func handleStoreStats() async -> Response {
        let vectors = await store.vectorCount()
        let jobs = await store.jobCount()
        let bytes = storeBytes()
        let path = store.dbPath.path
        return Self.json(
            StoreStatsResponse(
                vectors: vectors, jobs: jobs, bytes: bytes, path: path))
    }

    // MARK: - Async request queue (M8.1)

    /// Per-submitter queue authorization (M12.6 → M15.2). `enforced`
    /// is auth being on; `principal` identifies the bearer's owning
    /// subject (`u:<user>` for a managed token, `t:<hash>` for a
    /// bootstrap key — AuthMiddleware already gated `.queueSubmit`).
    /// `isAdmin` (sees every tenant's jobs) = the caller holds the
    /// full permission set, i.e. the `admin` role.
    private func queuePrincipal(_ request: Request) async -> (
        principal: String?, isAdmin: Bool, enforced: Bool
    ) {
        guard auth.isEnabled else { return (nil, false, false) }
        guard
            let h = request.headers[.authorization],
            h.hasPrefix("Bearer "),
            case let tok = String(h.dropFirst(7)), !tok.isEmpty,
            let subject = await auth.resolve(bearer: tok)
        else { return (nil, false, true) }
        let isAdmin = Set(Permission.allCases).isSubset(
            of: subject.permissions)
        return (subject.principal, isAdmin, true)
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

    /// May this caller see/act on `job`? Open when auth is off;
    /// admin sees all; a nil-owner row is legacy/unowned (pre-M12.6,
    /// back-compat); otherwise only the submitting principal.
    private static func canAccess(
        _ job: JobRow, principal: String?, isAdmin: Bool,
        enforced: Bool
    ) -> Bool {
        if !enforced || isAdmin { return true }
        guard let owner = job.owner else { return true }
        return owner == principal
    }

    private func handleQueueSubmit(
        _ kind: String?, _ request: Request
    ) async -> Response {
        guard let kind, RequestQueue.publicKinds.contains(kind) else {
            return Self.error(
                status: .badRequest,
                message:
                    "unknown queue kind; expected one of "
                    + RequestQueue.publicKinds.sorted().joined(
                        separator: ", "),
                type: "invalid_request_error", code: "invalid_kind")
        }
        let body: Data
        do {
            let buf = try await request.body.collect(
                upTo: 8 * 1024 * 1024)
            body = Data(buffer: buf)
        } catch {
            return Self.error(
                status: .badRequest,
                message: "Invalid request body: \(error)",
                type: "invalid_request_error", code: "invalid_body")
        }
        let who = await queuePrincipal(request)
        do {
            let id = try await queue.submit(
                kind: kind, request: body,
                owner: who.enforced ? who.principal : nil)
            return Self.json(
                QueueSubmitResponse(id: id, status: "queued"))
        } catch {
            return Self.error(
                status: .internalServerError,
                message: "queue submit failed: \(error)",
                type: "server_error", code: "queue_error")
        }
    }

    private static func isTerminal(_ s: String) -> Bool {
        s == "done" || s == "error" || s == "canceled"
    }

    private static func statusResponse(_ job: JobRow) -> Response {
        let result = job.result.flatMap {
            try? JSONDecoder().decode(JSONValue.self, from: $0)
        }
        return json(
            QueueStatusResponse(
                id: job.id, kind: job.kind, status: job.status,
                result: result, error: job.error))
    }

    /// `?wait=N` long-polls up to N s (clamped 0…120) for a terminal
    /// state before responding — inbound-only, no outbound callback.
    private func handleQueueStatus(
        _ id: String?, _ request: Request
    ) async -> Response {
        guard let id else {
            return Self.error(
                status: .notFound, message: "no job ''",
                type: "invalid_request_error", code: "not_found")
        }
        var wait = 0
        if let q = request.uri.query {
            for kv in q.split(separator: "&")
            where kv.hasPrefix("wait=") {
                wait = min(120, max(0, Int(kv.dropFirst(5)) ?? 0))
            }
        }
        let who = await queuePrincipal(request)
        let deadline = Date().addingTimeInterval(Double(wait))
        while true {
            guard let job = await queue.status(id: id),
                Self.canAccess(
                    job, principal: who.principal,
                    isAdmin: who.isAdmin, enforced: who.enforced)
            else {
                // 404 (not 403) — don't reveal another tenant's job.
                return Self.error(
                    status: .notFound, message: "no job '\(id)'",
                    type: "invalid_request_error", code: "not_found")
            }
            if Self.isTerminal(job.status) || Date() >= deadline {
                return Self.statusResponse(job)
            }
            try? await Task.sleep(nanoseconds: 250_000_000)
        }
    }

    private func handleQueueEvents(
        _ id: String?, _ request: Request
    ) async -> Response {
        let who = await queuePrincipal(request)
        if let id, let job = await queue.status(id: id),
            !Self.canAccess(
                job, principal: who.principal, isAdmin: who.isAdmin,
                enforced: who.enforced)
        {
            return Self.error(
                status: .notFound, message: "no job '\(id)'",
                type: "invalid_request_error", code: "not_found")
        }
        let q = queue
        let stream = AsyncStream<ByteBuffer> { continuation in
            let task = Task {
                func emit(_ job: JobRow) {
                    let r = job.result.flatMap {
                        try? JSONDecoder().decode(
                            JSONValue.self, from: $0)
                    }
                    guard
                        let data = try? JSONEncoder().encode(
                            QueueStatusResponse(
                                id: job.id, kind: job.kind,
                                status: job.status, result: r,
                                error: job.error))
                    else { return }
                    var b = ByteBuffer()
                    b.writeString("data: ")
                    b.writeBytes(data)
                    b.writeString("\n\n")
                    continuation.yield(b)
                }
                var last = ""
                // Cap the stream so a stuck job can't hold a
                // connection forever (~600 × 0.5 s = 5 min).
                for _ in 0..<600 {
                    guard let id, let job = await q.status(id: id)
                    else { break }
                    if job.status != last {
                        emit(job)
                        last = job.status
                    }
                    if Self.isTerminal(job.status) { break }
                    try? await Task.sleep(nanoseconds: 500_000_000)
                }
                var done = ByteBuffer()
                done.writeString("data: [DONE]\n\n")
                continuation.yield(done)
                continuation.finish()
            }
            continuation.onTermination = { _ in task.cancel() }
        }
        var headers = HTTPFields()
        headers[.contentType] = "text/event-stream"
        headers[.cacheControl] = "no-cache"
        return Response(
            status: .ok, headers: headers,
            body: ResponseBody(asyncSequence: stream))
    }

    private func handleQueueRemove(
        _ id: String?, _ request: Request
    ) async -> Response {
        guard let id else {
            return Self.error(
                status: .badRequest, message: "missing job id",
                type: "invalid_request_error", code: "missing_id")
        }
        // Only the owner (or admin) may remove — and a non-owner
        // gets 404, not 403, so job existence stays hidden.
        let who = await queuePrincipal(request)
        if let job = await queue.status(id: id),
            !Self.canAccess(
                job, principal: who.principal, isAdmin: who.isAdmin,
                enforced: who.enforced)
        {
            return Self.error(
                status: .notFound, message: "no job '\(id)'",
                type: "invalid_request_error", code: "not_found")
        }
        let removed = await queue.remove(id: id)
        if !removed {
            return Self.error(
                status: .notFound, message: "no job '\(id)'",
                type: "invalid_request_error", code: "not_found")
        }
        return Self.json(
            QueueRemoveResponse(id: id, removed: true))
    }

    /// `GET /v1/queue[?status=...]` — job summaries, oldest first.
    private func handleQueueList(_ request: Request) async -> Response {
        var status: String?
        if let qy = request.uri.query {
            for kv in qy.split(separator: "&")
            where kv.hasPrefix("status=") {
                status = String(kv.dropFirst(7))
            }
        }
        let who = await queuePrincipal(request)
        let jobs = await queue.list(status: status).filter {
            Self.canAccess(
                $0, principal: who.principal, isAdmin: who.isAdmin,
                enforced: who.enforced) && (
                    !who.enforced || who.isAdmin
                        || $0.owner == who.principal)
        }
        return Self.json(
            QueueListResponse(
                jobs: jobs.map {
                    QueueJobSummary(
                        id: $0.id, kind: $0.kind, status: $0.status,
                        created: $0.created, updated: $0.updated)
                }))
    }

    /// Runs a queued job through the same governed paths as the sync
    /// endpoints. Returns (resultJSON, nil) or (nil, errorMessage).
    /// `owner` is the submitting principal (M12.6) so token usage is
    /// metered against the same principal a sync request would be
    /// (M27.2); nil ⇒ auth disabled, metered under the `xenos` sentinel.
    private func queuedExecute(
        kind: String, request: Data, owner: String?
    ) async -> (result: Data?, error: String?) {
        switch kind {
        case "conversation":
            // M24.2: decode the OpenAI `ChatCompletionRequest` (a superset
            // of the native chat body — a `{messages}` job still decodes)
            // so the queued path honors `response_format`/`tools` the SAME
            // way the sync `/v1/chat/completions` handler does. Without
            // this, queued generation ran unconstrained and structured
            // jobs came back as free text.
            guard
                let req = try? JSONDecoder().decode(
                    ChatCompletionRequest.self, from: request)
            else { return (nil, "invalid conversation body") }
            let turns = req.messages.compactMap { m -> ChatTurn? in
                guard let c = m.content else { return nil }
                return ChatTurn(role: m.role, content: c)
            }
            do {
                try await governor.ensureLoaded(.llm)
                // M41.2: a queued job selects the LLM model the same
                // way the sync chat handler does — rebind under the
                // governor before the preflight + decode.
                if let m = req.model, !m.isEmpty {
                    try await selectable(.llm).rebind(to: m)
                }
                try await llm.preflightPromptCache(messages: turns)
            } catch let e as AthenaError {
                return (nil, e.message)
            } catch {
                return (nil, String(describing: error))
            }
            let effective = req.effectiveSchema()
            let collected: GenCollected
            do {
                collected = try await collectMetered(
                    llm.generateMetered(
                        messages: turns, schemaJSON: effective?.json,
                        tools: req.toolSpecs(), maxTokens: req.max_tokens,
                        temperature: req.temperature,
                        topP: req.top_p, seed: req.seed,
                        speculative: req.speculative))
            } catch let e as AthenaError {
                return (nil, e.message)  // M33.1: timeout → job error
            } catch {
                return (nil, String(describing: error))
            }
            var text = collected.text
            let usage = collected.usage
            var finish = collected.finish
            let stops = req.stopSequences()
            if !stops.isEmpty {
                let cut = StopStreamFilter.truncate(text, stops: stops)
                if cut.stopped {
                    text = cut.text
                    finish = .stop
                }
            }
            await meter(principal: owner, usage: usage)
            // M24.6: store the full OpenAI ChatCompletionResponse as the
            // job result so a polled queued job carries the SAME
            // `choices[0].message.{content,tool_calls}` shape as the sync
            // endpoint — one result envelope across sync and async.
            // M27.1/.2: the envelope carries real `usage` too. M41.2: the
            // `model` field reports the LLM actually served (post-rebind),
            // not the unused request echo.
            let servedModel = await servedLLMModel()
            return (
                try? JSONEncoder().encode(
                    Self.chatCompletionResponse(
                        id: "chatcmpl-\(UUID().uuidString)",
                        model: servedModel,
                        created: Int(Date().timeIntervalSince1970),
                        text: text,
                        isToolCall: effective?.isToolCall == true,
                        usage: usage, finish: finish)), nil
            )
        case "embeddings":
            guard
                let req = try? JSONDecoder().decode(
                    AthenaEmbedRequest.self, from: request)
            else { return (nil, "invalid embeddings body") }
            do {
                try await governor.ensureLoaded(.textEmbedding)
                // M39: select per request; report the served model in the
                // stored result (an unknown id fails the job, not a silent
                // wrong-dimension fallback).
                let batch = try await embedding.embed(
                    req.input, model: req.model)
                await meter(
                    principal: owner,
                    usage: TokenUsage(
                        promptTokens: batch.promptTokens,
                        completionTokens: 0))
                return (
                    try? JSONEncoder().encode(
                        QueuedEmbeddingResult(
                            model: batch.model,
                            embeddings: batch.vectors)),
                    nil
                )
            } catch let e as AthenaError {
                return (nil, e.message)
            } catch {
                return (nil, String(describing: error))
            }
        case "model_pull":
            // Egress (proxy + HF token) was exported process-wide at
            // daemon startup (Load); the #hubDownloader reads it. Only
            // sanctioned model-fetch egress — passive oracle intact.
            guard
                let req = try? JSONDecoder().decode(
                    ModelPullRequest.self, from: request)
            else { return (nil, "invalid model_pull body") }
            do {
                let dest = try await ModelPull.pull(
                    id: req.id, revision: req.revision,
                    into: modelStoreRoot)
                return (
                    try? JSONEncoder().encode(
                        QueuedModelPullResult(
                            name: dest.lastPathComponent,
                            path: dest.path)), nil)
            } catch {
                return (nil, "pull failed: \(error)")
            }
        case "model_convert":
            guard
                let req = try? JSONDecoder().decode(
                    ModelConvertRequest.self, from: request)
            else { return (nil, "invalid model_convert body") }
            do {
                // M-conv: `bits` is opt-in (mlx_lm-style). Omit ⇒ no
                // quantization; explicit N ⇒ quantize to N-bit.
                let r = try await ModelConvert.convert(
                    id: req.id, revision: req.revision,
                    bits: req.bits,
                    groupSize: req.group_size ?? 64,
                    into: modelStoreRoot, name: req.name)
                return (
                    try? JSONEncoder().encode(
                        QueuedModelConvertResult(
                            path: r.path.path, bytes: r.bytes)), nil)
            } catch {
                return (nil, "convert failed: \(error)")
            }
        case "model_prune":
            let req =
                (try? JSONDecoder().decode(
                    ModelPruneRequest.self, from: request))
                ?? ModelPruneRequest(dry_run: false)
            do {
                let pr = try ModelStoreOps.prune(
                    root: modelStoreRoot,
                    dryRun: req.dry_run ?? false)
                return (
                    try? JSONEncoder().encode(
                        QueuedModelPruneResult(
                            candidates: pr.victims.map { $0.name },
                            removed: pr.removed, dry_run: pr.dryRun)),
                    nil)
            } catch {
                return (nil, "prune failed: \(error)")
            }
        default:
            return (nil, "unknown kind '\(kind)'")
        }
    }

    // MARK: - Native /api inference (M16)

    /// `Response` isn't `Error`, so a plain success-or-error-response.
    private enum Outcome<T> {
        case ok(T)
        case fail(Response)
    }

    private func decodeJSON<T: Decodable>(
        _ request: Request, _ type: T.Type
    ) async -> Outcome<T> {
        do {
            let buf = try await request.body.collect(upTo: 4 * 1024 * 1024)
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

    /// Governed gate for `/api/chat`: ensureLoaded(.llm) + (M41.2)
    /// optional per-request rebind to `requestedModel` + the 4b
    /// prompt-cache preflight, all classified. Returns an error
    /// `Response` to send, or nil when the request may proceed.
    private func governedPreflight(
        messages: [ChatTurn], requestedModel: String? = nil,
        request: Request? = nil
    ) async -> Response? {
        if let err = await governedLLM(
            request: request, requestedModel: requestedModel)
        {
            return err
        }
        do {
            try await llm.preflightPromptCache(messages: messages)
            return nil
        } catch let e as AthenaError {
            return Self.error(
                status: HTTPResponse.Status(code: e.httpStatus),
                message: e.message, type: "server_error", code: e.code)
        } catch {
            return Self.classified(error, module: .llm)
        }
    }

    /// Newline-delimited JSON streamer. Each generated piece → one
    /// JSON line; a final `done` line closes the stream.
    private static func streamNDJSON(
        tokens: AsyncStream<String>,
        line: @escaping @Sendable (_ content: String, _ done: Bool)
            -> Data?
    ) -> Response {
        let stream = AsyncStream<ByteBuffer> { continuation in
            let task = Task {
                func emit(_ d: Data?) {
                    guard let d else { return }
                    var b = ByteBuffer()
                    b.writeBytes(d)
                    b.writeString("\n")
                    continuation.yield(b)
                }
                for await piece in tokens { emit(line(piece, false)) }
                emit(line("", true))
                continuation.finish()
            }
            continuation.onTermination = { _ in task.cancel() }
        }
        var headers = HTTPFields()
        headers[.contentType] = "application/x-ndjson"
        return Response(
            status: .ok, headers: headers,
            body: ResponseBody(asyncSequence: stream))
    }

    /// `POST /api/chat` — Athena-native chat. Same governed LLM path as
    /// `/v1/chat/completions`; only the JSON dialect differs. Non-
    /// stream → one `AthenaChatResponse`; `stream:true` → NDJSON
    /// `AthenaChatChunk` lines, final `{content:"",done:true}`.
    private func handleNativeChat(_ request: Request) async -> Response {
        let decoded = await decodeJSON(request, AthenaChatRequest.self)
        guard case .ok(let body) = decoded else {
            if case .fail(let r) = decoded { return r }
            fatalError()
        }
        // M24.1: full conversation reaches the model (system + prior
        // turns), not a user-only join. M41.2: body.model selects the
        // LLM slot's resident model from the allowlist (governedPreflight
        // does the rebind under the governor).
        let turns = body.messages.map {
            ChatTurn(role: $0.role, content: $0.content)
        }
        if let err = await governedPreflight(
            messages: turns, requestedModel: body.model,
            request: request)
        {
            return err
        }
        let model = await servedLLMModel()
        if body.stream == true {
            return Self.streamNDJSON(
                tokens: deadlineBounded(
                    seconds: requestTimeoutSecs,
                    llm.generate(
                        messages: turns, schemaJSON: nil, tools: nil,
                        maxTokens: body.max_tokens,
                        temperature: body.temperature,
                        speculative: body.speculative))
            ) { content, done in
                try? JSONEncoder().encode(
                    AthenaChatChunk(
                        content: done ? "" : content, done: done))
            }
        }
        let stream = llm.generate(
            messages: turns, schemaJSON: nil, tools: nil,
            maxTokens: body.max_tokens, temperature: body.temperature,
            speculative: body.speculative)
        let text: String
        do {
            text = try await withInferenceDeadline(
                seconds: requestTimeoutSecs
            ) {
                var acc = ""
                for await c in stream { acc += c }
                return acc
            }
        } catch let e as AthenaError {
            return Self.error(
                status: HTTPResponse.Status(code: e.httpStatus),
                message: e.message, type: "server_error", code: e.code)
        } catch {
            return Self.classified(error, module: .llm)
        }
        return Self.json(
            AthenaChatResponse(
                model: model, content: text, done: true))
    }

    /// Governed embedding helper shared by `/api/embed`, `/v1/vectors`
    /// text resolution, and the queued `embeddings` kind. Returns the
    /// whole batch so callers can echo the model ACTUALLY served (M39).
    /// `model` selects among the configured set (nil ⇒ default); an
    /// unknown id surfaces as a classified 400 `model_not_available`.
    private func governedEmbed(
        _ inputs: [String], module: ModuleID, model: String? = nil
    ) async -> Outcome<EmbeddingBatch> {
        do {
            // M43.2: non-blocking cold-load — surfaces as
            // `503 module_loading` to the native /api/embed caller too.
            switch try await governor.beginLoadIfNeeded(.textEmbedding) {
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
            if case .fail(let r) = decoded { return r }
            fatalError()
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
            return Self.json(
                AthenaEmbedResponse(
                    model: batch.model, embeddings: batch.vectors))
        }
    }

    // MARK: - Model store (M16.2)

    private static func iso(_ d: Date) -> String {
        ISO8601DateFormatter().string(from: d)
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
        // `show` does not carry mtime; read it back from `list` (cheap —
        // a directory scan) so `created` matches the list projection.
        let created =
            ModelStoreOps.list(root: modelStoreRoot)
            .first { $0.name == d.name }?.modified ?? Date()
        return Self.json(Self.openAIModel(d.name, created))
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
            if case .fail(let r) = decoded { return r }
            fatalError()
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
            if case .fail(let r) = decoded { return r }
            fatalError()
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
            // M43.2: non-blocking cold-load — covers /v1/chat/completions
            // AND /api/chat (the brief's biggest hang vector).
            switch try await governor.beginLoadIfNeeded(.llm) {
            case .loaded: break
            case .loading: return Self.coldLoadResponse(.llm)
            }
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
    /// non-HTTP-driven callers (queue worker) can skip the audit (the
    /// queue owner is recorded on the JobRow itself).
    private func auditedRebind(
        _ request: Request?, module: ModuleID, target: String
    ) async throws {
        let sel = selectable(module)
        let before = await sel.residentModelId()
        try await sel.rebind(to: target)
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
            if case .fail(let r) = decoded { return r }
            fatalError()
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
        let target = body.id ?? def
        guard allowed.contains(target) else {
            await audit(
                request, action: "model.load",
                target: "\(moduleId.rawValue):\(target)",
                result: "denied", detail: "id outside allowlist")
            return Self.error(
                status: .badRequest,
                message:
                    "Model '\(target)' is not available. Configured "
                    + "models for \(moduleId.rawValue): "
                    + "\(allowed.joined(separator: ", ")).",
                type: "invalid_request_error",
                code: "model_not_available")
        }
        do {
            try await governor.ensureLoaded(moduleId)
            try await sel.rebind(to: target)
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
            if case .fail(let r) = decoded { return r }
            fatalError()
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

    // MARK: - Persistent model allowlist (M42.2)

    /// `GET /api/models/allow` (M42.2) — all rows, or just `?module=M`.
    /// Backs `athena allowlist list`; the on-disk allowlist is what
    /// every daemon-start now resolves from (M42.1 boot-seed +
    /// `/api/models/allow` mutations from this surface).
    private func handleAllowlistList(_ request: Request) async
        -> Response
    {
        var moduleFilter: String?
        if let q = request.uri.query {
            for kv in q.split(separator: "&")
            where kv.hasPrefix("module=") {
                let v = String(kv.dropFirst(7))
                moduleFilter = v.isEmpty ? nil : v
            }
        }
        if let m = moduleFilter, ModuleID(rawValue: m) == nil {
            return Self.error(
                status: .badRequest,
                message: "unknown module '\(m)'",
                type: "invalid_request_error", code: "invalid_module")
        }
        let rows = await store.listModelAllowlist(module: moduleFilter)
        return Self.json(
            AllowlistResponse(
                allowlist: rows.map {
                    AllowlistEntryDTO(
                        module: $0.module, id: $0.id,
                        default: $0.isDefault,
                        declared: $0.declared)
                }))
    }

    /// `POST /api/models/allow` (M42.2) — add `id` to a module's
    /// allowlist (idempotent on duplicate). `default: true` marks the
    /// new row as the module's default. Audited (M30). M42.3 also
    /// pushes the updated set into the running module so the next
    /// inference request sees it without a restart.
    private func handleAllowlistAdd(_ request: Request) async
        -> Response
    {
        let decoded = await decodeJSON(request, AddAllowlistRequest.self)
        guard case .ok(let body) = decoded else {
            if case .fail(let r) = decoded { return r }
            fatalError()
        }
        guard
            let moduleId = ModuleID(rawValue: body.module),
            !body.id.isEmpty
        else {
            await audit(
                request, action: "model.allow.add",
                target: body.module + ":" + body.id,
                result: "denied",
                detail: "invalid module or empty id")
            return Self.error(
                status: .badRequest,
                message:
                    "module must be a ModuleID and id must be non-empty",
                type: "invalid_request_error", code: "invalid_body")
        }
        do {
            try await store.addModelAllowlist(
                module: moduleId.rawValue, id: body.id,
                isDefault: body.default ?? false)
        } catch {
            await audit(
                request, action: "model.allow.add",
                target: moduleId.rawValue + ":" + body.id,
                result: "denied", detail: "\(error)")
            return Self.error(
                status: .internalServerError, message: "\(error)",
                type: "server_error", code: "store_error")
        }
        await refreshAllowlist(module: moduleId)
        await audit(
            request, action: "model.allow.add",
            target: moduleId.rawValue + ":" + body.id,
            result: "ok",
            detail: (body.default ?? false) ? "default=true" : nil)
        return Self.json(
            AllowlistMutationResponse(
                module: moduleId.rawValue, id: body.id,
                status: "added"))
    }

    /// `DELETE /api/models/allow?module=M&id=X` (M42.2). Removing the
    /// current default rotates the default to whichever row remains
    /// first by declaration time (the module's `resolveAllowlist`
    /// ordering at next refresh). Removing the only remaining row in a
    /// module leaves that slot un-servable until another `add` (the
    /// next inference returns 400 model_not_available — explicit
    /// failure, the safe direction).
    private func handleAllowlistRemove(_ request: Request) async
        -> Response
    {
        var moduleS: String?
        var idS: String?
        if let q = request.uri.query {
            for kv in q.split(separator: "&") {
                let p = kv.split(separator: "=", maxSplits: 1)
                guard p.count == 2 else { continue }
                let raw = String(p[1])
                let v = raw.removingPercentEncoding ?? raw
                switch String(p[0]) {
                case "module": moduleS = v
                case "id": idS = v
                default: break
                }
            }
        }
        guard
            let m = moduleS, let id = idS,
            !m.isEmpty, !id.isEmpty,
            let moduleId = ModuleID(rawValue: m)
        else {
            return Self.error(
                status: .badRequest,
                message: "expected ?module=M&id=X with a known module",
                type: "invalid_request_error", code: "invalid_query")
        }
        return await removeAllowlistRow(
            request, moduleId: moduleId, id: id)
    }

    /// Shared store-write + refresh + audit path used by the bearer
    /// `DELETE /api/models/allow` (query-string) and the cookie
    /// `POST /ui/api/allowlist/rm` (JSON body). One implementation so
    /// the audit + live-refresh contract can't drift between the two.
    private func removeAllowlistRow(
        _ request: Request, moduleId: ModuleID, id: String
    ) async -> Response {
        let removed = await store.removeModelAllowlist(
            module: moduleId.rawValue, id: id)
        if !removed {
            return Self.error(
                status: .notFound,
                message: "no \(moduleId.rawValue):\(id) in allowlist",
                type: "invalid_request_error", code: "not_found")
        }
        await refreshAllowlist(module: moduleId)
        await audit(
            request, action: "model.allow.rm",
            target: moduleId.rawValue + ":" + id, result: "ok")
        return Self.json(
            AllowlistMutationResponse(
                module: moduleId.rawValue, id: id, status: "removed"))
    }

    /// `PUT /api/models/allow/default` (M42.2) — mark `id` as `module`'s
    /// default (clearing the prior default). The id must already be in
    /// the allowlist; this only flips the flag, never adds.
    private func handleAllowlistSetDefault(_ request: Request) async
        -> Response
    {
        let decoded = await decodeJSON(
            request, SetAllowlistDefaultRequest.self)
        guard case .ok(let body) = decoded else {
            if case .fail(let r) = decoded { return r }
            fatalError()
        }
        guard
            let moduleId = ModuleID(rawValue: body.module),
            !body.id.isEmpty
        else {
            return Self.error(
                status: .badRequest,
                message:
                    "module must be a ModuleID and id must be non-empty",
                type: "invalid_request_error", code: "invalid_body")
        }
        do {
            try await store.setModelAllowlistDefault(
                module: moduleId.rawValue, id: body.id)
        } catch {
            await audit(
                request, action: "model.allow.default",
                target: moduleId.rawValue + ":" + body.id,
                result: "denied", detail: "\(error)")
            return Self.error(
                status: .badRequest, message: "\(error)",
                type: "invalid_request_error",
                code: "model_not_available")
        }
        await refreshAllowlist(module: moduleId)
        await audit(
            request, action: "model.allow.default",
            target: moduleId.rawValue + ":" + body.id, result: "ok")
        return Self.json(
            AllowlistMutationResponse(
                module: moduleId.rawValue, id: body.id,
                status: "default_set"))
    }

    /// M42.3 — push the just-mutated DB allowlist into the running
    /// module so its next request validates against the new set
    /// without a daemon restart. Default-first ordering matches what
    /// `resolveAllowlist` does at boot.
    ///
    /// M43.1 — every module's `setAllowedModelIds` nils its container
    /// directly when the resident id drops out of the new list. That
    /// bypasses the governor's bookkeeping, so the slot's reservation +
    /// state would otherwise stay stale (the lying-`/healthz` symptom).
    /// After the live push, ask the module for `residentBytes` — 0 means
    /// the container was just dropped — and reconcile the governor.
    private func refreshAllowlist(module: ModuleID) async {
        let rows = await store.listModelAllowlist(
            module: module.rawValue)
        let ordered: [String]
        if rows.isEmpty {
            ordered = []
        } else {
            let def = rows.first { $0.isDefault } ?? rows[0]
            var arr = [def.id]
            for r in rows where r.id != def.id { arr.append(r.id) }
            ordered = arr
        }
        await selectable(module).setAllowedModelIds(ordered)
        let bytes = await inferenceModule(module).residentBytes
        if bytes == 0 {
            await governor.releaseSlot(module)
        }
    }

    /// `any InferenceModule` corresponding to `id`. Parallel to
    /// `selectable(_:)` — used where the governor-side residentBytes
    /// probe is needed (M43.1 allowlist-drop reconcile).
    private func inferenceModule(_ id: ModuleID) -> any InferenceModule {
        switch id {
        case .llm: return llm
        case .textEmbedding: return embedding
        case .transcription: return transcription
        case .diarization: return diarization
        case .speakerEmbedding: return speakerEmbedding
        }
    }

    /// Enqueue a long-running model op (M16.3). The route is already
    /// `.modelWrite`-gated; the job is owner-scoped to the caller so
    /// `GET /v1/queue/:id` only reveals it to the submitter (or an
    /// admin). `validate` returns an error message for a 400, or nil.
    private func enqueueModelOp(
        kind: String, _ request: Request,
        validate: (Data) -> String?
    ) async -> Response {
        let body: Data
        do {
            let buf = try await request.body.collect(
                upTo: 1 * 1024 * 1024)
            body = Data(buffer: buf)
        } catch {
            return Self.error(
                status: .badRequest,
                message: "Invalid request body: \(error)",
                type: "invalid_request_error", code: "invalid_body")
        }
        if let msg = validate(body) {
            return Self.error(
                status: .badRequest, message: msg,
                type: "invalid_request_error", code: "invalid_body")
        }
        let who = await queuePrincipal(request)
        do {
            let id = try await queue.submit(
                kind: kind, request: body,
                owner: who.enforced ? who.principal : nil)
            return Self.json(
                ModelJobResponse(job_id: id, status: "queued"),
                status: .accepted)
        } catch {
            return Self.error(
                status: .internalServerError,
                message: "queue submit failed: \(error)",
                type: "server_error", code: "queue_error")
        }
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
    /// the shared M16 op (ModelStoreOps / enqueueModelOp). Mutations
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
                if case .fail(let f) = decoded { return f }
                fatalError()
            }
            return await self.handleModelRemove(body.name, req)
        }
    }

    /// Console job-progress poll. /ui ops are submitted with no
    /// bearer ⇒ nil-owner (console-scoped, visible to any console
    /// admin — matches the legacy unowned-job semantics);
    /// `.modelRead` gates the peek. Reuses the queue + M16 status
    /// DTO (`statusResponse`).
    private func uiJobStatus(_ r: Request) async -> Response {
        guard await uiCaller(r).perms.contains(.modelRead) else {
            return Self.uiDeny("need model.read")
        }
        guard let id = Self.uiQuery(r, "id"),
            let job = await queue.status(id: id)
        else {
            return Self.error(
                status: .notFound, message: "no such job",
                type: "invalid_request_error", code: "not_found")
        }
        return Self.statusResponse(job)
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
                admins: await store.usersWithRole("admin").count))
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

    // MARK: - WebUI allowlist reuse (M44.1)

    /// `/ui/api/allowlist` (cookie). Mirrors `handleAllowlistList` —
    /// `.modelRead` re-check on the logged-in user, then the same
    /// list+JSON path.
    private func uiAllowlistList(_ r: Request) async -> Response {
        guard await uiCaller(r).perms.contains(.modelRead) else {
            return Self.uiDeny("need model.read")
        }
        return await handleAllowlistList(r)
    }

    /// `/ui/api/allowlist` POST (cookie). CSRF + `.modelWrite`, then
    /// delegates to `handleAllowlistAdd` — same JSON body shape, same
    /// audit + refreshAllowlist tail.
    private func uiAllowlistAdd(_ r: Request) async -> Response {
        guard csrfOK(r) else {
            return Self.uiDeny("csrf token missing or invalid")
        }
        guard await uiCaller(r).perms.contains(.modelWrite) else {
            return Self.uiDeny("need model.write")
        }
        return await handleAllowlistAdd(r)
    }

    /// `/ui/api/allowlist/rm` POST (cookie). The bearer endpoint reads
    /// `?module=&id=` from the query string; the cookie one reads them
    /// from the JSON body so the WebUI can POST them like every other
    /// /ui mutation. Both end at `removeAllowlistRow`.
    private func uiAllowlistRm(_ r: Request) async -> Response {
        guard csrfOK(r) else {
            return Self.uiDeny("csrf token missing or invalid")
        }
        guard await uiCaller(r).perms.contains(.modelWrite) else {
            return Self.uiDeny("need model.write")
        }
        let decoded = await decodeJSON(r, SetAllowlistDefaultRequest.self)
        guard case .ok(let body) = decoded else {
            if case .fail(let f) = decoded { return f }
            fatalError()
        }
        guard
            let moduleId = ModuleID(rawValue: body.module),
            !body.id.isEmpty
        else {
            return Self.error(
                status: .badRequest,
                message:
                    "module must be a ModuleID and id must be non-empty",
                type: "invalid_request_error", code: "invalid_body")
        }
        return await removeAllowlistRow(
            r, moduleId: moduleId, id: body.id)
    }

    /// `/ui/api/allowlist/default` POST (cookie). CSRF + `.modelWrite`,
    /// then `handleAllowlistSetDefault` (already JSON-body-shaped).
    private func uiAllowlistDefault(_ r: Request) async -> Response {
        guard csrfOK(r) else {
            return Self.uiDeny("csrf token missing or invalid")
        }
        guard await uiCaller(r).perms.contains(.modelWrite) else {
            return Self.uiDeny("need model.write")
        }
        return await handleAllowlistSetDefault(r)
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
        guard auth.isEnabled else { return Set(Permission.allCases) }
        if let h = request.headers[.authorization],
            h.hasPrefix("Bearer "),
            case let tok = String(h.dropFirst(7)), !tok.isEmpty,
            let s = await auth.resolve(bearer: tok)
        {
            return s.permissions
        }
        if let ck = Session.token(
            fromCookieHeader: request.headers[.cookie]),
            let user = session.validate(ck)
        {
            return await auth.permissions(forUser: user)
        }
        return []
    }

    /// The acting principal for an audit record, resolved for BOTH
    /// surfaces — a Bearer subject (`/api/*`), a WebUI session cookie
    /// (`/ui/*` → `u:<user>`), or the auth-off loopback operator
    /// (`xenos`). Mirrors `callerPermissions` so neither path is
    /// missed.
    private func auditPrincipal(_ request: Request) async -> String {
        guard auth.isEnabled else { return Self.xenos }
        if let h = request.headers[.authorization],
            h.hasPrefix("Bearer "),
            case let tok = String(h.dropFirst(7)), !tok.isEmpty,
            let s = await auth.resolve(bearer: tok)
        {
            return s.principal
        }
        if let ck = Session.token(
            fromCookieHeader: request.headers[.cookie]),
            let user = session.validate(ck)
        {
            return "u:" + user
        }
        return "unknown"
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
            Self.auditLog.warning(
                "audit write failed action=\(action): \(error)")
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
            if case .fail(let r) = decoded { return r }
            fatalError()
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
            return Self.error(
                status: .internalServerError, message: "\(error)",
                type: "server_error", code: "store_error")
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
        if await store.usersWithRole("admin") == [username] {
            await audit(
                request, action: "user.delete", target: username,
                result: "denied", detail: "only admin")
            return Self.deny403(
                "'\(username)' is the only admin — refusing "
                    + "(grant admin to another user first)")
        }
        let ok = await store.deleteUser(username: username)
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
            return Self.error(
                status: .internalServerError, message: "\(error)",
                type: "server_error", code: "store_error")
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
        if role == "admin",
            await store.usersWithRole("admin") == [username]
        {
            await audit(
                request, action: "role.revoke",
                target: "\(username):\(role)", result: "denied",
                detail: "only admin")
            return Self.deny403(
                "'\(username)' is the only admin — refusing to "
                    + "revoke admin")
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
                        hash_prefix: String($0.hex.prefix(12)),
                        label: $0.label, expires: $0.expires)
                }))
    }

    private func handleTokenCreate(_ request: Request) async
        -> Response
    {
        let decoded = await decodeJSON(
            request, CreateTokenRequest.self)
        guard case .ok(let body) = decoded else {
            if case .fail(let r) = decoded { return r }
            fatalError()
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
            return Self.error(
                status: .internalServerError, message: "\(error)",
                type: "server_error", code: "store_error")
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
        let matches = await store.listTokens().filter {
            $0.hex.hasPrefix(prefix)
        }
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
            if case .fail(let r) = decoded { return r }
            fatalError()
        }
        let matches = await store.listTokens().filter {
            $0.hex.hasPrefix(prefix)
        }
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
        let (key, hash) = AuthConfig.mintToken()
        _ = await store.deleteToken(hash: oldHash)
        do {
            try await store.putToken(
                hash: hash, username: m.username, scopedRoles: m.scoped,
                label: m.label, expires: expires)
        } catch {
            return Self.error(
                status: .internalServerError, message: "\(error)",
                type: "server_error", code: "store_error")
        }
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

    private static func deny403(_ msg: String) -> Response {
        Self.error(
            status: .forbidden, message: msg,
            type: "auth_error", code: "forbidden")
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
        stops: [String] = [],
        record: @escaping @Sendable (TokenUsage) async -> Void
    ) -> Response {
        let stream = AsyncStream<ByteBuffer> { continuation in
            let task = Task {
                func emit(_ chunk: ChatCompletionChunk) {
                    if let data = try? JSONEncoder().encode(chunk) {
                        var buf = ByteBuffer()
                        buf.writeString("data: ")
                        buf.writeBytes(data)
                        buf.writeString("\n\n")
                        continuation.yield(buf)
                    }
                }
                func emitDelta(_ piece: String) {
                    guard !piece.isEmpty else { return }
                    emit(
                        ChatCompletionChunk(
                            id: id, object: "chat.completion.chunk",
                            created: created, model: model,
                            choices: [
                                ChatChunkChoice(
                                    index: 0,
                                    delta: ChatDelta(
                                        role: nil, content: piece),
                                    finish_reason: nil)
                            ]))
                }
                emit(
                    ChatCompletionChunk(
                        id: id, object: "chat.completion.chunk",
                        created: created, model: model,
                        choices: [
                            ChatChunkChoice(
                                index: 0,
                                delta: ChatDelta(role: "assistant", content: ""),
                                finish_reason: nil)
                        ]))
                var usage = TokenUsage.zero
                var finish: FinishReason = .stop
                // M31.3: filter the streamed deltas through the stop
                // sequences; once a sequence matches, latch `stop`, emit
                // only the text up to it, and suppress the rest while still
                // draining the generator for an accurate usage count.
                var stopFilter = StopStreamFilter(stops: stops)
                for await event in events {
                    switch event {
                    case .text(let piece):
                        if stopFilter.isActive {
                            let wasStopped = stopFilter.stopped
                            emitDelta(stopFilter.push(piece))
                            if stopFilter.stopped && !wasStopped {
                                finish = .stop
                            }
                        } else {
                            emitDelta(piece)
                        }
                    case .usage(let u):
                        usage = u
                    case .finish(let r):
                        // A stop-sequence hit wins over the generator's
                        // own length/stop reason.
                        if !stopFilter.stopped { finish = r }
                    }
                }
                if stopFilter.isActive && !stopFilter.stopped {
                    emitDelta(stopFilter.flush())
                }
                // M31.2: the terminal chunk carries `length` when the
                // request hit max_tokens, `stop` otherwise (or on a stop
                // sequence hit, M31.3).
                emit(
                    ChatCompletionChunk(
                        id: id, object: "chat.completion.chunk",
                        created: created, model: model,
                        choices: [
                            ChatChunkChoice(
                                index: 0,
                                delta: ChatDelta(role: nil, content: nil),
                                finish_reason: finish.rawValue)
                        ]))
                // OpenAI emits usage in a final chunk with empty choices,
                // only when the client opted in.
                if includeUsage {
                    emit(
                        ChatCompletionChunk(
                            id: id, object: "chat.completion.chunk",
                            created: created, model: model, choices: [],
                            usage: Usage(
                                prompt_tokens: usage.promptTokens,
                                completion_tokens: usage.completionTokens,
                                total_tokens: usage.totalTokens)))
                }
                var done = ByteBuffer()
                done.writeString("data: [DONE]\n\n")
                continuation.yield(done)
                await record(usage)
                continuation.finish()
            }
            continuation.onTermination = { _ in task.cancel() }
        }
        var headers = HTTPFields()
        headers[.contentType] = "text/event-stream"
        headers[.cacheControl] = "no-cache"
        return Response(
            status: .ok, headers: headers,
            body: ResponseBody(asyncSequence: stream))
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
        code: String
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
                error: .init(message: message, type: type, code: code)),
            status: status)
    }

    /// M43.2 — response for a request that hit a still-cold module. The
    /// load runs detached on the background path; the caller gets a
    /// `503` with a fixed `Retry-After: 5` so the next attempt is paced
    /// instead of hammering the daemon while the multi-GB download
    /// finishes. 5 s is the brief's locked default; long enough to not
    /// thrash, short enough to feel responsive.
    private static func coldLoadResponse(_ id: ModuleID) -> Response {
        let body =
            #"{"error":{"message":"module "# + id.rawValue
            + #" is loading; retry shortly","type":"server_error","#
            + #""code":"module_loading"}}"#
        var buf = ByteBuffer()
        buf.writeBytes(Data(body.utf8))
        var headers = HTTPFields()
        headers[.contentType] = "application/json"
        headers[.retryAfter] = "5"
        return Response(
            status: .serviceUnavailable, headers: headers,
            body: ResponseBody(byteBuffer: buf))
    }

    /// Classify an arbitrary inference error: a genuine MLX/Metal OOM
    /// becomes a governed 503 (`metal_oom`), never a bare 500 / process
    /// abort (brief item 4a). Existing `AthenaError`s pass through.
    private static let log = Logger(label: AthenaLog.daemonLabel)
    private static let auditLog = Logger(label: AthenaLogLabel.audit)

    private static func classified(
        _ err: any Error, module: ModuleID
    ) -> Response {
        let e = AthenaError.classify(err, module: module)
        log.warning(
            """
            governed request failed module=\(module) \
            status=\(e.httpStatus) code=\(e.code) \(e.message)
            """)
        return error(
            status: HTTPResponse.Status(code: e.httpStatus),
            message: e.message, type: "server_error", code: e.code)
    }
}

/// M43.1 — /healthz response, flattening `GovernorSnapshot` for
/// backwards compat with consumers that read top-level `reservedBytes`
/// etc., plus three new live signals so a hung daemon is legible
/// without scraping /metrics: `inflight` (active request count),
/// `queueDepth` (jobs in `queued` status), `lastRequestAt` (epoch
/// seconds; 0 ⇒ none since boot).
struct HealthResponse: Encodable {
    let totalBudgetBytes: Int
    let reservedBytes: Int
    let freeBytes: Int
    let promptCacheCapBytes: Int
    let modules: [ModuleSnapshot]
    let inflight: Int
    let queueDepth: Int
    let lastRequestAt: Double

    init(
        snapshot: GovernorSnapshot, inflight: Int,
        queueDepth: Int, lastRequestAt: Double
    ) {
        self.totalBudgetBytes = snapshot.totalBudgetBytes
        self.reservedBytes = snapshot.reservedBytes
        self.freeBytes = snapshot.freeBytes
        self.promptCacheCapBytes = snapshot.promptCacheCapBytes
        self.modules = snapshot.modules
        self.inflight = inflight
        self.queueDepth = queueDepth
        self.lastRequestAt = lastRequestAt
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
