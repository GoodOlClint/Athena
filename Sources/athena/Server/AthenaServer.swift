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
import Logging
import NIOCore

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
    /// WebUI session signer (M12.2). Per-process random secret —
    /// sessions invalidate on restart (acceptable for an appliance).
    let session = Session()

    func run() async throws {
        let router = Router()
        // Auth outermost — reject at the edge, before timing work.
        router.add(
            middleware: AuthMiddleware(
                config: auth, session: session))
        router.add(middleware: MetricsMiddleware(metrics: metrics))

        router.get("/healthz") { _, _ -> Response in
            let snapshot = await governor.snapshot()
            return Self.json(snapshot)
        }

        router.get("/metrics") { _, _ -> Response in
            Self.json(await metrics.snapshot())
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

        router.post("/v1/embeddings") { request, _ -> Response in
            await handleEmbeddings(request)
        }

        router.post("/v1/audio/transcriptions") { request, _ -> Response in
            await handleTranscriptions(request)
        }

        router.post("/v1/audio/diarizations") { request, _ -> Response in
            await handleDiarizations(request)
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
        router.post("/api/admin/stop") { _, _ -> Response in
            await adminUnloadLLM()
        }
        router.get("/api/admin/status") { _, _ -> Response in
            await adminStatus()
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
        router.get("/api/models/:name") { _, context -> Response in
            handleModelShow(context.parameters.get("name"))
        }
        router.delete("/api/models/:name") { _, context -> Response in
            handleModelRemove(context.parameters.get("name"))
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
            Self.json(
                RolesResponse(
                    roles: RBAC.roleNames.map { r in
                        RoleCatalogEntry(
                            role: r,
                            permissions: (RBAC.catalog[r] ?? [])
                                .map(\.rawValue).sorted())
                    }))
        }
        router.get("/api/tokens") { _, _ -> Response in
            await handleTokensList()
        }
        router.post("/api/tokens") { request, _ -> Response in
            await handleTokenCreate(request)
        }
        router.delete("/api/tokens/:prefix") { _, context
            -> Response in
            await handleTokenDelete(
                context.parameters.get("prefix"))
        }

        // Wire the queue executor to the governed module paths and
        // start the single serial worker (M8.1).
        await queue.setExecutor { kind, data in
            await self.queuedExecute(kind: kind, request: data)
        }
        let worker = Task { await queue.runWorker() }
        defer { worker.cancel() }

        let app = Application(
            router: router,
            configuration: .init(
                address: .hostname(
                    config.listenHost, port: config.listenPort),
                serverName: "athena"
            )
        )
        try await app.runService()
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

        // The governed path: load the LLM under the global budget. A budget
        // event becomes a classified 503 here, never a Metal abort.
        do {
            try await governor.ensureLoaded(.llm)
        } catch let e as AthenaError {
            return Self.error(
                status: HTTPResponse.Status(code: e.httpStatus),
                message: e.message,
                type: "server_error",
                code: e.code)
        } catch {
            return Self.error(
                status: .internalServerError,
                message: String(describing: error),
                type: "server_error",
                code: "internal_error")
        }

        let model = body.model ?? "athena-stub"
        let prompt = body.messages
            .filter { $0.role == "user" }
            .compactMap(\.content)
            .joined(separator: "\n")
        // Brief 4b: refuse an over-cap prompt up front as a governed
        // 503, before any KV cache is allocated.
        do {
            try await llm.preflightPromptCache(prompt: prompt)
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

        if body.stream == true {
            return Self.streamSSE(
                id: id, model: model, created: created,
                tokens: llm.generate(
                    prompt: prompt, schemaJSON: schemaJSON,
                    tools: toolSpecs))
        }

        var text = ""
        for await chunk in llm.generate(
            prompt: prompt, schemaJSON: schemaJSON, tools: toolSpecs)
        {
            text += chunk
        }

        // Tool call: the enforced JSON span is the {"name","arguments"}
        // object — surface it as OpenAI tool_calls, not content.
        let choice: ChatChoice
        if effective?.isToolCall == true,
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
            choice = ChatChoice(
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
        } else {
            choice = ChatChoice(
                index: 0,
                message: ChatMessage(role: "assistant", content: text),
                finish_reason: "stop")
        }
        let response = ChatCompletionResponse(
            id: id, object: "chat.completion", created: created,
            model: model, choices: [choice],
            usage: Usage(
                prompt_tokens: 0, completion_tokens: 0, total_tokens: 0)
        )
        return Self.json(response)
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
            try await governor.ensureLoaded(.textEmbedding)
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

        let vectors: [[Float]]
        do {
            vectors = try await embedding.embed(body.input)
        } catch {
            return Self.classified(error, module: .textEmbedding)
        }

        let response = EmbeddingResponse(
            object: "list",
            data: vectors.enumerated().map {
                EmbeddingObject(
                    object: "embedding", embedding: $0.element,
                    index: $0.offset)
            },
            model: body.model ?? "athena-embedding",
            usage: Usage(
                prompt_tokens: 0, completion_tokens: 0, total_tokens: 0))
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
            try await governor.ensureLoaded(.transcription)
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

        let result: TranscriptionResult
        do {
            result = try await transcription.transcribe(
                audio: file.data, filename: file.filename,
                language: form.text("language"))
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
                    try await governor.ensureLoaded(.diarization)
                    turns = try await diarization.diarize(
                        audio: file.data, filename: file.filename
                    ).turns
                } catch {
                    return Self.classified(error, module: .diarization)
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
                            speaker: turns.isEmpty
                                ? nil
                                : Self.speaker(
                                    start: $0.element.start,
                                    end: $0.element.end, turns: turns))
                    }))
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

        do {
            try await governor.ensureLoaded(.diarization)
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
        case .ok(let vs):
            return .ok(vs.first ?? [])
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
    private func queuedExecute(
        kind: String, request: Data
    ) async -> (result: Data?, error: String?) {
        switch kind {
        case "conversation":
            guard
                let req = try? JSONDecoder().decode(
                    AthenaChatRequest.self, from: request)
            else { return (nil, "invalid conversation body") }
            let prompt = req.messages
                .filter { $0.role == "user" }
                .map { $0.content }
                .joined(separator: "\n")
            do {
                try await governor.ensureLoaded(.llm)
                try await llm.preflightPromptCache(prompt: prompt)
            } catch let e as AthenaError {
                return (nil, e.message)
            } catch {
                return (nil, String(describing: error))
            }
            var text = ""
            for await c in llm.generate(prompt: prompt) { text += c }
            return (
                try? JSONEncoder().encode(
                    QueuedTextResult(text: text)), nil
            )
        case "embeddings":
            guard
                let req = try? JSONDecoder().decode(
                    AthenaEmbedRequest.self, from: request)
            else { return (nil, "invalid embeddings body") }
            do {
                try await governor.ensureLoaded(.textEmbedding)
                let v = try await embedding.embed(req.input)
                return (
                    try? JSONEncoder().encode(
                        QueuedEmbeddingResult(embeddings: v)), nil
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
                let r = try await ModelConvert.convert(
                    id: req.id, revision: req.revision,
                    bits: req.bits ?? 4,
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

    /// Governed gate for `/api/chat`: ensureLoaded(.llm) + the 4b
    /// prompt-cache preflight, both classified. Returns an error
    /// `Response` to send, or nil when the request may proceed.
    private func governedPreflight(
        prompt: String
    ) async -> Response? {
        do {
            try await governor.ensureLoaded(.llm)
            try await llm.preflightPromptCache(prompt: prompt)
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
        let prompt = body.messages
            .filter { $0.role == "user" }
            .map { $0.content }
            .joined(separator: "\n")
        if let err = await governedPreflight(prompt: prompt) {
            return err
        }
        let model = body.model ?? modelName
        if body.stream == true {
            return Self.streamNDJSON(
                tokens: llm.generate(prompt: prompt)
            ) { content, done in
                try? JSONEncoder().encode(
                    AthenaChatChunk(
                        content: done ? "" : content, done: done))
            }
        }
        var text = ""
        for await c in llm.generate(prompt: prompt) { text += c }
        return Self.json(
            AthenaChatResponse(
                model: model, content: text, done: true))
    }

    /// Governed embedding helper shared by `/api/embed`, `/v1/vectors`
    /// text resolution, and the queued `embeddings` kind.
    private func governedEmbed(_ inputs: [String], module: ModuleID)
        async -> Outcome<[[Float]]>
    {
        do {
            try await governor.ensureLoaded(.textEmbedding)
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
            return .ok(try await embedding.embed(inputs))
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
        switch await governedEmbed(body.input, module: .textEmbedding) {
        case .fail(let r): return r
        case .ok(let v):
            return Self.json(
                AthenaEmbedResponse(
                    model: body.model ?? modelName, embeddings: v))
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

    private func handleModelRemove(_ name: String?) -> Response {
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
        return Self.json(
            DefaultModelResponse(model: name, source: "config"))
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
            return self.handleModelRemove(body.name)
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
    private func adminUnloadLLM() async -> Response {
        await governor.unload(.llm)
        return Self.json(
            AthenaStopResponse(status: "unloaded", model: modelName))
    }

    /// Pre-warm the LLM module so the next inference is hot. New
    /// daemon-control verb (M18.3); exposed only on the cookie+CSRF
    /// /ui surface (the public /api admin surface is frozen at M16).
    private func adminLoadLLM() async -> Response {
        do {
            try await governor.ensureLoaded(.llm)
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
        return await adminUnloadLLM()
    }

    private func uiAdminLoad(_ r: Request) async -> Response {
        guard csrfOK(r) else {
            return Self.uiDeny("csrf token missing or invalid")
        }
        guard await uiCaller(r).perms.contains(.daemonAdmin) else {
            return Self.uiDeny("need daemon.admin")
        }
        return await adminLoadLLM()
    }

    // MARK: - RBAC admin (M16.4)

    /// The CALLER's effective permission set. Auth-off loopback is a
    /// single trusted operator (mirrors the offline CLI's implicit-
    /// admin grantor). With auth on, a missing/invalid bearer ⇒ empty
    /// set ⇒ every escalation check fails closed (AuthMiddleware has
    /// already gated the route, so this is defense-in-depth).
    private func callerPermissions(_ request: Request) async
        -> Set<Permission>
    {
        guard auth.isEnabled else { return Set(Permission.allCases) }
        guard
            let h = request.headers[.authorization],
            h.hasPrefix("Bearer "),
            case let tok = String(h.dropFirst(7)), !tok.isEmpty,
            let s = await auth.resolve(bearer: tok)
        else { return [] }
        return s.permissions
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
            return Self.deny403(
                "'\(username)' is the only admin — refusing "
                    + "(grant admin to another user first)")
        }
        let ok = await store.deleteUser(username: username)
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
            return Self.deny403(
                "'\(username)' is the only admin — refusing to "
                    + "revoke admin")
        }
        let ok = await store.revokeRole(
            username: username, role: role)
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
                        label: $0.label)
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
                return Self.deny403(
                    "token scope exceeds your permissions")
            }
        } else {
            // Unscoped ⇒ inherits the user's full roles; refuse to
            // mint a token more powerful than the caller.
            let inherited = RBAC.permissions(
                forRoles: await store.rolesForUser(username: user))
            guard inherited.isSubset(of: caller) else {
                return Self.deny403(
                    "an unscoped token for '\(user)' would exceed "
                        + "your permissions — pass an explicit role "
                        + "scope you can grant")
            }
        }
        let (key, hash) = AuthConfig.mintToken()
        do {
            try await store.putToken(
                hash: hash, username: user, scopedRoles: scoped,
                label: body.label)
        } catch {
            return Self.error(
                status: .internalServerError, message: "\(error)",
                type: "server_error", code: "store_error")
        }
        return Self.json(
            CreateTokenResponse(
                user: user, scope: scoped, token: key,
                hash_prefix: String(
                    AuthConfig.hex(Array(hash)).prefix(12))),
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

    private func handleTokenDelete(_ prefix: String?) async
        -> Response
    {
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
        return Self.json(TokensRemovedResponse(removed: removed))
    }

    private static func deny403(_ msg: String) -> Response {
        Self.error(
            status: .forbidden, message: msg,
            type: "auth_error", code: "forbidden")
    }

    // MARK: - Response helpers

    private static func streamSSE(
        id: String, model: String, created: Int,
        tokens: AsyncStream<String>
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
                for await piece in tokens {
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
                                delta: ChatDelta(role: nil, content: nil),
                                finish_reason: "stop")
                        ]))
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

    private static func error(
        status: HTTPResponse.Status, message: String, type: String,
        code: String
    ) -> Response {
        json(
            APIErrorBody(
                error: .init(message: message, type: type, code: code)),
            status: status)
    }

    /// Classify an arbitrary inference error: a genuine MLX/Metal OOM
    /// becomes a governed 503 (`metal_oom`), never a bare 500 / process
    /// abort (brief item 4a). Existing `AthenaError`s pass through.
    private static let log = Logger(label: AthenaLog.daemonLabel)

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
