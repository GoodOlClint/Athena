import AthenaCore
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
    /// Display name reported by the Ollama shim (`/api/tags` etc.).
    let modelName: String
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
        router.get("/ui") { _, _ -> Response in
            Self.html(Self.uiPage)
        }
        router.get("/ui/api/state") { _, _ -> Response in
            await handleUIState()
        }
        router.get("/ui/config") { _, _ -> Response in
            Self.html(Self.configPage)
        }
        router.get("/ui/api/config") { _, _ -> Response in
            await handleUIConfigGet()
        }
        router.post("/ui/api/config") { request, _ -> Response in
            await handleUIConfigPost(request)
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
        router.delete("/v1/queue/:arg") { _, context -> Response in
            await handleQueueRemove(context.parameters.get("arg"))
        }
        // SSE: stream status transitions until terminal (inbound only —
        // the passive-oracle thesis forbids outbound webhooks).
        router.get("/v1/queue/:arg/events") { _, context -> Response in
            handleQueueEvents(context.parameters.get("arg"))
        }

        // Ollama-compatible shim (M6.1, non-streaming).
        router.get("/api/version") { _, _ -> Response in
            Self.json(OllamaVersionResponse(version: OllamaShim.version))
        }
        router.get("/api/tags") { _, _ -> Response in
            Self.json(
                OllamaTagsResponse(models: [
                    OllamaModelInfo(
                        name: modelName, model: modelName,
                        modified_at: Self.now(), size: 0,
                        digest: "")
                ]))
        }
        router.post("/api/chat") { request, _ -> Response in
            await handleOllamaChat(request)
        }
        router.post("/api/generate") { request, _ -> Response in
            await handleOllamaGenerate(request)
        }
        router.post("/api/embeddings") { request, _ -> Response in
            await handleOllamaEmbeddings(request)
        }
        router.post("/api/embed") { request, _ -> Response in
            await handleOllamaEmbed(request)
        }
        router.post("/api/stop") { _, _ -> Response in
            await governor.unload(.llm)
            return Self.json([
                "status": "unloaded", "model": modelName,
            ])
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
        switch await ollamaEmbed([text], module: .textEmbedding) {
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

    private func handleQueueSubmit(
        _ kind: String?, _ request: Request
    ) async -> Response {
        guard let kind, RequestQueue.kinds.contains(kind) else {
            return Self.error(
                status: .badRequest,
                message:
                    "unknown queue kind; expected one of "
                    + RequestQueue.kinds.sorted().joined(
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
        do {
            let id = try await queue.submit(
                kind: kind, request: body)
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
        let deadline = Date().addingTimeInterval(Double(wait))
        while true {
            guard let job = await queue.status(id: id) else {
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

    private func handleQueueEvents(_ id: String?) -> Response {
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

    private func handleQueueRemove(_ id: String?) async -> Response {
        guard let id else {
            return Self.error(
                status: .badRequest, message: "missing job id",
                type: "invalid_request_error", code: "missing_id")
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
        let jobs = await queue.list(status: status)
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
                    OllamaChatRequest.self, from: request)
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
                    OllamaEmbedRequest.self, from: request)
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
        default:
            return (nil, "unknown kind '\(kind)'")
        }
    }

    // MARK: - Ollama shim (M6.1)

    /// `Response` isn't `Error`, so a plain success-or-error-response.
    private enum Outcome<T> {
        case ok(T)
        case fail(Response)
    }

    private static func now() -> String {
        ISO8601DateFormatter().string(from: Date())
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

    /// The governed text path shared by `/api/chat` + `/api/generate`:
    /// ensureLoaded(.llm) + the 4b prompt-cache preflight (both
    /// classified), then non-streamed accumulation. Mirrors the
    /// OpenAI non-stream path.
    /// Governed gate shared by chat/generate: ensureLoaded(.llm) + the
    /// 4b prompt-cache preflight, both classified. Returns an error
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

    /// Newline-delimited JSON streamer (Ollama's stream format). Each
    /// generated piece → one JSON line; a final `done` line closes it.
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

    private func handleOllamaChat(_ request: Request) async -> Response {
        let decoded = await decodeJSON(request, OllamaChatRequest.self)
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
        if body.stream == false {
            var text = ""
            for await c in llm.generate(prompt: prompt) { text += c }
            return Self.json(
                OllamaChatResponse(
                    model: model, created_at: Self.now(),
                    message: OllamaMessage(
                        role: "assistant", content: text),
                    done: true, done_reason: "stop"))
        }
        return Self.streamNDJSON(tokens: llm.generate(prompt: prompt)) {
            content, done in
            try? JSONEncoder().encode(
                OllamaChatResponse(
                    model: model, created_at: Self.now(),
                    message: OllamaMessage(
                        role: "assistant",
                        content: done ? "" : content),
                    done: done, done_reason: done ? "stop" : ""))
        }
    }

    private func handleOllamaGenerate(_ request: Request) async -> Response
    {
        let decoded = await decodeJSON(
            request, OllamaGenerateRequest.self)
        guard case .ok(let body) = decoded else {
            if case .fail(let r) = decoded { return r }
            fatalError()
        }
        if let err = await governedPreflight(prompt: body.prompt) {
            return err
        }
        let model = body.model ?? modelName
        if body.stream == false {
            var text = ""
            for await c in llm.generate(prompt: body.prompt) {
                text += c
            }
            return Self.json(
                OllamaGenerateResponse(
                    model: model, created_at: Self.now(),
                    response: text, done: true, done_reason: "stop"))
        }
        return Self.streamNDJSON(
            tokens: llm.generate(prompt: body.prompt)
        ) { content, done in
            try? JSONEncoder().encode(
                OllamaGenerateResponse(
                    model: model, created_at: Self.now(),
                    response: done ? "" : content,
                    done: done, done_reason: done ? "stop" : ""))
        }
    }

    private func ollamaEmbed(_ inputs: [String], module: ModuleID)
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

    private func handleOllamaEmbeddings(_ request: Request) async
        -> Response
    {
        let decoded = await decodeJSON(
            request, OllamaEmbeddingsRequest.self)
        guard case .ok(let body) = decoded else {
            if case .fail(let r) = decoded { return r }
            fatalError()
        }
        switch await ollamaEmbed([body.prompt], module: .textEmbedding) {
        case .fail(let r): return r
        case .ok(let v):
            return Self.json(
                OllamaEmbeddingsResponse(embedding: v.first ?? []))
        }
    }

    private func handleOllamaEmbed(_ request: Request) async -> Response {
        let decoded = await decodeJSON(request, OllamaEmbedRequest.self)
        guard case .ok(let body) = decoded else {
            if case .fail(let r) = decoded { return r }
            fatalError()
        }
        guard !body.input.isEmpty else {
            return Self.error(
                status: .badRequest, message: "'input' is required",
                type: "invalid_request_error", code: "invalid_input")
        }
        switch await ollamaEmbed(body.input, module: .textEmbedding) {
        case .fail(let r): return r
        case .ok(let v):
            return Self.json(
                OllamaEmbedResponse(
                    model: body.model ?? modelName, embeddings: v))
        }
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
