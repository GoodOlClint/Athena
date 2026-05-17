import AthenaCore
import AthenaEmbedding
import AthenaLLM
import AthenaTranscription
import Foundation
import HTTPTypes
import Hummingbird
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

    func run() async throws {
        let router = Router()

        router.get("/healthz") { _, _ -> Response in
            let snapshot = await governor.snapshot()
            return Self.json(snapshot)
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
            return Self.json(
                VerboseTranscriptionResponse(
                    task: "transcribe", language: result.language,
                    duration: result.duration, text: result.text,
                    segments: result.segments.enumerated().map {
                        VerboseSegment(
                            id: $0.offset, start: $0.element.start,
                            end: $0.element.end, text: $0.element.text)
                    }))
        default:  // "json" / nil
            return Self.json(TranscriptionResponse(text: result.text))
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

    private static func json<T: Encodable>(
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
    private static func classified(
        _ err: any Error, module: ModuleID
    ) -> Response {
        let e = AthenaError.classify(err, module: module)
        return error(
            status: HTTPResponse.Status(code: e.httpStatus),
            message: e.message, type: "server_error", code: e.code)
    }
}
