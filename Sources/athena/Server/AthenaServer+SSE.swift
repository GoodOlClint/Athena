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
import HummingbirdCore
import HummingbirdTLS
import Logging
import MLX
import NIOCore
import NIOSSL

// ADR 036 — the SSE streaming machinery shared by every chat dialect: the two
// producers (`streamSSE` OpenAI, `streamAnthropic`), the cold-load keep-alive
// stream, and the `foldGenChunks` fold both pumps run over. Encode-only; the
// orchestration/admission decisions live on the base `AthenaServer`.
extension AthenaServer {
    /// Stream `/v1/chat/completions` over SSE (M27.4). Consumes the
    /// metered stream so it can (a) emit a terminal usage chunk when the
    /// client set `stream_options.include_usage` and (b) always meter the
    /// request via `record` once generation finishes — closing the
    /// streaming metering gap from M27.1. `record` runs inside the
    /// streaming task (the body is produced lazily).
    static func streamSSE(
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
    static func streamSSEAwaitingLoad(
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
            (
                _ reason: FinishReason, _ toolCalled: Bool, _ stopHit: String?,
                _ usage: TokenUsage
            ) -> Void
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
    static func streamAnthropic(
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
}
