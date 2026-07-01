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

// ADR 036 — the Anthropic Messages dialect (`POST /v1/messages`) over the shared
// inference engine. Decode+encode only; the orchestration seam (`prepareChat`),
// SSE machinery, and metering live on the base `AthenaServer`.
extension AthenaServer {
    /// ADR 036 — register the Anthropic Messages route. Called from `run()`.
    func registerAnthropicRoutes(_ router: Router<AppRequestContext>) {
        router.post("/v1/messages") { request, _ -> Response in
            await self.handleAnthropicMessages(request)
        }
    }


    /// ADR 036 S2 — the Anthropic Messages adapter (`POST /v1/messages`). Decode
    /// → `NativeChatRequest`, run the shared `prepareChat` seam, drain the one
    /// `GenChunk` stream, encode the Anthropic response. No orchestration or
    /// engine logic here — it reuses the exact path the OpenAI adapter does.
    /// Increment 1: non-streaming. Streaming (the SSE event protocol) lands in
    /// the next slice; a `stream:true` request is refused with a clean 400 until
    /// then.
    func handleAnthropicMessages(_ request: Request) async -> Response {
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
}
