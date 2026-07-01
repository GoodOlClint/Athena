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

// The OpenAI `POST /v1/chat/completions` adapter: decode → the shared
// `prepareChat` orchestration seam → drain/stream the `GenChunk` stream →
// encode the OpenAI response. `collectMetered` (the blocking-path collector,
// shared with the Anthropic adapter) rides along.
extension AthenaServer {
    /// Register the OpenAI chat route (`AthenaServer+ChatOpenAI.swift`).
    /// Called from `run()`.
    func registerChatRoutes(_ router: Router<AppRequestContext>) {
        router.post("/v1/chat/completions") { request, _ -> Response in
            await self.handleChatCompletions(request)
        }
    }

    func handleChatCompletions(_ request: Request) async -> Response {
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
    static func elapsedMs(_ start: Date) -> Int {
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
    func collectMetered(
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
}
