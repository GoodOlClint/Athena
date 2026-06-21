import AthenaCore
import Foundation

/// Typed inference contract for the LLM module. The MLX-backed
/// implementation lands in M1; this protocol is what the serve path holds.
public protocol LLMModule: InferenceModule {
    /// M71.2 — true when the resident model accepts image inputs (a vision
    /// checkpoint loaded via the substrate VLM path). The serve path gates
    /// OpenAI `image_url` content-parts on this. Default false (text-only
    /// conformers, the stub).
    var servesVision: Bool { get async }

    /// Stream generated text chunks. M0 is a canned stub; M1 replaces the
    /// body with native `TokenIterator` generation.
    nonisolated func generate(prompt: String) -> AsyncStream<String>
    /// Structured + tool-aware variant. `schemaJSON` (when non-nil)
    /// constrains output to the JSON schema; `tools` (OpenAI function
    /// specs, already lowered to plain Foundation containers) are rendered
    /// into the model's chat template so it knows the tool-call format.
    /// Default ignores both (no structured/tool support).
    nonisolated func generate(
        prompt: String, schemaJSON: String?,
        tools: [[String: any Sendable]]?
    ) -> AsyncStream<String>

    /// Role-aware variant (M24.1). `messages` carries the FULL
    /// conversation (system/user/assistant/tool) so the model's chat
    /// template sees system instructions and prior turns — not just a
    /// user-only join. `schemaJSON`/`tools` behave as in the prompt
    /// variant. Default bridges to the override-aware variant with no
    /// per-request overrides.
    nonisolated func generate(
        messages: [ChatTurn], schemaJSON: String?,
        tools: [[String: any Sendable]]?
    ) -> AsyncStream<String>

    /// Metered generation (M27.1) — the canonical entry point. Yields
    /// `.text` chunks exactly like the String `generate` overloads, then
    /// a single terminal `.usage(TokenUsage)` carrying the true
    /// prompt/completion token counts for THIS request, then a terminal
    /// `.finish(FinishReason)` (M31.2). Every other `generate` overload is
    /// a thin filter over this that drops the usage/finish events.
    /// `schemaJSON`/`tools`/`maxTokens`/`temperature` behave as on the
    /// String override-aware variant. `topP`/`seed` (M31.3) override the
    /// loaded sampling defaults on the substrate sampling path only —
    /// inert on the greedy/MTP/structured (argmax) paths.
    /// `chatTemplateKwargs` (M46.3b) are passed through to the model's
    /// chat template at rendering time (e.g.
    /// `{"enable_thinking": false}` on Qwen3-class models); nil ⇒ run
    /// the template with its built-in defaults.
    /// `promptCacheKey` (M59.3) is the OpenAI `prompt_cache_key` scoping
    /// hint; `principal` is the authenticated caller. Both feed the
    /// cross-request prompt-prefix cache's scope (a rebind never crosses
    /// models; the default scope never crosses principals). Ignored by
    /// conformers without the prefix cache (the stub).
    nonisolated func generateMetered(
        messages: [ChatTurn], schemaJSON: String?,
        tools: [[String: any Sendable]]?,
        maxTokens: Int?, temperature: Double?,
        topP: Double?, seed: Int?,
        speculative: Bool?,
        chatTemplateKwargs: [String: any Sendable]?,
        promptCacheKey: String?,
        principal: String?,
        logprobs: LogprobsRequest?
    ) -> AsyncStream<GenChunk>

    /// Override-aware String variant (M24.3). `maxTokens`/`temperature`,
    /// when non-nil and valid, override the daemon-load defaults for THIS
    /// request (e.g. OpenAI `max_tokens`/`temperature`). nil ⇒ the loaded
    /// `LLMGenerationParameters`. A filter over `generateMetered` that
    /// drops the terminal usage; callers that need usage consume
    /// `generateMetered` directly.
    nonisolated func generate(
        messages: [ChatTurn], schemaJSON: String?,
        tools: [[String: any Sendable]]?,
        maxTokens: Int?, temperature: Double?,
        speculative: Bool?
    ) -> AsyncStream<String>

    /// Reject — before any generation — a prompt whose KV/prompt-cache
    /// would exceed the governor-owned cap (brief 4b). Default: no cap
    /// (stub / modules without a model). Throws `AthenaError`
    /// (`.promptCacheCapExceeded`) so the serve path returns a
    /// governed 503.
    func preflightPromptCache(prompt: String) async throws

    /// Role-aware preflight (M24.1). Default bridges to the prompt path.
    /// NC3: `tools` + `chatTemplateKwargs` are included so the cap check
    /// renders the SAME prompt generation will (a tool/kwargs-bearing
    /// request otherwise undercounts and can exceed the cap mid-prefill).
    func preflightPromptCache(
        messages: [ChatTurn],
        tools: [[String: any Sendable]]?,
        chatTemplateKwargs: [String: any Sendable]?
    ) async throws

    /// M62 — choose the model the NEXT governor cold-load will bind, WITHOUT
    /// loading now: the non-blocking cold-load path (`beginLoadIfNeeded`)
    /// drives the actual load under the governor's reservation, and a cold
    /// slot would otherwise always bind the DEFAULT — so a request for a
    /// non-default model on a just-restarted/evicted slot silently served
    /// the wrong model. Set the target here (before the load is kicked off)
    /// and the cold-load binds it directly. Validated against the allowlist
    /// so an unknown id throws `AthenaError.modelNotAvailable` (400) BEFORE a
    /// doomed multi-GB load starts; `nil`/empty clears the override so the
    /// cold-load uses the default. Default no-op for conformers whose slot
    /// can't swap.
    func selectColdLoadModel(_ id: String?) async throws
}

extension LLMModule {
    /// Default: no vision (text-only conformers, the stub). M71.2.
    public var servesVision: Bool { get async { false } }

    public func preflightPromptCache(prompt: String) async throws {}

    public func selectColdLoadModel(_ id: String?) async throws {}

    public func preflightPromptCache(
        messages: [ChatTurn],
        tools: [[String: any Sendable]]? = nil,
        chatTemplateKwargs: [String: any Sendable]? = nil
    ) async throws {
        // Default conformers have no governed cap; the bridge ignores tools/
        // kwargs (only the real MLX cap renders them into the token count).
        try await preflightPromptCache(prompt: messages.flattenedPrompt())
    }

    public nonisolated func generate(
        messages: [ChatTurn], schemaJSON: String?,
        tools: [[String: any Sendable]]?
    ) -> AsyncStream<String> {
        generate(
            messages: messages, schemaJSON: schemaJSON, tools: tools,
            maxTokens: nil, temperature: nil, speculative: nil)
    }

    public nonisolated func generate(
        messages: [ChatTurn], schemaJSON: String?,
        tools: [[String: any Sendable]]?,
        maxTokens: Int?, temperature: Double?,
        speculative: Bool?
    ) -> AsyncStream<String> {
        // Single source of truth: stream the metered events and forward
        // only the text chunks (drop the terminal usage). M46.3b's
        // chat-template-kwargs default to nil here — the String variant
        // is convenience-only; if a caller needs template kwargs they
        // use generateMetered directly.
        let events = generateMetered(
            messages: messages, schemaJSON: schemaJSON, tools: tools,
            maxTokens: maxTokens, temperature: temperature,
            topP: nil, seed: nil, speculative: speculative,
            chatTemplateKwargs: nil, promptCacheKey: nil, principal: nil,
            logprobs: nil)
        return AsyncStream { continuation in
            let task = Task {
                for await event in events {
                    if case .text(let t) = event { continuation.yield(t) }
                }
                continuation.finish()
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    /// Default metered generation for conformers without native token
    /// counts (the stub): stream chunks from the String `generate(prompt:
    /// schemaJSON:tools:)` path and synthesize whitespace-delimited token
    /// counts so `usage` and the metrics counter are non-zero end-to-end
    /// under `--engine stub`. The MLX module overrides this with real
    /// tokenizer counts.
    public nonisolated func generateMetered(
        messages: [ChatTurn], schemaJSON: String?,
        tools: [[String: any Sendable]]?,
        maxTokens: Int?, temperature: Double?,
        topP: Double?, seed: Int?,
        speculative: Bool?,
        chatTemplateKwargs: [String: any Sendable]?,
        promptCacheKey: String? = nil,
        principal: String? = nil,
        logprobs: LogprobsRequest? = nil
    ) -> AsyncStream<GenChunk> {
        // The model-free stub has no sampler, so topP/seed/speculative
        // are accepted and ignored; the e2e gate exercises the sampling
        // knobs via stop/max_tokens, and the speculative flag itself is
        // exercised end-to-end against the real MLX module on the
        // manual host-bound tier. M46.3b's `chatTemplateKwargs` is
        // accepted and ignored here for the same reason: the stub has
        // no tokenizer/chat template, so the kwarg is exercised
        // end-to-end against the real MLX module + a real tokenizer
        // on the manual host-bound tier.
        let prompt = messages.flattenedPrompt()
        let chunks = generate(
            prompt: prompt, schemaJSON: schemaJSON, tools: tools)
        // M31.2: honor a positive `max_tokens` so the synthetic stream
        // truncates and reports `.finish(.length)` — gives the stub
        // engine a deterministic truncation signal for the e2e gate. A
        // 0/negative/absent cap means "no limit" (the loaded default has
        // no meaning for the model-less stub).
        let cap = (maxTokens.flatMap { $0 > 0 ? $0 : nil }) ?? Int.max
        return AsyncStream { continuation in
            let task = Task {
                var completion = 0
                var truncated = false
                var emitted: [String] = []
                for await chunk in chunks {
                    if completion >= cap {
                        truncated = true
                        break
                    }
                    continuation.yield(.text(chunk))
                    emitted.append(chunk)
                    completion += Self.approxTokens(chunk)
                }
                continuation.yield(
                    .usage(
                        TokenUsage(
                            promptTokens: Self.approxTokens(prompt),
                            completionTokens: completion)))
                // C2 — the model-free stub has no logits, so synthesize a
                // logprob record per emitted chunk (treating each as a token)
                // with a fixed score, so the request→response→logprobs shape
                // is exercisable under `--engine stub`. The MLX module emits
                // real captures.
                if let lp = logprobs {
                    let synth = emitted.map { piece -> TokenLogprob in
                        let bytes = Array(piece.utf8).map(Int.init)
                        let top =
                            lp.topLogprobs > 0
                            ? [
                                TopLogprob(
                                    token: piece, logprob: -0.1, bytes: bytes)
                            ] : []
                        return TokenLogprob(
                            token: piece, logprob: -0.1, bytes: bytes, top: top)
                    }
                    continuation.yield(.logprobs(synth))
                }
                continuation.yield(
                    .finish(truncated ? .length : .stop))
                continuation.finish()
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    /// Whitespace-delimited token estimate (≥1 for non-empty text) used
    /// only by the stub's synthetic usage — never the real model path.
    static func approxTokens(_ s: String) -> Int {
        let n = s.split(whereSeparator: { $0.isWhitespace }).count
        return s.isEmpty ? 0 : max(1, n)
    }

    public nonisolated func generate(
        prompt: String, schemaJSON: String?,
        tools: [[String: any Sendable]]?
    ) -> AsyncStream<String> {
        generate(prompt: prompt)
    }

    /// Source-compat overload for callers that don't pass tools.
    public nonisolated func generate(
        prompt: String, schemaJSON: String?
    ) -> AsyncStream<String> {
        generate(prompt: prompt, schemaJSON: schemaJSON, tools: nil)
    }
}

/// M0 governed stub. It holds no model — it exists to prove the thesis path
/// end-to-end: a request loads it through the governor (reserving real
/// budget), streams tokens over the API, and releases on unload.
public actor StubLLMModule: LLMModule, ModelSelectable {
    public nonisolated let id: ModuleID = .llm
    public nonisolated var moduleID: ModuleID { .llm }

    private let reserveBytes: Int
    private let modelIds: [String]
    private let configuredDefault: String?
    private var residentId: String?
    /// M62 — the model the next cold `load` will bind (nil ⇒ the default).
    private var desiredId: String?

    /// `reserveBytes` defaults to a representative multi-GB LLM footprint so
    /// budget pressure behaves realistically; tests inject small values.
    /// `modelIds` stands in for the store's LLM models (the stub has no disk);
    /// `configuredDefault` is the per-module TOML default (ADR 026 — nil ⇒
    /// resolve by the ambiguity rule). The stub "serves" any of `modelIds`
    /// truthfully so `/api/models/load` and the inference-time `model` field
    /// exercise rebind under the stub engine without a model on disk.
    public init(
        reserveBytes: Int = 8 * 1024 * 1024 * 1024,
        modelIds: [String] = ["athena-stub"],
        configuredDefault: String? = nil
    ) {
        precondition(
            !modelIds.isEmpty,
            "StubLLMModule needs at least one model id")
        self.reserveBytes = reserveBytes
        self.modelIds = modelIds
        self.configuredDefault =
            (configuredDefault?.isEmpty == true) ? nil : configuredDefault
    }

    public var residentBytes: Int { residentId == nil ? 0 : reserveBytes }

    public func memoryEstimate() -> Int { reserveBytes }

    public func load(reservation: MemoryReservation) async throws {
        // M62 — bind the requested cold-load target (set via
        // selectColdLoadModel), else the resolved default (ADR 026).
        guard !modelIds.isEmpty else { return }
        if residentId == nil {
            residentId = try desiredId ?? resolvedDefaultId()
        }
    }

    public func unload() async {
        residentId = nil
    }

    public func allowedModelIds() -> [String] { modelIds }
    public func defaultModelId() -> String {
        ModelSelection.displayDefault(
            available: modelIds, configuredDefault: configuredDefault)
    }
    public func residentModelId() -> String? { residentId }
    public func rebind(to id: String?) async throws {
        residentId = try resolve(id)
    }

    public func selectColdLoadModel(_ id: String?) async throws {
        desiredId = try resolve(id)
    }

    /// ADR 026 resolution against the injected stub model set.
    private func resolve(_ id: String?) throws -> String {
        switch ModelSelection.resolve(
            available: modelIds, configuredDefault: configuredDefault,
            requested: id)
        {
        case .resolved(let t): return t
        case .notAvailable:
            throw AthenaError.modelNotAvailable(
                requested: id ?? (configuredDefault ?? ""),
                available: modelIds)
        case .ambiguous:
            throw AthenaError.ambiguousModel(module: .llm, available: modelIds)
        }
    }
    private func resolvedDefaultId() throws -> String { try resolve(nil) }

    public nonisolated func generate(prompt: String) -> AsyncStream<String> {
        AsyncStream { continuation in
            let chunks = [
                "Athena ", "M0 ", "stub", ": ", "governed ", "memory ",
                "path ", "is ", "live", ".",
            ]
            let delay = Self.chunkDelayNanos
            let task = Task {
                for chunk in chunks {
                    continuation.yield(chunk)
                    try? await Task.sleep(nanoseconds: delay)
                }
                continuation.finish()
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    /// Per-chunk pacing for the canned stream (default 15 ms). The
    /// dev/e2e-only `ATHENA_STUB_DELAY_MS` env var widens it so the e2e
    /// gate can drive a generation past a whole-second request timeout
    /// (the real model has no such knob — it's just slow enough on its
    /// own). Read once per stream.
    private static var chunkDelayNanos: UInt64 {
        if let raw = ProcessInfo.processInfo.environment[
            "ATHENA_STUB_DELAY_MS"], let ms = UInt64(raw), ms > 0 {
            return ms * 1_000_000
        }
        return 15_000_000
    }
}
