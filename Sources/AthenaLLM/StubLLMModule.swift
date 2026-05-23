import AthenaCore
import Foundation

/// Typed inference contract for the LLM module. The MLX-backed
/// implementation lands in M1; this protocol is what the serve path holds.
public protocol LLMModule: InferenceModule {
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
    /// prompt/completion token counts for THIS request. Every other
    /// `generate` overload is a thin filter over this that drops the
    /// usage event. `schemaJSON`/`tools`/`maxTokens`/`temperature` behave
    /// as on the String override-aware variant.
    nonisolated func generateMetered(
        messages: [ChatTurn], schemaJSON: String?,
        tools: [[String: any Sendable]]?,
        maxTokens: Int?, temperature: Double?
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
        maxTokens: Int?, temperature: Double?
    ) -> AsyncStream<String>

    /// Reject — before any generation — a prompt whose KV/prompt-cache
    /// would exceed the governor-owned cap (brief 4b). Default: no cap
    /// (stub / modules without a model). Throws `AthenaError`
    /// (`.promptCacheCapExceeded`) so the serve path returns a
    /// governed 503.
    func preflightPromptCache(prompt: String) async throws

    /// Role-aware preflight (M24.1). Default bridges to the prompt path.
    func preflightPromptCache(messages: [ChatTurn]) async throws
}

extension LLMModule {
    public func preflightPromptCache(prompt: String) async throws {}

    public func preflightPromptCache(messages: [ChatTurn]) async throws {
        try await preflightPromptCache(prompt: messages.flattenedPrompt())
    }

    public nonisolated func generate(
        messages: [ChatTurn], schemaJSON: String?,
        tools: [[String: any Sendable]]?
    ) -> AsyncStream<String> {
        generate(
            messages: messages, schemaJSON: schemaJSON, tools: tools,
            maxTokens: nil, temperature: nil)
    }

    public nonisolated func generate(
        messages: [ChatTurn], schemaJSON: String?,
        tools: [[String: any Sendable]]?,
        maxTokens: Int?, temperature: Double?
    ) -> AsyncStream<String> {
        // Single source of truth: stream the metered events and forward
        // only the text chunks (drop the terminal usage).
        let events = generateMetered(
            messages: messages, schemaJSON: schemaJSON, tools: tools,
            maxTokens: maxTokens, temperature: temperature)
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
        maxTokens: Int?, temperature: Double?
    ) -> AsyncStream<GenChunk> {
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
                for await chunk in chunks {
                    if completion >= cap {
                        truncated = true
                        break
                    }
                    continuation.yield(.text(chunk))
                    completion += Self.approxTokens(chunk)
                }
                continuation.yield(
                    .usage(
                        TokenUsage(
                            promptTokens: Self.approxTokens(prompt),
                            completionTokens: completion)))
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
public actor StubLLMModule: LLMModule {
    public nonisolated let id: ModuleID = .llm

    private let reserveBytes: Int
    private var loaded = false

    /// `reserveBytes` defaults to a representative multi-GB LLM footprint so
    /// budget pressure behaves realistically; tests inject small values.
    public init(reserveBytes: Int = 8 * 1024 * 1024 * 1024) {
        self.reserveBytes = reserveBytes
    }

    public var residentBytes: Int { loaded ? reserveBytes : 0 }

    public func memoryEstimate() -> Int { reserveBytes }

    public func load(reservation: MemoryReservation) async throws {
        loaded = true
    }

    public func unload() async {
        loaded = false
    }

    public nonisolated func generate(prompt: String) -> AsyncStream<String> {
        AsyncStream { continuation in
            let chunks = [
                "Athena ", "M0 ", "stub", ": ", "governed ", "memory ",
                "path ", "is ", "live", ".",
            ]
            let task = Task {
                for chunk in chunks {
                    continuation.yield(chunk)
                    try? await Task.sleep(nanoseconds: 15_000_000)
                }
                continuation.finish()
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }
}
