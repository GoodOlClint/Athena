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
    /// variant. Default bridges to the prompt path by flattening.
    nonisolated func generate(
        messages: [ChatTurn], schemaJSON: String?,
        tools: [[String: any Sendable]]?
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
            prompt: messages.flattenedPrompt(), schemaJSON: schemaJSON,
            tools: tools)
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
