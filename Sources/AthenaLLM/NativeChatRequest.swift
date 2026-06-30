import AthenaCore
import Foundation

/// The dialect-agnostic engine request (ADR 036). A protocol adapter — OpenAI
/// (`/v1/chat/completions`), Anthropic (`/v1/messages`), … — decodes its wire
/// request into this; the engine consumes it. It bundles exactly the parameters
/// of `generateMetered(...)` so a new dialect maps onto one explicit boundary
/// instead of the engine call's bare argument list, and so native knobs
/// (`speculative`, `chatTemplateKwargs`, …) have a single home each adapter
/// surfaces in its own idiom (ADR 036 §7, three-rung rule).
public struct NativeChatRequest: Sendable {
    public var messages: [ChatTurn]
    public var schemaJSON: String?
    public var tools: [[String: any Sendable]]?
    public var maxTokens: Int?
    public var temperature: Double?
    public var topP: Double?
    public var seed: Int?
    public var speculative: Bool?
    public var chatTemplateKwargs: [String: any Sendable]?
    public var promptCacheKey: String?
    public var principal: String?
    public var logprobs: LogprobsRequest?

    public init(
        messages: [ChatTurn], schemaJSON: String? = nil,
        tools: [[String: any Sendable]]? = nil,
        maxTokens: Int? = nil, temperature: Double? = nil,
        topP: Double? = nil, seed: Int? = nil,
        speculative: Bool? = nil,
        chatTemplateKwargs: [String: any Sendable]? = nil,
        promptCacheKey: String? = nil, principal: String? = nil,
        logprobs: LogprobsRequest? = nil
    ) {
        self.messages = messages
        self.schemaJSON = schemaJSON
        self.tools = tools
        self.maxTokens = maxTokens
        self.temperature = temperature
        self.topP = topP
        self.seed = seed
        self.speculative = speculative
        self.chatTemplateKwargs = chatTemplateKwargs
        self.promptCacheKey = promptCacheKey
        self.principal = principal
        self.logprobs = logprobs
    }
}

extension LLMModule {
    /// ADR 036 — consume a `NativeChatRequest` at the engine boundary. A thin
    /// forward to the canonical `generateMetered(...)` so every protocol adapter
    /// shares one call site. Behaviour is byte-identical to spelling out the
    /// argument list (every conformer inherits this; the engine call is
    /// unchanged).
    public nonisolated func generateMetered(
        _ req: NativeChatRequest
    ) -> AsyncStream<GenChunk> {
        generateMetered(
            messages: req.messages, schemaJSON: req.schemaJSON,
            tools: req.tools, maxTokens: req.maxTokens,
            temperature: req.temperature, topP: req.topP, seed: req.seed,
            speculative: req.speculative,
            chatTemplateKwargs: req.chatTemplateKwargs,
            promptCacheKey: req.promptCacheKey, principal: req.principal,
            logprobs: req.logprobs)
    }
}
