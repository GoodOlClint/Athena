import AthenaCore
import AthenaLLM
import Foundation

extension LLMModule {
    /// Test helper: run `generateMetered` and collect just the `.text`
    /// chunks — replaces the deleted convenience String `generate` overloads
    /// (the schema/tools/override variants). Plain-prompt tests still use
    /// `generate(prompt:)` directly. Called through the protocol, so every
    /// `generateMetered` argument is passed explicitly (no extension defaults
    /// apply in a generic context).
    func generatedText(
        prompt: String,
        schemaJSON: String? = nil,
        tools: [[String: any Sendable]]? = nil,
        maxTokens: Int? = nil,
        speculative: Bool? = nil
    ) async -> String {
        var out = ""
        for await event in generateMetered(
            messages: [ChatTurn(role: "user", content: prompt)],
            schemaJSON: schemaJSON, tools: tools,
            maxTokens: maxTokens, temperature: nil,
            topP: nil, seed: nil, speculative: speculative,
            chatTemplateKwargs: nil, promptCacheKey: nil, principal: nil,
            logprobs: nil, requestedModel: nil)
        {
            if case .text(let t) = event { out += t }
        }
        return out
    }
}
