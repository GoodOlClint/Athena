/// True per-request token accounting for a single generation (M27.1).
/// `prompt` is the tokenized input length the model actually consumed;
/// `completion` is the number of tokens it emitted. Carried out of the
/// generate path so the OpenAI `usage` object and the metrics/metering
/// counters reflect real work instead of hardcoded zeros.
public struct TokenUsage: Sendable, Equatable {
    public var promptTokens: Int
    public var completionTokens: Int
    public init(promptTokens: Int, completionTokens: Int) {
        self.promptTokens = promptTokens
        self.completionTokens = completionTokens
    }
    public var totalTokens: Int { promptTokens + completionTokens }
    public static let zero = TokenUsage(promptTokens: 0, completionTokens: 0)
}

/// One element of the metered generation stream (M27.1). Text chunks
/// stream exactly as the String `generate` overloads yield them; a
/// single terminal `.usage` carries the true token counts once the
/// model has finished. Callers that don't need usage consume the
/// String `generate` overloads, which are thin filters over this.
public enum GenChunk: Sendable {
    case text(String)
    case usage(TokenUsage)
}
