import AthenaCore

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

/// Why a generation stopped (M31.2). `length` ⇒ the model hit the
/// effective `max_tokens` cap and was truncated; `stop` ⇒ it ended
/// naturally (EOS / structured-output completion). Maps to the OpenAI
/// `finish_reason` (the server upgrades `stop` to `tool_calls` when the
/// emitted text is a tool call — that determination is the server's, not
/// the module's). Raw value IS the OpenAI string.
public enum FinishReason: String, Sendable {
    case stop
    case length
}

/// One element of the metered generation stream (M27.1). Text chunks
/// stream exactly as the String `generate` overloads yield them; a
/// single terminal `.usage` carries the true token counts, followed by a
/// single terminal `.finish` carrying the stop reason (M31.2), once the
/// model has finished. Callers that don't need usage/finish consume the
/// String `generate` overloads, which are thin filters over this.
///
/// M49.5.2 — `.error` is a terminal alternative to `.finish`: the
/// generation aborted before producing a complete result, the carried
/// `AthenaError` is already classified, and the consumer MUST re-throw
/// it so the HTTP layer can return the right status / code. Before
/// M49.5.2, a thrown classified error inside `generateMetered`'s catch
/// became a fake `.text("[athena: generation failed: ...]")` event and
/// the request returned 200 with the error stringified into the chat
/// content — caught when v0.10.84 classified schema-complexity 400 (removed in M53) came back
/// as a 200 with the error message in `choices[0].message.content`.
public enum GenChunk: Sendable {
    case text(String)
    case usage(TokenUsage)
    case finish(FinishReason)
    case error(AthenaError)
}
