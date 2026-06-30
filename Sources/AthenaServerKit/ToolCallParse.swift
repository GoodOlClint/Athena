import Foundation

/// Parse a Guide-constrained tool-call object into the OpenAI
/// `(name, stringified-arguments)` pair. The model may emit `arguments`
/// before `name` (Gemma 4 does) — JSON key order is irrelevant here.
/// `nil` ⇒ not a parseable tool call (caller falls back to plain content).
///
/// Lives in AthenaServerKit (MLX-free) so the streaming/non-streaming tool-call
/// shape is unit-pinned without `@testable`-importing the MLX-linked executable
/// (ADR 008/009). Both `chatChoice` (non-stream) and `pumpTokens` (stream) call
/// this so they surface the identical tool call.
public func parseToolCall(
    _ text: String
) -> (name: String, argsJSON: String)? {
    guard let data = text.data(using: .utf8),
        let obj = try? JSONSerialization.jsonObject(with: data)
            as? [String: Any],
        let name = obj["name"] as? String
    else { return nil }
    let args = obj["arguments"] ?? [String: Any]()
    let argsJSON =
        (try? JSONSerialization.data(
            withJSONObject: args, options: [.sortedKeys]))
        .flatMap { String(data: $0, encoding: .utf8) } ?? "{}"
    return (name, argsJSON)
}
