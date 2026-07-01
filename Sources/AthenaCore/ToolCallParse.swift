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
    return (name, toolArgumentsJSON(obj["arguments"]))
}

/// The tool-call precedence one of the four chat encode sites (OpenAI /
/// Anthropic × streaming / non-streaming) resolves to. ADR 034/036: a
/// substrate-`detected` free call (`tool_choice:auto`) wins and leaves the
/// assistant text intact as content/reasoning; otherwise a Guide-`forced`
/// call is the decoded JSON text itself (so the text IS the call, not
/// content); otherwise plain text.
public enum ToolCallOutcome: Equatable {
    case detected(name: String, argsJSON: String)
    case forced(name: String, argsJSON: String)
    case none
}

/// The single tool-call precedence algebra all four chat encode sites switch
/// on (WP7). Was copy-pasted at each of `chatChoice`, the Anthropic
/// non-stream encoder, and both SSE pumps' terminals; drift between the copies
/// is exactly the class ADR 036 promised to close. MLX-free + unit-pinned
/// (ADR 008/009).
public func resolveToolCallOutcome(
    detected: (name: String, argsJSON: String)?,
    text: String,
    isToolCall: Bool
) -> ToolCallOutcome {
    if let d = detected {
        return .detected(name: d.name, argsJSON: d.argsJSON)
    }
    if isToolCall, let p = parseToolCall(text) {
        return .forced(name: p.name, argsJSON: p.argsJSON)
    }
    return .none
}

/// Serialize a tool call's `arguments` object to the OpenAI stringified-JSON
/// form with stable (sorted) keys. nil / non-object ⇒ `"{}"`. Shared by
/// `parseToolCall` (Guide-forced text) and the substrate `.toolCall` mapping
/// (ADR 034) so both surface byte-identical `arguments`.
public func toolArgumentsJSON(_ arguments: Any?) -> String {
    let args = arguments ?? [String: Any]()
    // `JSONSerialization.data(withJSONObject:)` raises an *ObjC exception*
    // (uncatchable by `try?`) when the top level isn't an object/array, so
    // gate on `isValidJSONObject` first — a bare string/number ⇒ "{}".
    guard JSONSerialization.isValidJSONObject(args),
        let data = try? JSONSerialization.data(
            withJSONObject: args, options: [.sortedKeys]),
        let s = String(data: data, encoding: .utf8)
    else { return "{}" }
    return s
}
