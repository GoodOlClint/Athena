import AthenaLLM
import AthenaStructured
import Foundation

// ADR 036 S2 — the Anthropic Messages (`POST /v1/messages`) protocol adapter.
// DECODE the Anthropic wire request into the dialect-agnostic `NativeChatRequest`
// (+ requestedModel/stops/stream) and ENCODE the engine's drained generation
// back into the Anthropic response shape. No orchestration and no engine logic
// live here — the handler runs the shared `prepareChat` seam and drains/forwards
// the one `GenChunk` stream, exactly as the OpenAI adapter does. First cut =
// what Claude Code drives: text + system + tool_use/tool_result + tools; image /
// document content blocks and prompt-caching are refused with a cause-naming
// 400 (deferred, ADR 036).

// MARK: - Request

/// A decode fault the handler renders as an Anthropic-shaped 400. `message`
/// names the structural reason; `code` is the Athena error `code`.
struct AnthropicDecodeError: Error {
    let message: String
    let code: String
}

/// `system`, message `content`, and `tool_result.content` are each "a string
/// OR an array of blocks" in the Anthropic wire format. This decodes both.
enum AnthropicContent: Decodable {
    case text(String)
    case blocks([AnthropicBlock])

    init(from decoder: Decoder) throws {
        let c = try decoder.singleValueContainer()
        if let s = try? c.decode(String.self) {
            self = .text(s)
        } else {
            self = .blocks(try c.decode([AnthropicBlock].self))
        }
    }

    /// Flatten to plain text (text/tool_result-text blocks joined). Tool-use
    /// blocks contribute no text.
    var plainText: String {
        switch self {
        case .text(let s): return s
        case .blocks(let bs):
            return bs.compactMap { $0.textValue }.joined()
        }
    }
}

/// One Anthropic content block. Only the block kinds the first cut supports are
/// modelled; an `image`/`document` (or any unknown `type`) decodes to
/// `.unsupported(type)` so the handler can 400 with the offending type named.
enum AnthropicBlock: Decodable {
    case text(String)
    case toolUse(id: String, name: String, input: JSONValue)
    case toolResult(toolUseID: String, content: AnthropicContent)
    case unsupported(String)

    private enum Key: String, CodingKey {
        case type, text, id, name, input, tool_use_id, content
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: Key.self)
        let type = try c.decode(String.self, forKey: .type)
        switch type {
        case "text":
            self = .text(try c.decode(String.self, forKey: .text))
        case "tool_use":
            self = .toolUse(
                id: try c.decode(String.self, forKey: .id),
                name: try c.decode(String.self, forKey: .name),
                input: (try? c.decode(JSONValue.self, forKey: .input))
                    ?? .object([:]))
        case "tool_result":
            self = .toolResult(
                toolUseID: try c.decode(String.self, forKey: .tool_use_id),
                content: (try? c.decode(AnthropicContent.self, forKey: .content))
                    ?? .text(""))
        default:
            self = .unsupported(type)
        }
    }

    var textValue: String? {
        switch self {
        case .text(let s): return s
        case .toolResult(_, let content): return content.plainText
        default: return nil
        }
    }
}

struct AnthropicMessage: Decodable {
    let role: String
    let content: AnthropicContent
}

struct AnthropicTool: Decodable {
    let name: String
    let description: String?
    let input_schema: JSONValue?
}

/// `tool_choice`: `{"type":"auto"|"any"|"tool"|"none", "name":...}`.
struct AnthropicToolChoice: Decodable {
    let type: String
    let name: String?
}

struct AnthropicMessagesRequest: Decodable {
    let model: String
    let max_tokens: Int
    let system: AnthropicContent?
    let messages: [AnthropicMessage]
    let tools: [AnthropicTool]?
    let tool_choice: AnthropicToolChoice?
    let stop_sequences: [String]?
    let temperature: Double?
    let top_p: Double?
    let stream: Bool?

    /// The result of lowering an Anthropic request to the engine boundary.
    struct Lowered {
        let native: NativeChatRequest
        let stops: [String]
        let wantStream: Bool
        /// Whether the forcing Guide is engaged (a `tool_choice` of `any`/`tool`)
        /// — the encoder uses this to parse a Guide-forced tool call out of text.
        let isToolCall: Bool
    }

    /// Decode → native (ADR 036). Throws `AnthropicDecodeError` for an
    /// unsupported content block (image/document) so the handler 400s with the
    /// type named, never a silent drop (passive-oracle, honesty boundary).
    func lower(principal: String?) throws -> Lowered {
        var turns: [ChatTurn] = []
        // Anthropic carries the system prompt out-of-band; the chat template
        // sees it as a leading system turn.
        if let system, !system.plainText.isEmpty {
            turns.append(ChatTurn(role: "system", content: system.plainText))
        }
        for m in messages {
            try Self.appendTurns(role: m.role, content: m.content, into: &turns)
        }

        // tools → the substrate ToolSpec menu shape (same as OpenAI's toolSpecs).
        let advertiseMenu = (tool_choice?.type != "none")
        let toolSpecs: [[String: any Sendable]]? = {
            guard let tools, !tools.isEmpty, advertiseMenu else { return nil }
            return tools.map { t in
                var fn: [String: any Sendable] = ["name": t.name]
                if let d = t.description { fn["description"] = d }
                fn["parameters"] =
                    (t.input_schema ?? .object(["type": .string("object")]))
                    .foundationValue()
                return ["type": "function", "function": fn]
            }
        }()

        // tool_choice → forcing schema (mirrors OpenAI effectiveSchema):
        // auto/none ⇒ no forcing (model decides; substrate detects a free call);
        // any ⇒ force the union of all tools; tool ⇒ force the named one.
        var schemaJSON: String?
        var isToolCall = false
        if let tc = tool_choice, let tools, !tools.isEmpty {
            switch tc.type {
            case "any":
                schemaJSON = StructuredSchema.toolCallUnionSchema(
                    tools: tools.map { ($0.name, $0.input_schema) })
                isToolCall = schemaJSON != nil
            case "tool":
                if let name = tc.name,
                    let t = tools.first(where: { $0.name == name })
                {
                    schemaJSON = StructuredSchema.toolCallSchema(
                        functionName: t.name, parameters: t.input_schema)
                    isToolCall = schemaJSON != nil
                }
            default: break  // auto / none
            }
        }

        let native = NativeChatRequest(
            messages: turns, schemaJSON: schemaJSON, tools: toolSpecs,
            maxTokens: max_tokens, temperature: temperature, topP: top_p,
            seed: nil, speculative: nil, chatTemplateKwargs: nil,
            promptCacheKey: nil, principal: principal, logprobs: nil)
        return Lowered(
            native: native, stops: stop_sequences ?? [],
            wantStream: stream == true, isToolCall: isToolCall)
    }

    /// Lower one Anthropic message into native `ChatTurn`s. An assistant turn
    /// folds its text + `tool_use` blocks into one turn (content + toolCalls);
    /// a user turn's `tool_result` blocks each become a `tool` turn (carrying
    /// the `tool_use_id`), and any user text becomes a `user` turn.
    private static func appendTurns(
        role: String, content: AnthropicContent, into turns: inout [ChatTurn]
    ) throws {
        switch content {
        case .text(let s):
            turns.append(ChatTurn(role: role, content: s))
        case .blocks(let blocks):
            var text = ""
            var toolCalls: [ChatToolCall] = []
            var toolResults: [(id: String, content: String)] = []
            for b in blocks {
                switch b {
                case .text(let s): text += s
                case .toolUse(let id, let name, let input):
                    toolCalls.append(
                        ChatToolCall(
                            id: id, name: name,
                            argumentsJSON: input.jsonString() ?? "{}"))
                case .toolResult(let toolUseID, let c):
                    toolResults.append((toolUseID, c.plainText))
                case .unsupported(let t):
                    throw AnthropicDecodeError(
                        message:
                            "content block type '\(t)' is not supported "
                            + "(text, tool_use, and tool_result only)",
                        code: "unsupported_content_block")
                }
            }
            if role == "assistant" {
                // ADR 034 — an assistant turn carries its tool_calls so the chat
                // template renders a coherent call→result history.
                if !text.isEmpty || !toolCalls.isEmpty {
                    turns.append(
                        ChatTurn(
                            role: "assistant", content: text,
                            toolCalls: toolCalls))
                }
            } else {
                if !text.isEmpty {
                    turns.append(ChatTurn(role: "user", content: text))
                }
                for r in toolResults {
                    turns.append(
                        ChatTurn(
                            role: "tool", content: r.content,
                            toolCallID: r.id))
                }
            }
        }
    }
}

// MARK: - Error

/// Anthropic error envelope: `{"type":"error","error":{"type","message"}}` —
/// distinct from Athena's canonical `{"error":{message,type,code}}`. The inner
/// `type` is derived from the HTTP status.
struct AnthropicErrorBody: Encodable {
    struct Inner: Encodable {
        let type: String
        let message: String
    }
    let type = "error"
    let error: Inner

    /// Map an HTTP status to Anthropic's error `type` taxonomy.
    static func errorType(forStatus code: Int) -> String {
        switch code {
        case 400: return "invalid_request_error"
        case 401: return "authentication_error"
        case 403: return "permission_error"
        case 404: return "not_found_error"
        case 413: return "request_too_large"
        case 429: return "rate_limit_error"
        case 500: return "api_error"
        case 503, 529: return "overloaded_error"
        default: return "api_error"
        }
    }
}

// MARK: - Response (non-streaming)

/// One Anthropic response content block (`text` or `tool_use`).
struct AnthropicResponseBlock: Encodable {
    let type: String
    var text: String?
    var id: String?
    var name: String?
    var input: JSONValue?
}

struct AnthropicUsage: Encodable {
    let input_tokens: Int
    let output_tokens: Int
}

struct AnthropicMessagesResponse: Encodable {
    let id: String
    let type = "message"
    let role = "assistant"
    let model: String
    let content: [AnthropicResponseBlock]
    let stop_reason: String
    let stop_sequence: String?
    let usage: AnthropicUsage

    /// Build from a drained generation (`GenCollected`-equivalent fields). A
    /// detected/Guide-forced tool call becomes a `tool_use` block + `stop_reason
    /// "tool_use"`; a stop-sequence hit ⇒ `"stop_sequence"`; a length cap ⇒
    /// `"max_tokens"`; otherwise `"end_turn"`.
    static func make(
        id: String, model: String, text: String,
        toolCall: (name: String, argsJSON: String)?,
        promptTokens: Int, completionTokens: Int,
        finishIsLength: Bool, stopSequenceHit: String?
    ) -> AnthropicMessagesResponse {
        var blocks: [AnthropicResponseBlock] = []
        var stopReason = "end_turn"
        if let call = toolCall {
            if !text.isEmpty {
                blocks.append(AnthropicResponseBlock(type: "text", text: text))
            }
            // Anthropic `input` is a JSON object, not the OpenAI stringified
            // args — parse the args JSON back into a value (empty object on a
            // parse miss, never a crash).
            let input: JSONValue =
                (call.argsJSON.data(using: .utf8).flatMap {
                    try? JSONDecoder().decode(JSONValue.self, from: $0)
                }) ?? .object([:])
            blocks.append(
                AnthropicResponseBlock(
                    type: "tool_use", id: "toolu_\(UUID().uuidString)",
                    name: call.name, input: input))
            stopReason = "tool_use"
        } else {
            blocks.append(AnthropicResponseBlock(type: "text", text: text))
            if let s = stopSequenceHit {
                stopReason = "stop_sequence"
                return AnthropicMessagesResponse(
                    id: id, model: model, content: blocks,
                    stop_reason: stopReason, stop_sequence: s,
                    usage: AnthropicUsage(
                        input_tokens: promptTokens,
                        output_tokens: completionTokens))
            }
            if finishIsLength { stopReason = "max_tokens" }
        }
        return AnthropicMessagesResponse(
            id: id, model: model, content: blocks, stop_reason: stopReason,
            stop_sequence: nil,
            usage: AnthropicUsage(
                input_tokens: promptTokens, output_tokens: completionTokens))
    }
}
