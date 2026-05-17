import AthenaStructured
import Foundation

// OpenAI-compatible DTOs for `/v1/chat/completions`. response_format /
// tools route into the M3 structured-output path; all new fields are
// optional so existing requests still decode.

struct FunctionCallOut: Codable {
    let name: String
    let arguments: String  // JSON-encoded argument object (OpenAI shape)
}

struct ToolCallOut: Codable {
    let id: String
    let type: String  // "function"
    let function: FunctionCallOut
}

struct ChatMessage: Codable {
    let role: String
    let content: String?  // null for a tool-call response
    var tool_calls: [ToolCallOut]?

    init(role: String, content: String?, tool_calls: [ToolCallOut]? = nil) {
        self.role = role
        self.content = content
        self.tool_calls = tool_calls
    }
}

struct JSONSchemaSpec: Codable {
    let name: String?
    let schema: JSONValue?
    let strict: Bool?
}

struct ResponseFormat: Codable {
    let type: String  // "text" | "json_object" | "json_schema"
    let json_schema: JSONSchemaSpec?
}

struct FunctionDef: Codable {
    let name: String
    let description: String?
    let parameters: JSONValue?
}

struct Tool: Codable {
    let type: String
    let function: FunctionDef
}

struct ChatCompletionRequest: Codable {
    let model: String?
    let messages: [ChatMessage]
    let stream: Bool?
    let response_format: ResponseFormat?
    let tools: [Tool]?
    let tool_choice: JSONValue?

    /// Tools to constrain to: `tool_choice` forcing one ⇒ `[that]`;
    /// `"none"` / no tools ⇒ nil; otherwise (auto/absent/"required") ⇒
    /// all declared tools (the caller compiles a single-fn schema for
    /// one tool, a `oneOf` union for many — free multi-tool choice). A
    /// forced name matching nothing ⇒ nil (falls through to
    /// response_format / unconstrained, preserving prior behavior).
    private func selectedTools() -> [Tool]? {
        guard let tools, !tools.isEmpty else { return nil }
        if case .string(let s)? = tool_choice,
            s == "none" { return nil }
        if case .object(let o)? = tool_choice,
            case .object(let f)? = o["function"],
            case .string(let name)? = f["name"]
        {
            return tools.first { $0.function.name == name }.map { [$0] }
        }
        return tools
    }

    /// All declared tools, lowered to the substrate `ToolSpec` shape
    /// (`{"type":"function","function":{name,description,parameters}}`),
    /// for the model's chat template. ALL tools are advertised even when
    /// the Guide constrains output to one — the model still needs the
    /// tool-call format and the full menu. nil ⇒ no tools.
    func toolSpecs() -> [[String: any Sendable]]? {
        guard let tools, !tools.isEmpty else { return nil }
        return tools.map { t in
            var fn: [String: any Sendable] = ["name": t.function.name]
            if let d = t.function.description { fn["description"] = d }
            fn["parameters"] =
                (t.function.parameters
                    ?? .object(["type": .string("object")]))
                .foundationValue()
            return ["type": "function", "function": fn]
        }
    }

    /// The constraining schema for the structured-output Guide and
    /// whether it is a tool call (tools take precedence over
    /// response_format). nil ⇒ unconstrained.
    func effectiveSchema() -> (json: String, isToolCall: Bool)? {
        if let ts = selectedTools(), !ts.isEmpty {
            let schema =
                ts.count == 1
                ? StructuredSchema.toolCallSchema(
                    functionName: ts[0].function.name,
                    parameters: ts[0].function.parameters)
                : StructuredSchema.toolCallUnionSchema(
                    tools: ts.map {
                        ($0.function.name, $0.function.parameters)
                    })
            if let schema { return (schema, true) }
        }
        if let s = StructuredSchema.schemaJSON(
            responseFormatType: response_format?.type,
            jsonSchema: response_format?.json_schema?.schema)
        {
            return (s, false)
        }
        return nil
    }
}

struct ChatChoice: Codable {
    let index: Int
    let message: ChatMessage
    let finish_reason: String
}

struct Usage: Codable {
    let prompt_tokens: Int
    let completion_tokens: Int
    let total_tokens: Int
}

struct ChatCompletionResponse: Codable {
    let id: String
    let object: String
    let created: Int
    let model: String
    let choices: [ChatChoice]
    let usage: Usage
}

struct ChatDelta: Codable {
    let role: String?
    let content: String?
}

struct ChatChunkChoice: Codable {
    let index: Int
    let delta: ChatDelta
    let finish_reason: String?
}

struct ChatCompletionChunk: Codable {
    let id: String
    let object: String
    let created: Int
    let model: String
    let choices: [ChatChunkChoice]
}

struct APIErrorBody: Codable {
    struct ErrorDetail: Codable {
        let message: String
        let type: String
        let code: String
    }
    let error: ErrorDetail
}
