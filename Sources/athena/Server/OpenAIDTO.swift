import AthenaStructured
import Foundation

// OpenAI-compatible DTOs for `/v1/chat/completions`. response_format /
// tools route into the M3 structured-output path; all new fields are
// optional so existing requests still decode.

struct ChatMessage: Codable {
    let role: String
    let content: String
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

    /// The schema string for the structured-output Guide, or nil for
    /// unconstrained generation.
    func structuredSchemaJSON() -> String? {
        StructuredSchema.schemaJSON(
            responseFormatType: response_format?.type,
            jsonSchema: response_format?.json_schema?.schema)
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
