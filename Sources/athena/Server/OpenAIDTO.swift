import Foundation

// Minimal OpenAI-compatible DTOs for `/v1/chat/completions`. Only the fields
// the M0 path needs; widened as later milestones add tools / json_schema.

struct ChatMessage: Codable {
    let role: String
    let content: String
}

struct ChatCompletionRequest: Codable {
    let model: String?
    let messages: [ChatMessage]
    let stream: Bool?
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
