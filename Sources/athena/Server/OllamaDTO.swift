import Foundation

// Ollama-compatible DTOs (M6). A single-model shim over Athena's
// governed modules — a requested `model` is echoed, not switched.

/// Version this shim reports to Ollama clients (format they parse).
enum OllamaShim { static let version = "0.7.14" }

struct OllamaMessage: Codable {
    let role: String
    let content: String
}

struct OllamaChatRequest: Codable {
    let model: String?
    let messages: [OllamaMessage]
    let stream: Bool?
}

struct OllamaChatResponse: Codable {
    let model: String
    let created_at: String
    let message: OllamaMessage
    let done: Bool
    let done_reason: String
}

struct OllamaGenerateRequest: Codable {
    let model: String?
    let prompt: String
    let stream: Bool?
}

struct OllamaGenerateResponse: Codable {
    let model: String
    let created_at: String
    let response: String
    let done: Bool
    let done_reason: String
}

/// `/api/embeddings` (legacy single `prompt` → single `embedding`).
struct OllamaEmbeddingsRequest: Codable {
    let model: String?
    let prompt: String
}
struct OllamaEmbeddingsResponse: Codable {
    let embedding: [Float]
}

/// `/api/embed` (newer: `input` string|[string] → `embeddings[][]`).
struct OllamaEmbedRequest: Codable {
    let model: String?
    let input: [String]

    private enum CodingKeys: String, CodingKey { case model, input }
    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        model = try c.decodeIfPresent(String.self, forKey: .model)
        if let one = try? c.decode(String.self, forKey: .input) {
            input = [one]
        } else {
            input = try c.decode([String].self, forKey: .input)
        }
    }
}
struct OllamaEmbedResponse: Codable {
    let model: String
    let embeddings: [[Float]]
}

struct OllamaModelInfo: Codable {
    let name: String
    let model: String
    let modified_at: String
    let size: Int
    let digest: String
}
struct OllamaTagsResponse: Codable {
    let models: [OllamaModelInfo]
}

struct OllamaVersionResponse: Codable {
    let version: String
}
