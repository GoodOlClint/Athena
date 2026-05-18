import Foundation

// Athena-native `/api/*` JSON (M16). Deliberately NOT Ollama and NOT
// OpenAI: a clean, minimal request/response with no vendor-mimicking
// envelope (`created_at`, `done_reason`, `object`, `choices`, …). The
// `/v1/*` surface remains OpenAI-compatible; `/api/*` is Athena's own
// dialect. Errors reuse the standard `{"error":{message,type,code}}`
// body (`APIErrorBody`) — not dialect-specific.

struct AthenaChatMessage: Codable {
    let role: String
    let content: String
}

struct AthenaChatRequest: Codable {
    let model: String?
    let messages: [AthenaChatMessage]
    let stream: Bool?
}

/// Non-streamed `/api/chat` reply. The full generation is in
/// `content`; `done` is always true here (a single object).
struct AthenaChatResponse: Codable {
    let model: String
    let content: String
    let done: Bool
}

/// One NDJSON line of a streamed `/api/chat`: an incremental
/// `content` piece, then a final `{content:"",done:true}` line.
struct AthenaChatChunk: Codable {
    let content: String
    let done: Bool
}

/// `/api/embed` — `input` is a string or an array of strings.
struct AthenaEmbedRequest: Codable {
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
    init(model: String?, input: [String]) {
        self.model = model
        self.input = input
    }
}

struct AthenaEmbedResponse: Codable {
    let model: String
    let embeddings: [[Float]]
}

/// `/api/admin/stop` — model unloaded; daemon keeps running.
struct AthenaStopResponse: Codable {
    let status: String  // "unloaded"
    let model: String
}
