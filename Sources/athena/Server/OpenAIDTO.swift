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

/// OpenAI `stream_options` (M27.4). `include_usage:true` ⇒ a streamed
/// response emits a final chunk with empty `choices` and a populated
/// `usage` object before `data: [DONE]`.
struct StreamOptions: Codable {
    let include_usage: Bool?
}

struct ChatCompletionRequest: Codable {
    let model: String?
    let messages: [ChatMessage]
    let stream: Bool?
    let stream_options: StreamOptions?
    let response_format: ResponseFormat?
    let tools: [Tool]?
    let tool_choice: JSONValue?
    /// Per-request generation overrides (M24.3). Absent ⇒ the daemon's
    /// loaded defaults. Honored on the sync, native, and queued paths.
    ///
    /// `max_completion_tokens` is OpenAI's CURRENT field for the chat
    /// completions output cap; `max_tokens` is its deprecated predecessor.
    /// Current SDKs send `max_completion_tokens`, so a client setting only
    /// that previously had its cap SILENTLY DROPPED — the generation then
    /// ran to the daemon's (large) default, blowing past the requested
    /// limit. Both are parsed; `tokenCap` prefers the current field.
    let max_tokens: Int?
    let max_completion_tokens: Int?
    /// The effective output-token cap: the current `max_completion_tokens`
    /// when present, else the deprecated `max_tokens`. (Computed ⇒ not
    /// part of the Codable surface.)
    var tokenCap: Int? { max_completion_tokens ?? max_tokens }
    let temperature: Double?
    /// Sampling overrides (M31.3). `top_p`/`seed` reach the substrate
    /// sampling path; they are INERT on the greedy/MTP/structured paths
    /// (which are deterministic by construction — argmax decoding). `stop`
    /// truncates the output at the first matching sequence (string or
    /// array of strings) and reports finish_reason "stop".
    let top_p: Double?
    let seed: Int?
    let stop: JSONValue?
    /// Unsupported under Athena's greedy/MTP/structured determinism — a
    /// request carrying any of these is a 400 (see `unsupportedParameter`)
    /// rather than a silently-ignored param: `n>1` (one decode per
    /// request), `logprobs`/`top_logprobs` (the Guide masks the
    /// distribution), `logit_bias` (ditto).
    let n: Int?
    let logprobs: JSONValue?
    let top_logprobs: Int?
    let logit_bias: JSONValue?
    /// Athena extension — per-request MTP speculative decoding override.
    /// Absent ⇒ the daemon's loaded `--speculative` default. `true`
    /// opts this request into the MTP speculative path: the
    /// bit-identical-greedy loop at `temperature == 0`, the
    /// Leviathan/Chen sampling loop (distributionally identical to
    /// non-speculative sampling at the same temp/top_p/seed) at
    /// `temperature > 0`. Requires the loaded model to have an MTP
    /// head. `false` forces the standard non-speculative path even
    /// when the daemon was loaded with `--speculative`.
    let speculative: Bool?
    /// Athena extension (M46.3) — per-request inference deadline override
    /// in seconds. Overrides the daemon-wide `request_timeout_secs` for
    /// this single call. `nil` ⇒ inherit the daemon default. `0` (or
    /// negative) ⇒ disable the deadline for this call only — useful for
    /// long extraction-shape decodes that legitimately exceed a
    /// generous daemon cap. Honored on the sync, streamed, and queued
    /// `conversation` paths.
    let timeout: Int?
    /// OpenAI-compat extension (M46.3b) — opaque kwargs passed through
    /// to the model's chat template at rendering time. Maps to
    /// HuggingFace `tokenizer.apply_chat_template(..., **kwargs)` and
    /// reaches mlx-swift's `UserInput.additionalContext`. The canonical
    /// use is `{"enable_thinking": false}` on Qwen3-class models to
    /// suppress the `<think>…</think>` reasoning prefix on plain (no
    /// `response_format`) chat. Structured/tool-aware calls are
    /// no-think by construction anyway (the Guide masks from token 0),
    /// so the kwargs only matter on unstructured chat. nil ⇒ template
    /// runs with its built-in defaults.
    let chat_template_kwargs: [String: JSONValue]?

    /// M46.3b — lower the OpenAI-style `chat_template_kwargs` dict into
    /// the `[String: any Sendable]` shape mlx-swift's UserInput wants.
    /// Empty → nil so a missing field and an `{}` field both desugar
    /// to "use the template defaults," matching downstream
    /// expectations. nil/unknown variants drop their values (the kwarg
    /// dict is opaque to Athena; an unrecognized key reaches the
    /// template, which is responsible for ignoring or erroring on it).
    func chatTemplateKwargsContext() -> [String: any Sendable]? {
        guard let raw = chat_template_kwargs, !raw.isEmpty else {
            return nil
        }
        var out: [String: any Sendable] = [:]
        for (k, v) in raw {
            switch v {
            case .string(let s): out[k] = s
            case .number(let n): out[k] = n
            case .bool(let b): out[k] = b
            case .null, .array, .object: continue
            }
        }
        return out.isEmpty ? nil : out
    }

    /// Normalized stop sequences: OpenAI accepts a string or an array of
    /// strings (commonly ≤4). Empty/whitespace and non-string array
    /// members are dropped; capped at 4. [] ⇒ no stop handling.
    func stopSequences() -> [String] {
        let raw: [String]
        switch stop {
        case .string(let s): raw = [s]
        case .array(let a):
            raw = a.compactMap {
                if case .string(let s) = $0 { return s } else { return nil }
            }
        default: raw = []
        }
        return Array(raw.filter { !$0.isEmpty }.prefix(4))
    }

    /// The first OpenAI param present that Athena cannot honor under its
    /// greedy/MTP/structured determinism, for a clear 400. nil ⇒ all
    /// present params are supported.
    func unsupportedParameter() -> String? {
        if let n, n > 1 { return "n" }
        switch logprobs {
        case .bool(true): return "logprobs"
        case .number(let d) where d > 0: return "logprobs"
        default: break
        }
        if top_logprobs != nil { return "top_logprobs" }
        if case .object(let o)? = logit_bias, !o.isEmpty {
            return "logit_bias"
        }
        return nil
    }

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
    /// Present only on the terminal usage chunk when the client requested
    /// `stream_options.include_usage` (M27.4); nil ⇒ omitted from JSON.
    var usage: Usage? = nil
}

struct APIErrorBody: Codable {
    struct ErrorDetail: Codable {
        let message: String
        let type: String
        let code: String
    }
    let error: ErrorDetail
}

// MARK: - /v1/models (M31.1 — OpenAI list/retrieve)

/// One model in the OpenAI `model` shape. `created` is the store entry's
/// mtime as a unix epoch; `owned_by` is the appliance itself (the
/// goddess — never another tool's value). Backed by the same
/// `ModelStoreOps` the native `/api/models` reads.
struct OpenAIModel: Codable {
    let id: String
    let object: String  // "model"
    let created: Int
    let owned_by: String
}

struct OpenAIModelList: Codable {
    let object: String  // "list"
    let data: [OpenAIModel]
}

// MARK: - /v1/embeddings

/// `input` is a string or an array of strings (OpenAI also allows token
/// arrays — uncommon for this surface; those decode-fail → 400).
struct EmbeddingRequest: Decodable {
    let model: String?
    let input: [String]
    let encoding_format: String?

    private enum CodingKeys: String, CodingKey {
        case model, input, encoding_format
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        model = try c.decodeIfPresent(String.self, forKey: .model)
        encoding_format = try c.decodeIfPresent(
            String.self, forKey: .encoding_format)
        if let one = try? c.decode(String.self, forKey: .input) {
            input = [one]
        } else {
            input = try c.decode([String].self, forKey: .input)
        }
    }
}

struct EmbeddingObject: Codable {
    let object: String  // "embedding"
    let embedding: [Float]
    let index: Int
}

struct EmbeddingResponse: Codable {
    let object: String  // "list"
    let data: [EmbeddingObject]
    let model: String
    let usage: Usage
}

// MARK: - /v1/audio/transcriptions

/// OpenAI default (`response_format: "json"`) shape. `"text"` format
/// returns the bare string instead.
struct TranscriptionResponse: Codable {
    let text: String
}

/// One word with DTW-aligned timing (M26.2). OpenAI `word` shape.
struct WordTimestamp: Codable {
    let word: String
    let start: Double
    let end: Double
    let probability: Double
}

struct VerboseSegment: Codable {
    let id: Int
    let start: Double
    let end: Double
    let text: String
    /// Mean per-token log-probability for the span (M26.1) — omitted
    /// when not tracked (e.g. the stub engine).
    let avg_logprob: Double?
    /// Speaker id (M4.3c) — present only when the request opted into
    /// diarization (`diarize=true`); omitted otherwise.
    let speaker: Int?
    /// Words within this span (M26.2) — present only when word
    /// timestamps were requested.
    let words: [WordTimestamp]?
}

/// OpenAI `response_format: "verbose_json"` shape.
struct VerboseTranscriptionResponse: Codable {
    let task: String  // "transcribe"
    let language: String
    let duration: Double
    let text: String
    let segments: [VerboseSegment]
    /// All aligned words (M26.2) — present only when word timestamps
    /// were requested (`timestamp_granularities[]=word`).
    let words: [WordTimestamp]?
}

// MARK: - /v1/audio/diarizations (M4.3c, standalone)

struct DiarizationSegmentDTO: Codable {
    let start: Double
    let end: Double
    let speaker: Int
}

struct DiarizationResponse: Codable {
    let num_speakers: Int
    let segments: [DiarizationSegmentDTO]
}

// MARK: - /v1/audio/embeddings (M25.2 voice/speaker embeddings)

/// One requested segment (seconds from clip start). Decodable for the
/// `segments` multipart field; Encodable so it echoes back in the reply.
struct SpeakerSegmentSpec: Codable {
    let start: Double
    let end: Double
}

struct SpeakerEmbeddingObject: Codable {
    let object: String  // "speaker_embedding"
    let index: Int
    let segment: SpeakerSegmentSpec
    let embedding: [Float]
    let duration_seconds: Double
}

struct SpeakerEmbeddingResponse: Codable {
    let object: String  // "list"
    let data: [SpeakerEmbeddingObject]
    let model: String
    let dimension: Int
}

// MARK: - /v1/vectors (M7.2 built-in vector DB)

struct VectorUpsertRequest: Decodable {
    let id: String
    let vector: [Float]?
    let text: String?
    let metadata: JSONValue?
}
struct VectorIdResponse: Codable { let id: String }

struct VectorQueryRequest: Decodable {
    let vector: [Float]?
    let text: String?
    let k: Int?
}
struct VectorMatch: Codable {
    let id: String
    let score: Float
    let metadata: JSONValue?
}
struct VectorQueryResponse: Codable { let matches: [VectorMatch] }

struct VectorStatsResponse: Codable {
    let count: Int
    let dim: Int
    let bytes: Int
    let cap_bytes: Int
}

// MARK: - /v1/queue (M8.1 async request queue)

struct QueueSubmitResponse: Codable {
    let id: String
    let status: String  // "queued"
}

struct QueueStatusResponse: Codable {
    let id: String
    let kind: String
    let status: String  // queued|running|done|error|canceled
    let result: JSONValue?
    let error: String?
}

/// Stored job results (encoded into the job row; surfaced under
/// `result` on status). Conversation jobs store a full
/// `ChatCompletionResponse` (M24.6); embeddings store this. `model`
/// (M39) is the embedding model actually served for the job.
struct QueuedEmbeddingResult: Codable {
    let model: String
    let embeddings: [[Float]]
}

struct QueueJobSummary: Codable {
    let id: String
    let kind: String
    let status: String
    let created: Double
    let updated: Double
}
struct QueueListResponse: Codable { let jobs: [QueueJobSummary] }
struct QueueRemoveResponse: Codable { let id: String; let removed: Bool }

// MARK: - /v1/store (M9.3 shared-store admin)

struct StoreExportRequest: Decodable { let path: String }
struct StoreExportResponse: Codable {
    let path: String
    let bytes: Int
}
struct StoreStatsResponse: Codable {
    let vectors: Int
    let jobs: Int
    let bytes: Int
    let path: String
}
