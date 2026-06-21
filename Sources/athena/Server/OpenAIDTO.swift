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
    /// M71.1 (vision input): the `image_url.url` strings extracted from an
    /// OpenAI content-parts array. Empty for a plain-string content. Populated
    /// only on DECODE; not part of the response-encoding surface (responses
    /// always carry a plain-string `content`).
    var imageURLs: [String]

    init(
        role: String, content: String?, tool_calls: [ToolCallOut]? = nil,
        imageURLs: [String] = []
    ) {
        self.role = role
        self.content = content
        self.tool_calls = tool_calls
        self.imageURLs = imageURLs
    }

    private enum CodingKeys: String, CodingKey {
        case role, content, tool_calls
    }

    /// `content` decodes as EITHER a plain string (unchanged) OR an OpenAI
    /// content-parts array (`[{type:"text",text} | {type:"image_url",...}]`).
    /// Text parts are flattened (newline-joined) into `content`; image parts'
    /// URLs land in `imageURLs`. Decoding the URLs into bytes (and the
    /// passive-oracle reject of `http(s)`) happens at the handler via
    /// `ChatImage.fromImageURL`.
    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        role = try c.decode(String.self, forKey: .role)
        tool_calls = try c.decodeIfPresent([ToolCallOut].self, forKey: .tool_calls)
        if !c.contains(.content)
            || ((try? c.decodeNil(forKey: .content)) == true)
        {
            content = nil
            imageURLs = []
        } else if let s = try? c.decode(String.self, forKey: .content) {
            content = s
            imageURLs = []
        } else {
            let parts = try c.decode([ContentPart].self, forKey: .content)
            let texts = parts.compactMap(\.text)
            content = texts.isEmpty ? nil : texts.joined(separator: "\n")
            imageURLs = parts.compactMap(\.imageURL)
        }
    }

    func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(role, forKey: .role)
        try c.encodeIfPresent(content, forKey: .content)
        try c.encodeIfPresent(tool_calls, forKey: .tool_calls)
    }
}

/// One OpenAI chat content-part (M71.1). `type` is "text" or "image_url".
private struct ContentPart: Codable {
    let type: String
    let text: String?
    let image_url: ContentPartImageURL?
    /// The image URL string when this is an image part (else nil).
    var imageURL: String? { image_url?.url }
}

private struct ContentPartImageURL: Codable {
    let url: String
    let detail: String?
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
    /// OpenAI-standard prompt-caching hint (M59.3). An opaque caller-chosen
    /// string that scopes the cross-request prompt-prefix cache so requests
    /// sharing a prefix also share a cache key, raising the hit rate. Absent
    /// ⇒ the cache scopes by the authenticated principal alone. Honored on
    /// the sync, streamed, and queued `conversation` paths.
    let prompt_cache_key: String?

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
            case .integer(let i): out[k] = i
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
        // C2 (ADR 013 §4): `logprobs`/`top_logprobs` are now HONORED on the
        // deterministic path (see `wantsLogprobs`/`logprobsValidationError`),
        // no longer rejected here.
        if case .object(let o)? = logit_bias, !o.isEmpty {
            return "logit_bias"
        }
        return nil
    }

    /// C2 — did the caller ask for logprobs? (`logprobs:true`, or the legacy
    /// integer/number forms some clients still send.)
    var wantsLogprobs: Bool {
        switch logprobs {
        case .bool(true): return true
        case .integer(let i) where i > 0: return true
        case .number(let d) where d > 0: return true
        default: return false
        }
    }

    /// C2 — the requested number of top alternatives per token (OpenAI
    /// `top_logprobs`); 0 when only `logprobs:true` is set.
    var topLogprobsValue: Int { top_logprobs ?? 0 }

    /// C2 — validate a logprobs request against OpenAI's rules + Athena's
    /// determinism boundary (ADR 013 §4). Returns `(message, code)` for a 400,
    /// or nil if the request is acceptable. `deterministic` is true when the
    /// request will decode greedily/structured (temp==0 or a schema present) —
    /// logprobs require it, since the sampling/substrate-stream path has no
    /// logit-capture seam.
    func logprobsValidationError(deterministic: Bool) -> (String, String)? {
        if top_logprobs != nil && !wantsLogprobs {
            return (
                "'top_logprobs' requires 'logprobs' to be true.",
                "invalid_top_logprobs")
        }
        guard wantsLogprobs else { return nil }
        if let k = top_logprobs, k < 0 || k > 20 {
            return (
                "'top_logprobs' must be between 0 and 20.",
                "invalid_top_logprobs")
        }
        if !deterministic {
            return (
                "'logprobs' is only supported on the deterministic decode "
                    + "path: set 'temperature' to 0, or use structured output "
                    + "(response_format / tools).",
                "logprobs_requires_deterministic")
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

    /// G4: validate a structured request up front. A `response_format` of
    /// `json_schema` whose `schema` is missing or unserializable must 400
    /// — never fall through to `effectiveSchema() == nil` and stream
    /// unconstrained output (a silent breach of the structured-output
    /// contract). Returns a problem description for the malformed case,
    /// else nil. Tools take precedence and carry their own schema, so an
    /// in-effect tool call is fine.
    func structuredRequestError() -> String? {
        guard response_format?.type == "json_schema" else { return nil }
        if let ts = selectedTools(), !ts.isEmpty { return nil }
        if StructuredSchema.schemaJSON(
            responseFormatType: "json_schema",
            jsonSchema: response_format?.json_schema?.schema) == nil
        {
            return "response_format.json_schema requires a valid, "
                + "serializable 'schema' object"
        }
        return nil
    }
}

/// C2 (ADR 013 §4) — OpenAI chat `logprobs` response object. One
/// `ChatCompletionTokenLogprob` per generated token, in order.
struct ChatLogprobs: Codable {
    let content: [ChatCompletionTokenLogprob]
}

struct ChatCompletionTokenLogprob: Codable {
    let token: String
    let logprob: Double
    let bytes: [Int]?
    let top_logprobs: [ChatTopLogprob]
}

struct ChatTopLogprob: Codable {
    let token: String
    let logprob: Double
    let bytes: [Int]?
}

struct ChatChoice: Codable {
    let index: Int
    let message: ChatMessage
    let finish_reason: String
    // C2 — nil (omitted) unless the request asked for logprobs.
    var logprobs: ChatLogprobs? = nil
}

/// OpenAI `usage.prompt_tokens_details` (M59.3). Reports how many input
/// tokens were served from the prompt-prefix cache. Omitted from JSON when
/// nil (e.g. embeddings), present (possibly 0) on chat completions.
struct PromptTokensDetails: Codable {
    let cached_tokens: Int
}

struct Usage: Codable {
    let prompt_tokens: Int
    let completion_tokens: Int
    let total_tokens: Int
    let prompt_tokens_details: PromptTokensDetails?

    init(
        prompt_tokens: Int, completion_tokens: Int, total_tokens: Int,
        cachedTokens: Int? = nil
    ) {
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens
        self.prompt_tokens_details =
            cachedTokens.map { PromptTokensDetails(cached_tokens: $0) }
    }
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
    // C2 — per-token logprobs for a streamed chunk (omitted unless requested).
    var logprobs: ChatLogprobs? = nil
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
        /// #12 / M43.4 — optional operator-facing remediation. nil ⇒
        /// omitted from JSON (encodeIfPresent), so the canonical
        /// `{message,type,code}` envelope is unchanged for the common
        /// case. The CLI client renders it as a `hint:` line; non-CLI
        /// consumers ignore it. Mirrors the auth-middleware deny hint.
        var hint: String? = nil
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

// ADR 025 S2 — the async request queue (`/v1/queue*`) and its DTOs were
// removed entirely. Inference is synchronous (`/v1/chat/completions`,
// `/v1/embeddings`, …); model lifecycle ops stream SSE progress on
// `/api/models/{pull,convert,prune}` (see NativeAPIDTO.swift).
