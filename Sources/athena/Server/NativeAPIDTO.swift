import AthenaStructured
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

// MARK: - /api/models (M16.2 — model-store read + sync ops)

struct ModelEntryDTO: Codable {
    let name: String
    let bytes: Int
    let modified: String  // ISO-8601
}
struct ModelListResponse: Codable { let models: [ModelEntryDTO] }

struct ModelDetailResponse: Codable {
    let name: String
    let path: String
    let bytes: Int
    /// Parsed `config.json` (the model's own config), embedded.
    let config: JSONValue?
}

struct ModelRemovedResponse: Codable {
    let name: String
    let removed: Bool
}

struct ModelCopyRequest: Codable {
    let src: String
    let dst: String
    /// Deep-copy instead of the default symlink alias.
    let copy: Bool?
    let force: Bool?
}
struct ModelCopyResponse: Codable {
    let src: String
    let dst: String
    let path: String
    let aliased: Bool  // false ⇒ deep copy
}

struct DefaultModelResponse: Codable {
    let model: String
    let source: String  // "config" | "builtin"
}
struct SetDefaultModelRequest: Codable { let name: String }

// MARK: - /api/models async ops (M16.3 — queue-dispatched)

struct ModelPullRequest: Codable {
    let id: String
    let revision: String?
}
struct ModelConvertRequest: Codable {
    let id: String
    let revision: String?
    let bits: Int?
    let group_size: Int?
    let name: String?
}
struct ModelPruneRequest: Codable { let dry_run: Bool? }

/// Returned by an async model op (202). Poll `GET /v1/queue/:job_id`
/// for progress/result — owner-scoped, so only the submitter (or an
/// admin) can see it.
struct ModelJobResponse: Codable {
    let job_id: String
    let status: String  // "queued"
}

// Stored job results (surfaced under `result` on /v1/queue/:id).
struct QueuedModelPullResult: Codable {
    let name: String
    let path: String
}
struct QueuedModelConvertResult: Codable {
    let path: String
    let bytes: Int
}
struct QueuedModelPruneResult: Codable {
    let candidates: [String]
    let removed: Int
    let dry_run: Bool
}

// MARK: - /api/users, /api/roles, /api/tokens (M16.4 — RBAC CRUD)

struct UserSummaryDTO: Codable {
    let username: String
    let roles: [String]
}
struct UserListResponse: Codable { let users: [UserSummaryDTO] }

struct CreateUserRequest: Codable {
    let username: String
    let password: String
    /// Initial role (default: member). Caller must be able to grant
    /// it (server-side `RBAC.canGrant`).
    let role: String?
}

struct RoleCatalogEntry: Codable {
    let role: String
    let permissions: [String]
}
struct RolesResponse: Codable { let roles: [RoleCatalogEntry] }

struct TokenSummaryDTO: Codable {
    let username: String
    let scope: [String]?  // nil ⇒ inherits the user's full roles
    let hash_prefix: String
    let label: String?
}
struct TokenListResponse: Codable { let tokens: [TokenSummaryDTO] }

struct CreateTokenRequest: Codable {
    let user: String
    /// Scope the token to a role subset (default: the user's full
    /// set). Caller must be able to grant each scoped role; an
    /// unscoped token may not exceed the caller's own permissions.
    let role: [String]?
    let label: String?
}
/// The minted key is returned ONCE here and never persisted.
struct CreateTokenResponse: Codable {
    let user: String
    let scope: [String]?
    let token: String
    let hash_prefix: String
}

/// Generic mutation acknowledgements.
struct OkResponse: Codable { let ok: Bool }
struct UserRemovedResponse: Codable {
    let username: String
    let removed: Bool
}
struct TokensRemovedResponse: Codable { let removed: Int }
