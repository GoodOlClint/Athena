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
    /// Per-request generation overrides (M24.3); absent ⇒ loaded defaults.
    let max_tokens: Int?
    let temperature: Double?
    /// Per-request MTP speculative override. Absent ⇒ loaded
    /// `--speculative` default; `true` opts into MTP speculative
    /// (greedy at temp 0, sampling at temp > 0; requires an
    /// MTP-capable model), `false` forces the standard path.
    let speculative: Bool?
    /// Prompt-prefix cache scoping hint (M59.3) — the native dialect's
    /// equivalent of OpenAI `prompt_cache_key`. Absent ⇒ scope by the
    /// authenticated principal alone.
    let prompt_cache_key: String?
}

/// Compact token accounting on the native non-stream reply (M59.3).
/// `cached_tokens` is the prompt-prefix-cache reuse count.
struct AthenaUsage: Codable {
    let prompt_tokens: Int
    let completion_tokens: Int
    let cached_tokens: Int
}

/// Non-streamed `/api/chat` reply. The full generation is in
/// `content`; `done` is always true here (a single object). `usage`
/// (M59.3) carries real token counts incl. `cached_tokens`.
struct AthenaChatResponse: Codable {
    let model: String
    let content: String
    let done: Bool
    let usage: AthenaUsage?
}

/// One NDJSON line of a streamed `/api/chat`: an incremental
/// `content` piece, then a final `{content:"",done:true}` line.
struct AthenaChatChunk: Codable {
    let content: String
    let done: Bool
}

/// `GET /api/cache/prompt` (M59.4) — prompt-prefix KV pool stats.
struct PromptCacheStatsResponse: Codable {
    let enabled: Bool
    let entries: Int
    let bytes: Int
    let hits: Int
    let misses: Int
    let evictions: Int
    let max_entries: Int
    let max_bytes: Int
    static let disabled = PromptCacheStatsResponse(
        enabled: false, entries: 0, bytes: 0, hits: 0, misses: 0,
        evictions: 0, max_entries: 0, max_bytes: 0)
}

/// `DELETE /api/cache/prompt` (M59.4) — flush result.
struct PromptCacheFlushResponse: Codable {
    let flushed: Int
    let entries: Int
    let bytes: Int
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

// MARK: - /api/models/{load,unload,resident} (M41.1 — explicit per-module
// model lifecycle, generalizing the M39 embedding pattern to every
// module class)

/// One module's selectable-set snapshot: the allowlist (operator-
/// declared at `load` time), the default (first declared), and the id
/// currently resident in the slot (nil ⇒ unloaded).
struct ModelSlotDTO: Codable {
    let module: String  // ModuleID.rawValue
    let allowed: [String]
    let `default`: String
    let resident: String?
}

/// `GET /api/models/resident` — every module's slot at once.
struct ModelResidentResponse: Codable {
    let slots: [ModelSlotDTO]
}

/// `POST /api/models/load` — rebind a module's slot to `id` (omit ⇒
/// the default), loading the module first if it is unloaded. An id
/// outside the module's allowlist is a 400 (`model_not_available`).
struct ModelLoadRequest: Codable {
    let module: String  // ModuleID.rawValue
    let id: String?
}

/// `POST /api/models/load` reply once the slot is resident.
struct ModelLoadResponse: Codable {
    let module: String
    let id: String  // id actually loaded
    let status: String  // "loaded"
}

/// `POST /api/models/unload` — release a module's reservation (or all
/// when `module` is absent / `"all"`); the daemon keeps running.
struct ModelUnloadRequest: Codable {
    let module: String?  // ModuleID.rawValue, or absent/"all"
}

struct ModelUnloadResponse: Codable {
    let modules: [String]  // modules actually unloaded
    let status: String  // "unloaded"
}

// MARK: - /api/models/allow (M42 — persistent operator-declared
// allowlist; survives daemon restarts. CLI flags are first-boot seeds
// only; mutations here are the source of truth from M42 onward).

struct AllowlistEntryDTO: Codable {
    let module: String  // ModuleID.rawValue
    let id: String
    let `default`: Bool
    let declared: Double  // epoch seconds
}

struct AllowlistResponse: Codable {
    let allowlist: [AllowlistEntryDTO]
}

struct AddAllowlistRequest: Codable {
    let module: String
    let id: String
    let `default`: Bool?
}

struct AllowlistMutationResponse: Codable {
    let module: String
    let id: String
    let status: String  // "added" | "removed" | "default_set"
}

struct SetAllowlistDefaultRequest: Codable {
    let module: String
    let id: String
}

/// `/api/admin/status` — native daemon + RBAC posture for a remote
/// admin (distinct from open `/healthz` governor state and
/// `/metrics`). M16.5.
struct AdminStatusResponse: Codable {
    let model: String
    let listen: String
    let auth_enabled: Bool
    let users: Int
    let tokens: Int
    let admins: Int
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
    /// Per-token expiry epoch (M36.2), nil ⇒ never expires. The
    /// global token_max_age_days cap is NOT folded in here (it's a
    /// daemon-level policy, reported by `athena doctor`).
    let expires: Double?
}
struct TokenListResponse: Codable { let tokens: [TokenSummaryDTO] }

struct CreateTokenRequest: Codable {
    let user: String
    /// Scope the token to a role subset (default: the user's full
    /// set). Caller must be able to grant each scoped role; an
    /// unscoped token may not exceed the caller's own permissions.
    let role: [String]?
    let label: String?
    /// Per-token lifetime in whole seconds (M36.1). Absent / non-
    /// positive ⇒ never expires (subject to the daemon's global
    /// token_max_age_days cap).
    let ttl_secs: Int?
}
/// The minted key is returned ONCE here and never persisted.
struct CreateTokenResponse: Codable {
    let user: String
    let scope: [String]?
    let token: String
    let hash_prefix: String
}

/// Body for `POST /api/tokens/{prefix}/rotate` (M36.2). Revoke + reissue:
/// the matched token's owner/scope/label carry over; `ttl_secs` sets the
/// NEW token's lifetime (absent ⇒ no expiry, the old TTL is not carried).
struct RotateTokenRequest: Codable {
    let ttl_secs: Int?
}

// MARK: - /api/usage (M27.3 — per-principal token metering, pull-only)

/// One principal's cumulative usage. `updated` is epoch seconds of the
/// last metered request.
struct UsageEntryDTO: Codable {
    let principal: String
    let requests: Int
    let prompt_tokens: Int
    let completion_tokens: Int
    let total_tokens: Int
    let updated: Double
}

/// `GET /api/usage` — own usage for a member; all principals for an
/// admin (owner-scoped like the queue). Always an array so the CLI
/// renders one table shape.
struct UsageReportResponse: Codable { let usage: [UsageEntryDTO] }

// MARK: - /api/audit (M30.2 — append-only RBAC/admin audit trail)

/// One audit entry. `ts` is epoch seconds; `target`/`detail` may be
/// absent for actions with no single subject or context.
struct AuditEntryDTO: Codable {
    let id: Int
    let ts: Double
    let principal: String
    let action: String
    let target: String?
    let result: String
    let detail: String?
}

/// `GET /api/audit` — admin-only oversight view, most-recent-first.
struct AuditReportResponse: Codable { let audit: [AuditEntryDTO] }

// MARK: - /api/logs (M45.5 — daemon's macOS unified-log entries)

/// One log entry, projected from the subset of `/usr/bin/log show
/// --style ndjson` fields Athena cares about. `ts` is ISO 8601 UTC
/// (the unified log's native timestamp). Level is the OSLogType name
/// (`debug` | `info` | `default` | `error` | `fault`); the daemon
/// emits at the swift-log mapping documented in
/// `docs/logging.md`. The full ndjson record carries more fields
/// (pid, tid, process, etc.) — we drop them for compactness.
struct LogEntryDTO: Codable {
    let ts: String
    let level: String
    let category: String
    let message: String
}

/// `GET /api/logs` — admin-only daemon-log oversight, oldest-first
/// (matches `log show` default order; client sorts/reverses as
/// needed). Pull only — the streaming sibling at
/// `/api/logs/stream` is SSE.
struct LogsReportResponse: Codable { let logs: [LogEntryDTO] }

/// Generic mutation acknowledgements.
struct OkResponse: Codable { let ok: Bool }
struct UserRemovedResponse: Codable {
    let username: String
    let removed: Bool
}
struct TokensRemovedResponse: Codable { let removed: Int }
