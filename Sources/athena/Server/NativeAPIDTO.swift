import AthenaStructured
import Foundation

// Athena-native `/api/*` JSON (M16). Deliberately NOT Ollama and NOT
// OpenAI: a clean, minimal request/response with no vendor-mimicking
// envelope (`created_at`, `done_reason`, `object`, `choices`, …). The
// `/v1/*` surface remains OpenAI-compatible; `/api/*` is Athena's own
// dialect. Errors reuse the standard `{"error":{message,type,code}}`
// body (`APIErrorBody`) — not dialect-specific.

// Native `/api/chat` inference DTOs (AthenaChatRequest/Response/Chunk/Message
// + AthenaUsage) were removed with the route (ADR 031/013) — `/v1` is the
// single inference surface and they duplicated its shapes.

/// `/api/admin/stop` — model unloaded; daemon keeps running.
struct AthenaStopResponse: Codable {
    let status: String  // "unloaded"
    let model: String
}

// MARK: - /api/models/{load,unload,resident} (M41.1 — explicit per-module
// model lifecycle, generalizing the M39 embedding pattern to every
// module class)

/// One module's selectable-set snapshot (ADR 026): `allowed` is the store's
/// models of the module's modality (classified by ModelSupport), `default` is
/// the configured/best-effort default, and `resident` is the id currently in
/// the slot (nil ⇒ unloaded). The `allowed`/`default` field names are retained
/// for wire compatibility.
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
/// not in the module's store models is a 400 (`model_not_available`).
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

// ADR 026 — the `/api/models/allow*` surface (and its DTOs) are retired.
// Availability is the model store classified by ModelSupport; the per-module
// default is a TOML config key set via `athena default --module M <id>`.

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
    // Typed listing (usability audit §4) — additive, classified by
    // `ModelSupport` (ADR 021) per entry. `modality`/`loadability` always
    // present; `engine`/`draft`/`fused_mtp` omitted when absent (Swift
    // synthesizes `encodeIfPresent` for optionals), so old clients + the
    // OpenAI `/v1/models` surface are unaffected.
    let modality: String?
    let engine: String?
    let loadability: String?
    let draft: Bool?
    let fused_mtp: Bool?
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

// ADR 025 S2 — model lifecycle ops (`/api/models/{pull,convert,prune}`)
// run **synchronously** and stream progress over Server-Sent Events; the
// async queue (and its job ids / persistence) is gone. The frames are built
// by the pinned `ModelOpProgressFrame` (AthenaServerKit) from the MLX-free
// `ModelOpProgress` enum (AthenaCore) — additive over the legacy
// `progress`-only shape, so an older client that only reads `progress`/`done`
// still works (usability audit 2026-07-02 §2/§3). Each SSE `data:` frame is
// one event:
//   - {"event":"progress","fraction":0…1,"bytes":B,"total":T}  (aggregate)
//   - {"event":"file","name","index","count","bytes","total","done"}  (pull:
//       one per shard, throttled ~500ms/1% per file)
//   - {"event":"phase","phase":"download|load|quantize|write"}  (convert)
//   - {"event":"quantize","index":i,"count":N}  (convert materialize loop)
//   - {"event":"done","result":{…op-specific…}}  (terminal success)
//   - {"event":"error","error":{message,type,code}}  (terminal failure)
// followed by the SSE `[DONE]` sentinel. `: keep-alive` comment frames still
// bridge any residual silent tail.

/// SSE `progress` event — the 0…1 download fraction. (Legacy shape; the live
/// encoder is `ModelOpProgressFrame.json`, which also carries `bytes`/`total`
/// and emits the additive `file`/`phase`/`quantize` events. Retained for the
/// documented wire contract.)
struct ModelOpProgressEvent: Codable {
    let event: String  // "progress"
    let fraction: Double
}

/// SSE `error` event — the canonical `{message,type,code}` error detail
/// under an `event` discriminator (the body is the same shape `/v1`+`/api`
/// error responses use, so a consumer parses one envelope everywhere).
struct ModelOpErrorEvent: Codable {
    let event: String  // "error"
    let error: APIErrorBody.ErrorDetail
}

// Terminal `result` payloads (the `result` field of a `done` event).
struct ModelPullResult: Codable {
    let name: String
    let path: String
}
struct ModelConvertResult: Codable {
    let path: String
    let bytes: Int
}
struct ModelPruneResult: Codable {
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

/// `GET /api/logs` — admin-only daemon-log oversight. Entries are the
/// **newest** `limit` of the requested window, in `log show`'s native
/// oldest-first order (the client prints verbatim, no sort). `truncated`
/// is present-and-true only when the window held more than `limit`
/// entries (or the read deadline was hit) and older entries were dropped
/// — a signal to narrow `--since`/raise `--limit` or use `--follow`;
/// omitted when nothing was dropped (backward-compatible). Pull only —
/// the streaming sibling at `/api/logs/stream` is SSE.
struct LogsReportResponse: Codable {
    let logs: [LogEntryDTO]
    let truncated: Bool?
}

// MARK: - ADR 037 — daemon-mediated config + restart

/// `GET /api/config` — the current config projected from the TOML. `values`
/// carries every known scalar (empty string when unset); `readonly_keys` are
/// the deny-listed keys that `PUT /api/config` refuses (edit the TOML + restart
/// with sudo instead).
struct ConfigGetResponse: Codable {
    let path: String
    let keys: [String]
    let values: [String: String]
    let readonly_keys: [String]
}

/// `PUT /api/config` — one scalar set/validated in place.
struct ConfigSetResponse: Codable {
    let ok: Bool
    let key: String
    let value: String
    let note: String
}

/// `POST /api/admin/restart` — acknowledgement before the daemon drains + exits.
struct RestartResponse: Codable {
    let restarting: Bool
    let note: String
}

/// Generic mutation acknowledgements.
struct OkResponse: Codable { let ok: Bool }
struct UserRemovedResponse: Codable {
    let username: String
    let removed: Bool
}
struct TokensRemovedResponse: Codable { let removed: Int }
