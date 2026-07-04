import AthenaCore
import AthenaDeploy
import AthenaEmbedding
import AthenaLLM
import AthenaServerKit
import AthenaStore
import AthenaStructured
import AthenaTranscription
import Foundation
import HTTPTypes
import Hummingbird
import MLX
import HummingbirdCore
import HummingbirdTLS
import Logging
import NIOCore
import NIOSSL

// The control plane: OpenAI `/v1/models` discovery plus the native `/api/*`
// surface — model store + lifecycle ops, RBAC administration, usage/audit,
// unified-log oversight, and the prompt-cache operator endpoints. Not
// inference (ADR 013): every route here is daemon control.
extension AthenaServer {
    /// Register the control-plane routes (`/v1/models` + all `/api/*`).
    /// Called from `run()`.
    func registerAdminRoutes(_ router: Router<AppRequestContext>) {
        // OpenAI model discovery (M31.1). Read-only projection of the
        // SAME model store the native `/api/models` serves, in the
        // OpenAI list/retrieve shape so drop-in SDK/LiteLLM clients can
        // probe. Gated `model.read` (AuthPolicy maps `/v1/models*`).
        router.get("/v1/models") { _, _ -> Response in
            handleOpenAIModelsList()
        }
        router.get("/v1/models/:id") { _, context -> Response in
            handleOpenAIModelRetrieve(context.parameters.get("id"))
        }

        router.post("/api/admin/stop") { request, _ -> Response in
            await adminUnloadLLM(request)
        }
        router.get("/api/admin/status") { _, _ -> Response in
            await adminStatus()
        }

        // Per-principal token usage (M27.3). Inference-tier (any
        // authenticated caller sees its OWN usage); an admin sees every
        // principal — owner-scoped like the queue. Pull only — the
        // passive oracle never pushes usage out.
        router.get("/api/usage") { request, _ -> Response in
            await handleUsage(request)
        }

        // Append-only RBAC/admin audit trail (M30.2). Admin-only
        // (AuthPolicy → daemon.admin); filterable, pull only.
        router.get("/api/audit") { request, _ -> Response in
            await handleAudit(request)
        }

        // M45.5 — daemon unified-log oversight. Admin-only; mirrors
        // the audit trail's sensitivity profile (entries carry
        // req=/principal=/function= across all users + reveal internal
        // call sites). `/api/logs` = one-shot historical;
        // `/api/logs/stream` = SSE follow.
        router.get("/api/logs") { request, _ -> Response in
            await handleLogs(request)
        }
        router.get("/api/logs/stream") { request, _ -> Response in
            await handleLogsStream(request)
        }

        // M59.4 — prompt-prefix cache operator surface. Admin-only
        // (AuthPolicy → daemon.admin). GET = pool stats; DELETE = flush the
        // pool (entries not in use), audited at the shared chokepoint.
        router.get("/api/cache/prompt") { _, _ -> Response in
            handlePromptCacheStats()
        }
        router.delete("/api/cache/prompt") { request, _ -> Response in
            await handlePromptCacheFlush(request)
        }

        // Model store (M16.2). Literal sub-paths are registered
        // BEFORE `:name` so Hummingbird's trie (sibling match in
        // registration order) resolves them first; `default`/`copy`
        // are therefore reserved names for show/rm.
        router.get("/api/models") { _, _ -> Response in
            handleModelsList()
        }
        router.get("/api/models/default") { _, _ -> Response in
            handleDefaultModelGet()
        }
        router.put("/api/models/default") { request, _ -> Response in
            await handleDefaultModelSet(request)
        }
        router.post("/api/models/copy") { request, _ -> Response in
            await handleModelCopy(request)
        }
        // Long-running ops (model.write-gated routes). ADR 025 S2: they run
        // SYNCHRONOUSLY and stream SSE progress directly on this route — no
        // job id, no async queue, no persistence. `data: {"event":…}` frames
        // (progress / done / error) end with `[DONE]`.
        router.post("/api/models/pull") { request, _ -> Response in
            await handleModelPull(request)
        }
        router.post("/api/models/convert") { request, _ -> Response in
            await handleModelConvert(request)
        }
        router.post("/api/models/prune") { request, _ -> Response in
            await handleModelPrune(request)
        }
        // Explicit per-module model lifecycle (M41.1): rebind a slot or
        // release it without bouncing the daemon. Generalizes M39's
        // embedding pattern across every module class (llm /
        // textEmbedding / transcription / diarization / speakerEmbedding).
        // Literal sub-paths so they win over the `:name` route below.
        router.get("/api/models/resident") { _, _ -> Response in
            await handleModelsResident()
        }
        router.post("/api/models/load") { request, _ -> Response in
            await handleModelsLoad(request)
        }
        router.post("/api/models/unload") { request, _ -> Response in
            await handleModelsUnload(request)
        }
        // ADR 026 — `/api/models/allow*` retired. Availability = the model
        // store classified by ModelSupport; the per-module default is a TOML
        // config key (`athena default --module M <id>`). The read-only
        // `/api/models/resident` projection + the M41 load/unload endpoints
        // remain, re-pointed at the store.
        router.get("/api/models/:name") { _, context -> Response in
            handleModelShow(context.parameters.get("name"))
        }
        router.delete("/api/models/:name") { request, context
            -> Response in
            await handleModelRemove(
                context.parameters.get("name"), request)
        }

        // RBAC administration over HTTP (M16.4). Every mutation
        // enforces RBAC.canGrant against the CALLER's permission set
        // (NOT implicit-admin like the offline CLI) + last-admin
        // protection. Perm-gated by AuthPolicy
        // (users.read/users.admin/tokens.admin).
        router.get("/api/users") { _, _ -> Response in
            await handleUsersList()
        }
        router.post("/api/users") { request, _ -> Response in
            await handleUserCreate(request)
        }
        router.delete("/api/users/:name") { request, context
            -> Response in
            await handleUserDelete(
                context.parameters.get("name"), request)
        }
        router.post("/api/users/:name/roles/:role") {
            request, context -> Response in
            await handleRoleGrant(
                context.parameters.get("name"),
                context.parameters.get("role"), request)
        }
        router.delete("/api/users/:name/roles/:role") {
            request, context -> Response in
            await handleRoleRevoke(
                context.parameters.get("name"),
                context.parameters.get("role"), request)
        }
        router.get("/api/roles") { _, _ -> Response in
            Self.rolesCatalogResponse()
        }
        router.get("/api/tokens") { _, _ -> Response in
            await handleTokensList()
        }
        router.post("/api/tokens") { request, _ -> Response in
            await handleTokenCreate(request)
        }
        router.delete("/api/tokens/:prefix") { request, context
            -> Response in
            await handleTokenDelete(
                context.parameters.get("prefix"), request)
        }
        router.post("/api/tokens/:prefix/rotate") { request, context
            -> Response in
            await handleTokenRotate(
                context.parameters.get("prefix"), request)
        }
    }


    /// `GET /api/usage` (M27.3). An admin (or the single-tenant
    /// loopback operator when auth is off) sees every principal's
    /// counters; any other authenticated caller sees only its own row.
    private func handleUsage(_ request: Request) async -> Response {
        let who = await bearerPrincipal(request)
        let rows: [UsageRow]
        if !who.enforced || who.isAdmin {
            rows = await store.allUsage()
        } else if let p = who.principal, let r = await store.usage(
            principal: p)
        {
            rows = [r]
        } else {
            rows = []
        }
        return Self.json(
            UsageReportResponse(
                usage: rows.map {
                    UsageEntryDTO(
                        principal: $0.principal, requests: $0.requests,
                        prompt_tokens: $0.promptTokens,
                        completion_tokens: $0.completionTokens,
                        total_tokens: $0.totalTokens,
                        updated: $0.updated)
                }))
    }

    /// `GET /api/audit` (M30.2). Admin-only (AuthPolicy →
    /// `daemon.admin`), so no owner-scoping is needed — the trail is a
    /// privileged oversight view. Filterable by `principal`, `action`,
    /// `since` (epoch seconds) and `limit` (default 100, capped 1000);
    /// rows come back most-recent-first. Pull only — the passive
    /// oracle never pushes the audit trail out.
    private func handleAudit(_ request: Request) async -> Response {
        var principal: String?
        var action: String?
        var since: Double?
        var limit = 100
        if let q = request.uri.query {
            for kv in q.split(separator: "&") {
                let p = kv.split(separator: "=", maxSplits: 1)
                guard let k = p.first.map(String.init) else {
                    continue
                }
                let raw = p.count == 2 ? String(p[1]) : ""
                let v = raw.removingPercentEncoding ?? raw
                switch k {
                case "principal": principal = v.isEmpty ? nil : v
                case "action": action = v.isEmpty ? nil : v
                case "since": since = Double(v)
                case "limit":
                    limit = min(1000, max(1, Int(v) ?? 100))
                default: break
                }
            }
        }
        let rows = await store.listAudit(
            principal: principal, action: action, since: since,
            limit: limit)
        return Self.json(
            AuditReportResponse(
                audit: rows.map {
                    AuditEntryDTO(
                        id: $0.id, ts: $0.ts,
                        principal: $0.principal, action: $0.action,
                        target: $0.target, result: $0.result,
                        detail: $0.detail)
                }))
    }

    /// `GET /api/cache/prompt` (M59.4). Admin-only stats for the
    /// cross-request prompt-prefix KV pool: whether it's enabled, the live
    /// entry/byte occupancy + caps, and cumulative hit/miss/eviction
    /// counters. Read-only ⇒ not audited (mirrors `/api/audit` itself).
    private func handlePromptCacheStats() -> Response {
        guard let prefixCache else {
            return Self.json(PromptCacheStatsResponse.disabled)
        }
        let s = prefixCache.stats()
        return Self.json(
            PromptCacheStatsResponse(
                enabled: true, entries: s.entries, bytes: s.bytes,
                hits: s.hits, misses: s.misses, evictions: s.evictions,
                max_entries: s.maxEntries, max_bytes: s.maxBytes))
    }

    /// `DELETE /api/cache/prompt` (M59.4). Admin-only flush of the pool —
    /// drops every entry NOT currently held by an in-flight generation
    /// (those are freed when their request releases them). Audited at the
    /// shared chokepoint (M30). Returns how many entries were freed plus the
    /// post-flush occupancy.
    private func handlePromptCacheFlush(_ request: Request) async -> Response {
        guard let prefixCache else {
            await audit(
                request, action: "prompt_cache.flush", target: nil,
                result: "ok", detail: "disabled")
            return Self.json(PromptCacheFlushResponse(flushed: 0, entries: 0, bytes: 0))
        }
        // ADR 029 (2026-07-02 audit residual) — with the disk tier on,
        // `flushIdle` demotes victims via MLX `eval`/encode, so this operator
        // flush must not run concurrently with a gated decode on the one
        // Metal pool. Same wrap as the governor relief hook (WP1).
        let freed: Int
        do {
            freed = try await InferenceGate.shared.withExclusiveExecution {
                prefixCache.flushIdle()
            }
        } catch {
            await audit(
                request, action: "prompt_cache.flush", target: nil,
                result: "error", detail: "\(type(of: error))")
            return Self.classified(error, module: .llm)
        }
        let s = prefixCache.stats()
        await audit(
            request, action: "prompt_cache.flush", target: nil,
            result: "ok", detail: "freed=\(freed) remaining=\(s.entries)")
        return Self.json(
            PromptCacheFlushResponse(
                flushed: freed, entries: s.entries, bytes: s.bytes))
    }

    /// `GET /api/logs` (M45.5). Admin-only daemon-log oversight,
    /// projected from `/usr/bin/log show --style ndjson` filtered to
    /// `subsystem == "athena"`. Query params:
    ///   - `since=<dur>` (default 1h; passed to `log show --last`)
    ///   - `category=<X>` (repeatable; AND'd into the predicate)
    ///   - `debug=1` (include info+debug entries; default notice+)
    ///   - `limit=<n>` (default 200, capped 5000)
    /// Pull-only JSON; the SSE sibling at `/api/logs/stream` follows
    /// new entries live.
    private func handleLogs(_ request: Request) async -> Response {
        var since = "1h"
        var categories: [String] = []
        var debug = false
        var limit = 200
        if let q = request.uri.query {
            for kv in q.split(separator: "&") {
                let p = kv.split(separator: "=", maxSplits: 1)
                guard let k = p.first.map(String.init) else { continue }
                let raw = p.count == 2 ? String(p[1]) : ""
                let v = raw.removingPercentEncoding ?? raw
                switch k {
                case "since": if !v.isEmpty { since = v }
                case "category": if !v.isEmpty { categories.append(v) }
                case "debug": debug = (v == "1" || v == "true")
                case "limit":
                    limit = min(5000, max(1, Int(v) ?? 200))
                default: break
                }
            }
        }
        // Build predicate the same way `athena logs` does — pre-baked
        // subsystem, optional categories.
        guard let predicate = Self.logPredicate(categories: categories)
        else {
            return Self.error(
                status: .badRequest,
                message: "invalid category (must be alphanum + dot)",
                type: "invalid_request_error",
                code: "invalid_category")
        }
        var args = [
            "show", "--style", "ndjson", "--last", since,
            "--predicate", predicate,
        ]
        if debug { args += ["--info", "--debug"] }
        let collected = (try? await Self.collectLogEntries(
            args: args, limit: limit)) ?? (entries: [], truncated: false)
        return Self.json(LogsReportResponse(
            logs: collected.entries,
            truncated: collected.truncated ? true : nil))
    }

    /// `GET /api/logs/stream` (M45.5). Admin-only SSE following
    /// `log stream --style ndjson` filtered to subsystem athena.
    /// Each event line is `data: {<LogEntryDTO JSON>}\n\n`. Capped at
    /// ~10 min so a dropped client doesn't pin a process forever; the
    /// subprocess is terminated on `continuation.onTermination`.
    private func handleLogsStream(_ request: Request) async -> Response {
        var categories: [String] = []
        var debug = false
        if let q = request.uri.query {
            for kv in q.split(separator: "&") {
                let p = kv.split(separator: "=", maxSplits: 1)
                guard let k = p.first.map(String.init) else { continue }
                let raw = p.count == 2 ? String(p[1]) : ""
                let v = raw.removingPercentEncoding ?? raw
                switch k {
                case "category": if !v.isEmpty { categories.append(v) }
                case "debug": debug = (v == "1" || v == "true")
                default: break
                }
            }
        }
        guard let predicate = Self.logPredicate(categories: categories)
        else {
            return Self.error(
                status: .badRequest,
                message: "invalid category (must be alphanum + dot)",
                type: "invalid_request_error",
                code: "invalid_category")
        }
        var args = [
            "stream", "--style", "ndjson", "--predicate", predicate,
        ]
        if debug { args += ["--info", "--debug"] }
        let stream = AsyncStream<ByteBuffer> { continuation in
            let proc = Process()
            proc.executableURL = URL(
                fileURLWithPath: "/usr/bin/log")
            proc.arguments = args
            let pipe = Pipe()
            proc.standardOutput = pipe
            proc.standardError = FileHandle.nullDevice
            let task = Task.detached {
                let h = pipe.fileHandleForReading
                var buffer = Data()
                let deadline = Date().addingTimeInterval(600)
                while Date() < deadline {
                    if Task.isCancelled { break }
                    let chunk = h.availableData
                    if chunk.isEmpty {
                        // Process exited — bail.
                        break
                    }
                    buffer.append(chunk)
                    while let nl = buffer.firstIndex(of: 0x0A) {
                        let line = buffer.subdata(
                            in: buffer.startIndex..<nl)
                        buffer = buffer.subdata(
                            in: (nl + 1)..<buffer.endIndex)
                        if let entry = Self.parseNDJSONLogLine(line),
                            let json = try? JSONEncoder().encode(entry)
                        {
                            var b = ByteBuffer()
                            b.writeString("data: ")
                            b.writeBytes(json)
                            b.writeString("\n\n")
                            continuation.yield(b)
                        }
                    }
                }
                if proc.isRunning { proc.terminate() }
                continuation.finish()
            }
            do { try proc.run() } catch {
                task.cancel()
                continuation.finish()
                return
            }
            continuation.onTermination = { _ in
                task.cancel()
                if proc.isRunning { proc.terminate() }
            }
        }
        var headers = HTTPFields()
        headers[.contentType] = "text/event-stream"
        headers[.cacheControl] = "no-cache"
        return Response(
            status: .ok, headers: headers,
            body: ResponseBody(asyncSequence: stream))
    }

    /// Build the `log` predicate: subsystem pin + optional category
    /// AND-filter. Returns nil if any category contains characters
    /// outside `[A-Za-z0-9._-]` (defensive — predicate-injection
    /// guard since the value is interpolated into a quoted string).
    private static func logPredicate(categories: [String]) -> String? {
        let safe = CharacterSet(
            charactersIn:
                "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
        )
        for c in categories {
            guard !c.isEmpty,
                c.unicodeScalars.allSatisfy({ safe.contains($0) })
            else { return nil }
        }
        var p = "subsystem == \"athena\""
        if !categories.isEmpty {
            let list = categories
                .map { "\"\($0)\"" }
                .joined(separator: ", ")
            p += " AND category IN { \(list) }"
        }
        return p
    }

    /// Run `/usr/bin/log` with the supplied args, parse each ndjson
    /// line into a LogEntryDTO, and return the **newest** `limit`
    /// entries (usability audit 2026-07-02 §1). `log show` emits
    /// oldest-first, so we drain the whole window into a fixed-capacity
    /// `LogTail` ring buffer and keep the tail — the old code stopped at
    /// the *oldest* `limit` and dropped everything after, leaving the
    /// operator ~30 min stale. `truncated` is true when the window held
    /// more than `limit` entries. The process is killed on a ~30s
    /// deadline so a hung `log` invocation can't pin a request thread.
    private static func collectLogEntries(
        args: [String], limit: Int
    ) async throws -> (entries: [LogEntryDTO], truncated: Bool) {
        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: "/usr/bin/log")
        proc.arguments = args
        let pipe = Pipe()
        proc.standardOutput = pipe
        proc.standardError = FileHandle.nullDevice
        try proc.run()
        let deadline = Date().addingTimeInterval(30)
        // Drain the WHOLE window (not just the first `limit` lines);
        // LogTail retains the newest `limit`. Deadline-bounded so a
        // stuck `log` can't hang the request — a hit there sets
        // `truncated` too (we couldn't prove we read the tail).
        var buffer = Data()
        var tail = LogTail<LogEntryDTO>(limit: limit)
        var deadlineHit = false
        let h = pipe.fileHandleForReading
        while true {
            if Date() >= deadline {
                deadlineHit = true
                break
            }
            let chunk = h.availableData
            if chunk.isEmpty {
                if !proc.isRunning { break }
                try await Task.sleep(nanoseconds: 50_000_000)
                continue
            }
            buffer.append(chunk)
            while let nl = buffer.firstIndex(of: 0x0A) {
                let line = buffer.subdata(
                    in: buffer.startIndex..<nl)
                buffer = buffer.subdata(
                    in: (nl + 1)..<buffer.endIndex)
                if let entry = parseNDJSONLogLine(line) {
                    tail.append(entry)
                }
            }
        }
        if proc.isRunning { proc.terminate() }
        return (tail.entries, tail.truncated || deadlineHit)
    }

    /// Decode one `log show --style ndjson` line into our compact
    /// LogEntryDTO. Returns nil for header rows / parse failures —
    /// best-effort projection, not strict decoding.
    private static func parseNDJSONLogLine(_ data: Data) -> LogEntryDTO?
    {
        guard !data.isEmpty,
            let obj = try? JSONSerialization.jsonObject(with: data)
                as? [String: Any]
        else { return nil }
        // `log show` emits a header row first; skip if there's no
        // eventMessage.
        guard let msg = obj["eventMessage"] as? String else {
            return nil
        }
        let ts = obj["timestamp"] as? String ?? ""
        let levelRaw = obj["messageType"] as? String ?? "default"
        let category = obj["category"] as? String ?? "daemon"
        return LogEntryDTO(
            ts: ts, level: levelRaw, category: category, message: msg)
    }

    // MARK: - Model lifecycle ops (ADR 025 S2 — synchronous + SSE)

    /// Per-key emit-throttle state for the model-op SSE stream. Holds the
    /// `(lastMs, lastFraction)` per key and defers the decision to the pinned
    /// `ModelOpProgressFrame.shouldEmit`. The progress closure may fire from the
    /// main actor (download) then a background task (quantize) but never
    /// concurrently; the lock is belt-and-suspenders.
    final class ModelOpThrottleState: @unchecked Sendable {
        private let lock = NSLock()
        private var last: [String: (ms: Int, frac: Double)] = [:]
        func shouldEmit(key: String, fraction: Double, done: Bool) -> Bool {
            lock.lock()
            defer { lock.unlock() }
            let now = Int(DispatchTime.now().uptimeNanoseconds / 1_000_000)
            let prev = last[key]
            let ok = ModelOpProgressFrame.shouldEmit(
                nowMs: now, lastMs: prev?.ms, fraction: fraction,
                lastFraction: prev?.frac, done: done)
            if ok { last[key] = (now, fraction) }
            return ok
        }
    }

    /// One SSE `data:` frame around an encoded JSON payload.
    private static func sseFrame(_ bytes: Data) -> ByteBuffer {
        var b = ByteBuffer()
        b.writeString("data: ")
        b.writeBytes(bytes)
        b.writeString("\n\n")
        return b
    }

    /// Run a model store op to completion. `progress` (0…1) reports the
    /// download fraction where the op has one (pull/convert); prune has
    /// none. Returns the op-specific result JSON, or an error envelope
    /// detail (`{message,type,code}`). ADR 025 S2: model ops are
    /// synchronous now — no async queue, no job id, no persistence. Shared
    /// by the SSE CLI handlers and the blocking `/ui` console.
    func performModelOp(
        kind: String, body: Data,
        progress: @escaping @Sendable (ModelOpProgress) -> Void = { _ in }
    ) async -> (result: Data?, error: APIErrorBody.ErrorDetail?) {
        switch kind {
        case "model_pull":
            // Egress (proxy + HF token) was exported process-wide at daemon
            // startup (Load); the #hubDownloader reads it. Only sanctioned
            // model-fetch egress — passive oracle intact.
            guard
                let req = try? JSONDecoder().decode(
                    ModelPullRequest.self, from: body), !req.id.isEmpty
            else {
                return (
                    nil,
                    .init(
                        message: "model_pull requires non-empty 'id'",
                        type: "invalid_request_error", code: "invalid_body"))
            }
            do {
                let dest = try await ModelPull.pull(
                    id: req.id, revision: req.revision,
                    into: modelStoreRoot, progress: progress)
                return (
                    try? JSONEncoder().encode(
                        ModelPullResult(
                            name: dest.lastPathComponent, path: dest.path)),
                    nil)
            } catch {
                return (
                    nil,
                    .init(
                        message: "pull failed: \(ModelPull.friendlyError(error))",
                        type: "server_error", code: "pull_failed"))
            }
        case "model_convert":
            guard
                let req = try? JSONDecoder().decode(
                    ModelConvertRequest.self, from: body), !req.id.isEmpty
            else {
                return (
                    nil,
                    .init(
                        message: "model_convert requires non-empty 'id'",
                        type: "invalid_request_error", code: "invalid_body"))
            }
            do {
                // M-conv: `bits` is opt-in (mlx_lm-style). Omit ⇒ no
                // quantization; explicit N ⇒ quantize to N-bit.
                // ADR 029 WP3 — convert loads weights + runs quantize kernels on
                // the shared Metal pool; gate it so it can't co-execute with a
                // decode (the "two uncoordinated allocators on one pool" hazard
                // ADR 011/029 forbid). A convert holds the gate for the whole
                // quantize (minutes) — deliberate: an operator convert blocks
                // inference for its duration rather than racing it on the device.
                let r = try await InferenceGate.shared.withExclusiveExecution {
                    try await ModelConvert.convert(
                        id: req.id, revision: req.revision, bits: req.bits,
                        groupSize: req.group_size ?? 64,
                        into: modelStoreRoot, name: req.name,
                        progress: progress)
                }
                return (
                    try? JSONEncoder().encode(
                        ModelConvertResult(path: r.path.path, bytes: r.bytes)),
                    nil)
            } catch let e as AthenaError {
                // Cause-naming convert errors (ADR 016 redirect / unsupported
                // class) carry an actionable message + stable code — surface
                // them verbatim instead of a raw substrate dump.
                return (
                    nil,
                    .init(message: e.message, type: e.type, code: e.code))
            } catch {
                return (
                    nil,
                    .init(
                        message: "convert failed: \(error)",
                        type: "server_error", code: "convert_failed"))
            }
        case "model_prune":
            let req =
                (try? JSONDecoder().decode(
                    ModelPruneRequest.self, from: body))
                ?? ModelPruneRequest(dry_run: false)
            do {
                let pr = try ModelStoreOps.prune(
                    root: modelStoreRoot, dryRun: req.dry_run ?? false)
                return (
                    try? JSONEncoder().encode(
                        ModelPruneResult(
                            candidates: pr.victims.map { $0.name },
                            removed: pr.removed, dry_run: pr.dryRun)),
                    nil)
            } catch {
                return (
                    nil,
                    .init(
                        message: "prune failed: \(error)",
                        type: "server_error", code: "prune_failed"))
            }
        default:
            return (
                nil,
                .init(
                    message: "unknown model op '\(kind)'",
                    type: "invalid_request_error", code: "invalid_kind"))
        }
    }

    /// Stream a model op's progress over SSE on the current request (ADR
    /// 025 S2). Emits `data: {"event":"progress","fraction":F}` frames
    /// during the download, `: keep-alive` comments during silent tails
    /// (e.g. the convert quantization phase, which has no HF progress), and
    /// a terminal `data: {"event":"done","result":…}` or
    /// `data: {"event":"error","error":…}` frame before `[DONE]`.
    private func streamModelOp(kind: String, body: Data) -> Response {
        let stream = AsyncStream<ByteBuffer> { continuation in
            // Keep-alive ticker so a long silent op doesn't look hung to a
            // client/proxy idle timeout.
            let keepAlive = Task {
                while !Task.isCancelled {
                    try? await Task.sleep(nanoseconds: 15_000_000_000)
                    if Task.isCancelled { return }
                    var b = ByteBuffer()
                    b.writeString(": keep-alive\n\n")
                    continuation.yield(b)
                }
            }
            let work = Task {
                // Per-key emit throttle (audit §2/§3): ~500ms or 1% per file /
                // per aggregate / per quantize; phase frames always pass. The
                // decision is the pinned ModelOpProgressFrame.shouldEmit.
                let throttle = ModelOpThrottleState()
                let progress: @Sendable (ModelOpProgress) -> Void = { p in
                    let key: String
                    let frac: Double
                    let done: Bool
                    switch p {
                    case let .download(f, _, _): (key, frac, done) = ("agg", f, false)
                    case let .file(name, _, _, b, t, d):
                        (key, frac, done) = (
                            "file:\(name)", t > 0 ? Double(b) / Double(t) : 0, d)
                    case .phase: (key, frac, done) = ("phase", -1, true)
                    case let .quantize(i, n):
                        (key, frac, done) = (
                            "quant", n > 0 ? Double(i) / Double(n) : 0, i >= n)
                    }
                    guard throttle.shouldEmit(key: key, fraction: frac, done: done)
                    else { return }
                    continuation.yield(
                        Self.sseFrame(ModelOpProgressFrame.json(p)))
                }
                let (result, error) = await performModelOp(
                    kind: kind, body: body, progress: progress)
                keepAlive.cancel()
                if let result {
                    var b = ByteBuffer()
                    b.writeString("data: {\"event\":\"done\",\"result\":")
                    b.writeBytes(result)
                    b.writeString("}\n\n")
                    continuation.yield(b)
                } else {
                    let detail =
                        error
                        ?? .init(
                            message: "model op failed",
                            type: "server_error", code: "model_op_failed")
                    if let d = try? JSONEncoder().encode(
                        ModelOpErrorEvent(event: "error", error: detail))
                    {
                        continuation.yield(Self.sseFrame(d))
                    }
                }
                var done = ByteBuffer()
                done.writeString("data: [DONE]\n\n")
                continuation.yield(done)
                continuation.finish()
            }
            continuation.onTermination = { _ in
                work.cancel()
                keepAlive.cancel()
            }
        }
        var headers = HTTPFields()
        headers[.contentType] = "text/event-stream"
        headers[.cacheControl] = "no-cache"
        return Response(
            status: .ok, headers: headers,
            body: ResponseBody(asyncSequence: stream))
    }

    /// Collect + validate a model-op body, then stream it. A bad body is a
    /// plain `400` (the stream never opens); a healthy submit returns the
    /// SSE stream (200). The `.modelWrite` gate is applied by the auth
    /// middleware (route → permission); audit it here at initiation.
    private func handleModelPull(_ request: Request) async -> Response {
        await beginModelOp(
            request, kind: "model_pull", action: "model.pull"
        ) { body in
            guard
                let r = try? JSONDecoder().decode(
                    ModelPullRequest.self, from: body), !r.id.isEmpty
            else { return ("model_pull requires non-empty 'id'", nil) }
            return (nil, r.id)
        }
    }

    private func handleModelConvert(_ request: Request) async -> Response {
        await beginModelOp(
            request, kind: "model_convert", action: "model.convert"
        ) { body in
            guard
                let r = try? JSONDecoder().decode(
                    ModelConvertRequest.self, from: body), !r.id.isEmpty
            else { return ("model_convert requires non-empty 'id'", nil) }
            return (nil, r.id)
        }
    }

    private func handleModelPrune(_ request: Request) async -> Response {
        await beginModelOp(
            request, kind: "model_prune", action: "model.prune"
        ) { _ in (nil, nil) }
    }

    /// Shared model-op entry: collect the body (≤ 1 MiB), run `validate`
    /// (returns an error message for a 400, or the audit target id), record
    /// the M30 audit row, and hand off to the SSE stream.
    private func beginModelOp(
        _ request: Request, kind: String, action: String,
        validate: (Data) -> (error: String?, target: String?)
    ) async -> Response {
        let body: Data
        do {
            let buf = try await request.body.collect(upTo: 1 * 1024 * 1024)
            body = Data(buffer: buf)
        } catch {
            return Self.error(
                status: .badRequest,
                message: "Invalid request body: \(error)",
                type: "invalid_request_error", code: "invalid_body")
        }
        let (problem, target) = validate(body)
        if let problem {
            return Self.error(
                status: .badRequest, message: problem,
                type: "invalid_request_error", code: "invalid_body")
        }
        await audit(
            request, action: action, target: target, result: "ok")
        // Operator legibility: a notice-level line per model op (replacing the
        // old `queue submit/running` lines). Emitted inside the request's
        // LogScope so it carries req=/principal=/function= (M45.3).
        Self.log.notice(
            "model op \(action) target=\(target ?? "-")")
        return streamModelOp(kind: kind, body: body)
    }
    private static func iso(_ d: Date) -> String {
        isoFormatter.string(from: d)
    }

    /// Conservative model-name guard for NETWORK input: a bare store
    /// child over `[A-Za-z0-9._-]`. Blocks path traversal and — for
    /// the `default` setter — TOML injection via quotes/newlines
    /// (`setScalarThrowing` does not escape string values).
    private static func safeModelName(_ s: String?) -> String? {
        guard let s, !s.isEmpty, s.count <= 128 else { return nil }
        let ok = s.allSatisfy {
            $0.isLetter || $0.isNumber || $0 == "." || $0 == "_"
                || $0 == "-"
        }
        guard ok, s != "..", s != "." else { return nil }
        return s
    }

    func handleModelsList() -> Response {
        Self.json(
            ModelListResponse(
                models: ModelStoreOps.list(root: modelStoreRoot)
                    .map {
                        ModelEntryDTO(
                            name: $0.name, bytes: $0.bytes,
                            modified: Self.iso($0.modified),
                            modality: $0.modality, engine: $0.engine,
                            loadability: $0.loadability,
                            draft: $0.draft ? true : nil,
                            fused_mtp: $0.fusedMTP ? true : nil)
                    }))
    }

    /// `GET /v1/models` (M31.1) — OpenAI list shape over the model store.
    private func handleOpenAIModelsList() -> Response {
        Self.json(
            OpenAIModelList(
                object: "list",
                data: ModelStoreOps.list(root: modelStoreRoot)
                    .map { Self.openAIModel($0.name, $0.modified) }))
    }

    /// `GET /v1/models/:id` (M31.1) — OpenAI retrieve shape; 404 in the
    /// OpenAI error envelope when the model is not in the store.
    private func handleOpenAIModelRetrieve(_ id: String?) -> Response {
        guard let name = Self.safeModelName(id) else {
            return Self.error(
                status: .badRequest, message: "invalid model name",
                type: "invalid_request_error", code: "invalid_name")
        }
        guard
            let d = ModelStoreOps.show(root: modelStoreRoot, name: name)
        else {
            return Self.error(
                status: .notFound, message: "no model '\(name)'",
                type: "invalid_request_error", code: "not_found")
        }
        // A16 — `show` now carries this entry's own mtime, so a single-model
        // retrieve no longer stat-walks the entire store via `list` just to
        // find one `created` time.
        return Self.json(Self.openAIModel(d.name, d.modified))
    }

    private static func openAIModel(_ name: String, _ modified: Date)
        -> OpenAIModel
    {
        OpenAIModel(
            id: name, object: "model",
            created: Int(modified.timeIntervalSince1970),
            owned_by: "athena")
    }

    func handleModelShow(_ name: String?) -> Response {
        guard let name = Self.safeModelName(name) else {
            return Self.error(
                status: .badRequest, message: "invalid model name",
                type: "invalid_request_error", code: "invalid_name")
        }
        guard
            let d = ModelStoreOps.show(
                root: modelStoreRoot, name: name)
        else {
            return Self.error(
                status: .notFound, message: "no model '\(name)'",
                type: "invalid_request_error", code: "not_found")
        }
        return Self.json(
            ModelDetailResponse(
                name: d.name, path: d.path, bytes: d.bytes,
                config: try? JSONDecoder().decode(
                    JSONValue.self, from: d.configJSON)))
    }

    func handleModelRemove(
        _ name: String?, _ request: Request
    ) async -> Response {
        guard let name = Self.safeModelName(name) else {
            return Self.error(
                status: .badRequest, message: "invalid model name",
                type: "invalid_request_error", code: "invalid_name")
        }
        do {
            try ModelStoreOps.remove(
                root: modelStoreRoot, name: name)
        } catch ModelStoreOps.OpError.notFound {
            return Self.error(
                status: .notFound, message: "no model '\(name)'",
                type: "invalid_request_error", code: "not_found")
        } catch {
            return Self.error(
                status: .internalServerError, message: "\(error)",
                type: "server_error", code: "remove_failed")
        }
        await audit(
            request, action: "model.remove", target: name,
            result: "ok")
        return Self.json(
            ModelRemovedResponse(name: name, removed: true))
    }

    func handleModelCopy(_ request: Request) async -> Response {
        let decoded = await decodeJSON(request, ModelCopyRequest.self)
        guard case .ok(let body) = decoded else {
            return decoded.orFail
        }
        // Network copies are store-name → store-name only (no
        // absolute path / traversal source).
        guard
            let src = Self.safeModelName(body.src),
            let dst = Self.safeModelName(body.dst)
        else {
            return Self.error(
                status: .badRequest,
                message: "src and dst must be bare model names",
                type: "invalid_request_error", code: "invalid_name")
        }
        let deep = body.copy ?? false
        do {
            let dest = try ModelStoreOps.copy(
                root: modelStoreRoot, src: src, dst: dst,
                deepCopy: deep, force: body.force ?? false)
            return Self.json(
                ModelCopyResponse(
                    src: src, dst: dst, path: dest.path,
                    aliased: !deep))
        } catch ModelStoreOps.OpError.notFound {
            return Self.error(
                status: .notFound, message: "no model '\(src)'",
                type: "invalid_request_error", code: "not_found")
        } catch ModelStoreOps.OpError.exists {
            return Self.error(
                status: .conflict, message: "'\(dst)' exists",
                type: "invalid_request_error", code: "exists")
        } catch {
            return Self.error(
                status: .internalServerError, message: "\(error)",
                type: "server_error", code: "copy_failed")
        }
    }

    func handleDefaultModelGet() -> Response {
        let url = ConfigEditor.resolvePath(nil)
        if let cfg = try? AthenaConfig.parse(file: url),
            let model = cfg.model, !model.isEmpty
        {
            return Self.json(
                DefaultModelResponse(model: model, source: "config"))
        }
        return Self.json(
            DefaultModelResponse(
                model: ModelStore.defaultModelName,
                source: "builtin"))
    }

    func handleDefaultModelSet(_ request: Request) async
        -> Response
    {
        let decoded = await decodeJSON(
            request, SetDefaultModelRequest.self)
        guard case .ok(let body) = decoded else {
            return decoded.orFail
        }
        guard let name = Self.safeModelName(body.name) else {
            return Self.error(
                status: .badRequest,
                message:
                    "name must be a bare model name [A-Za-z0-9._-]",
                type: "invalid_request_error", code: "invalid_name")
        }
        let url = ConfigEditor.resolvePath(nil)
        do {
            try ConfigEditor.setScalarThrowing(
                key: "model", value: name, in: url)
        } catch {
            return Self.error(
                status: .badRequest, message: "\(error)",
                type: "invalid_request_error", code: "config_error")
        }
        await audit(
            request, action: "model.default_set", target: name,
            result: "ok")
        return Self.json(
            DefaultModelResponse(model: name, source: "config"))
    }

    private func handleModelsResident() async -> Response {
        var slots: [ModelSlotDTO] = []
        for id in ModuleID.allCases {
            let sel = selectable(id)
            let allowed = await sel.allowedModelIds()
            let def = await sel.defaultModelId()
            let resident = await sel.residentModelId()
            slots.append(
                ModelSlotDTO(
                    module: id.rawValue, allowed: allowed,
                    default: def, resident: resident))
        }
        return Self.json(ModelResidentResponse(slots: slots))
    }

    /// `POST /api/models/load` (M41.1) — rebind `body.module`'s slot to
    /// `body.id` (omit ⇒ the module's default). The slot is governor-
    /// loaded first (so the rebind has its fixed reservation), then
    /// the module rebinds in place. An id outside the module's
    /// allowlist surfaces as a classified 400 (`model_not_available`);
    /// a substrate load failure becomes a classified
    /// 500/503 (Metal OOM). Gated `model.write` by AuthPolicy. Audited
    /// (M30): caller + module + id + result.
    private func handleModelsLoad(_ request: Request) async -> Response {
        let decoded = await decodeJSON(request, ModelLoadRequest.self)
        guard case .ok(let body) = decoded else {
            return decoded.orFail
        }
        guard
            let moduleId = ModuleID(rawValue: body.module)
        else {
            await audit(
                request, action: "model.load", target: body.module,
                result: "denied", detail: "unknown module")
            return Self.error(
                status: .badRequest,
                message: "unknown module '\(body.module)'",
                type: "invalid_request_error", code: "invalid_module")
        }
        let sel = selectable(moduleId)
        let allowed = await sel.allowedModelIds()
        let def = await sel.defaultModelId()
        let requested = body.id ?? def
        // NE5 — store-identity lookup (bare name OR full HF id), uniform
        // with the LLM rebind/select path and the embedding/audio modules;
        // the canonical stored id drives the load.
        guard let target =
            allowed.canonicalByStoreIdentity(requested)
        else {
            await audit(
                request, action: "model.load",
                target: "\(moduleId.rawValue):\(requested)",
                result: "denied", detail: "id outside allowlist")
            return Self.error(
                status: .badRequest,
                message:
                    "Model '\(requested)' is not available. Configured "
                    + "models for \(moduleId.rawValue): "
                    + "\(allowed.dedupedCaseInsensitive().joined(separator: ", ")).",
                type: "invalid_request_error",
                code: "model_not_available")
        }
        do {
            try await governor.ensureLoaded(moduleId)
            // ADR 029 WP3 — gate the rebind exactly as the request-path
            // `auditedRebind` (:3761) does: a warm swap drops+loads weights on
            // the Metal pool and must not run while a decode holds the slot
            // (transient double-residency → OOM) or another tenant executes.
            // This control-plane path previously rebound off-gate (the H3
            // hazard reachable via /api/models/load).
            try await InferenceGate.shared.withExclusiveExecution {
                try await sel.rebind(to: target)
            }
        } catch let e as AthenaError {
            await audit(
                request, action: "model.load",
                target: "\(moduleId.rawValue):\(target)",
                result: "denied", detail: e.code)
            return Self.error(
                status: HTTPResponse.Status(code: e.httpStatus),
                message: e.message, type: "server_error", code: e.code)
        } catch {
            await audit(
                request, action: "model.load",
                target: "\(moduleId.rawValue):\(target)",
                result: "denied", detail: String(describing: error))
            return Self.classified(error, module: moduleId)
        }
        await audit(
            request, action: "model.load",
            target: "\(moduleId.rawValue):\(target)", result: "ok")
        return Self.json(
            ModelLoadResponse(
                module: moduleId.rawValue, id: target,
                status: "loaded"))
    }

    /// `POST /api/models/unload` (M41.1) — release a single module's
    /// slot, or all of them when `module` is absent / `"all"`. Daemon
    /// keeps running; the next inference reloads the module's default
    /// lazily. Gated `model.write` by AuthPolicy. Audited (M30).
    private func handleModelsUnload(_ request: Request) async
        -> Response
    {
        let decoded = await decodeJSON(request, ModelUnloadRequest.self)
        guard case .ok(let body) = decoded else {
            return decoded.orFail
        }
        let targets: [ModuleID]
        if let raw = body.module, raw != "all", !raw.isEmpty {
            guard let m = ModuleID(rawValue: raw) else {
                await audit(
                    request, action: "model.unload", target: raw,
                    result: "denied", detail: "unknown module")
                return Self.error(
                    status: .badRequest,
                    message: "unknown module '\(raw)'",
                    type: "invalid_request_error",
                    code: "invalid_module")
            }
            targets = [m]
        } else {
            targets = ModuleID.allCases
        }
        var unloaded: [String] = []
        for m in targets {
            await governor.unload(m)
            unloaded.append(m.rawValue)
        }
        await audit(
            request, action: "model.unload",
            target: unloaded.joined(separator: ","),
            result: "ok")
        return Self.json(
            ModelUnloadResponse(
                modules: unloaded, status: "unloaded"))
    }

    // MARK: - Daemon control ops (M16 admin + M18.3 reuse)

    /// Unload the LLM module (frees its memory). Shared by
    /// `POST /api/admin/stop` (bearer) and the M18.3 `/ui/api/admin/
    /// stop` (cookie) — one implementation, no duplication.
    func adminUnloadLLM(_ request: Request) async -> Response {
        await governor.unload(.llm)
        await audit(
            request, action: "daemon.unload", target: modelName,
            result: "ok")
        return Self.json(
            AthenaStopResponse(status: "unloaded", model: modelName))
    }

    /// Pre-warm the LLM module so the next inference is hot. New
    /// daemon-control verb (M18.3); exposed only on the cookie+CSRF
    /// /ui surface (the public /api admin surface is frozen at M16).
    func adminLoadLLM(_ request: Request) async -> Response {
        do {
            try await governor.ensureLoaded(.llm)
            await audit(
                request, action: "daemon.load", target: modelName,
                result: "ok")
            return Self.json(
                AthenaStopResponse(
                    status: "loaded", model: modelName))
        } catch {
            return Self.classified(error, module: .llm)
        }
    }

    /// Daemon posture snapshot. Shared by `GET /api/admin/status`
    /// (bearer) and `/ui/api/admin/status` (cookie).
    func adminStatus() async -> Response {
        Self.json(
            AdminStatusResponse(
                model: modelName,
                listen:
                    "\(config.listenHost):\(config.listenPort)",
                auth_enabled: auth.isEnabled,
                users: await store.userCount(),
                tokens: await store.tokenCount(),
                admins:
                    (try? await store.usersWithRole("admin"))?.count ?? 0))
    }

    /// Bare identifier guard for network input (username/role names):
    /// `[A-Za-z0-9._-]`, ≤64, not `.`/`..`.
    private static func safeIdent(_ s: String?) -> String? {
        guard let s, !s.isEmpty, s.count <= 64 else { return nil }
        let ok = s.allSatisfy {
            $0.isLetter || $0.isNumber || $0 == "." || $0 == "_"
                || $0 == "-"
        }
        guard ok, s != "..", s != "." else { return nil }
        return s
    }

    /// Caller may assign every role in `roles` only if it holds every
    /// permission each confers (fail-closed on an unknown role).
    private static func canGrantAll(
        _ roles: some Sequence<String>, _ caller: Set<Permission>
    ) -> Bool {
        roles.allSatisfy {
            RBAC.canGrant(role: $0, grantorPermissions: caller)
        }
    }

    func handleUsersList() async -> Response {
        var out: [UserSummaryDTO] = []
        for u in await store.listUsers() {
            out.append(
                UserSummaryDTO(
                    username: u,
                    roles: await store.rolesForUser(username: u)))
        }
        return Self.json(UserListResponse(users: out))
    }

    func handleUserCreate(_ request: Request) async
        -> Response
    {
        let decoded = await decodeJSON(request, CreateUserRequest.self)
        guard case .ok(let body) = decoded else {
            return decoded.orFail
        }
        guard let username = Self.safeIdent(body.username) else {
            return Self.error(
                status: .badRequest,
                message: "invalid username [A-Za-z0-9._-]",
                type: "invalid_request_error", code: "invalid_name")
        }
        guard body.password.count >= 8 else {
            return Self.error(
                status: .badRequest,
                message: "password must be >= 8 chars",
                type: "invalid_request_error", code: "weak_password")
        }
        let role = body.role ?? "member"
        guard RBAC.isValidRole(role) else {
            return Self.error(
                status: .badRequest,
                message: "unknown role '\(role)'",
                type: "invalid_request_error", code: "unknown_role")
        }
        let caller = await callerPermissions(request)
        guard
            RBAC.canGrant(role: role, grantorPermissions: caller)
        else {
            await audit(
                request, action: "user.create", target: username,
                result: "denied", detail: "role '\(role)' exceeds "
                    + "grantor permissions")
            return Self.deny403(
                "cannot grant a role conferring permissions you do "
                    + "not hold")
        }
        // Replacing an existing account = a password reset; refuse if
        // the target currently outranks the caller (account takeover).
        if await store.getUser(username: username) != nil {
            let cur = await store.rolesForUser(username: username)
            guard
                RBAC.permissions(forRoles: cur).isSubset(of: caller)
            else {
                await audit(
                    request, action: "user.create",
                    target: username, result: "denied",
                    detail: "target outranks caller")
                return Self.deny403(
                    "refusing to replace a user whose roles exceed "
                        + "your permissions")
            }
        }
        let salt = Passwords.randomSalt()
        let hash = Passwords.derive(
            password: body.password, salt: salt,
            iters: Passwords.defaultIterations)
        do {
            try await store.putUser(
                username: username, salt: salt, hash: hash,
                iters: Passwords.defaultIterations)
            try await store.grantRole(
                username: username, role: role)
        } catch {
            return Self.storeError(error)
        }
        await audit(
            request, action: "user.create", target: username,
            result: "ok", detail: "role=\(role)")
        return Self.json(
            UserSummaryDTO(
                username: username,
                roles: await store.rolesForUser(
                    username: username)))
    }

    func handleUserDelete(
        _ name: String?, _ request: Request
    ) async -> Response {
        guard let username = Self.safeIdent(name) else {
            return Self.error(
                status: .badRequest, message: "invalid username",
                type: "invalid_request_error", code: "invalid_name")
        }
        guard await store.getUser(username: username) != nil else {
            return Self.error(
                status: .notFound, message: "no user '\(username)'",
                type: "invalid_request_error", code: "not_found")
        }
        let onlyAdmin: Bool
        do {
            onlyAdmin = try await store.usersWithRole("admin") == [username]
        } catch {
            // NB11: fail closed — refuse the delete if the admin set can't
            // be read, rather than risk stripping the last admin.
            await audit(
                request, action: "user.delete", target: username,
                result: "denied", detail: "admin-set check failed")
            return Self.error(
                status: .internalServerError,
                message: "could not verify the admin set",
                type: "server_error", code: "store_error")
        }
        if onlyAdmin {
            await audit(
                request, action: "user.delete", target: username,
                result: "denied", detail: "only admin")
            return Self.deny403(
                "'\(username)' is the only admin — refusing "
                    + "(grant admin to another user first)")
        }
        let ok: Bool
        do {
            ok = try await store.deleteUser(username: username)
        } catch {
            // H11 (M66.1): the cascade rolled back as a unit — surface the
            // failure instead of reporting a partial delete as success.
            await audit(
                request, action: "user.delete", target: username,
                result: "denied", detail: "store error")
            return Self.error(
                status: .internalServerError,
                message: "failed to delete user",
                type: "server_error", code: "store_error")
        }
        await audit(
            request, action: "user.delete", target: username,
            result: ok ? "ok" : "denied")
        return Self.json(
            UserRemovedResponse(username: username, removed: ok))
    }

    func handleRoleGrant(
        _ name: String?, _ role: String?, _ request: Request
    ) async -> Response {
        guard let username = Self.safeIdent(name),
            let role = Self.safeIdent(role)
        else {
            return Self.error(
                status: .badRequest, message: "invalid name/role",
                type: "invalid_request_error", code: "invalid_name")
        }
        guard RBAC.isValidRole(role) else {
            return Self.error(
                status: .badRequest,
                message: "unknown role '\(role)'",
                type: "invalid_request_error", code: "unknown_role")
        }
        guard await store.getUser(username: username) != nil else {
            return Self.error(
                status: .notFound, message: "no user '\(username)'",
                type: "invalid_request_error", code: "not_found")
        }
        let caller = await callerPermissions(request)
        guard
            RBAC.canGrant(role: role, grantorPermissions: caller)
        else {
            await audit(
                request, action: "role.grant",
                target: "\(username):\(role)", result: "denied",
                detail: "role exceeds grantor permissions")
            return Self.deny403(
                "cannot grant a role conferring permissions you do "
                    + "not hold")
        }
        do {
            try await store.grantRole(
                username: username, role: role)
        } catch {
            return Self.storeError(error)
        }
        await audit(
            request, action: "role.grant",
            target: "\(username):\(role)", result: "ok")
        return Self.json(OkResponse(ok: true))
    }

    func handleRoleRevoke(
        _ name: String?, _ role: String?, _ request: Request
    ) async -> Response {
        guard let username = Self.safeIdent(name),
            let role = Self.safeIdent(role)
        else {
            return Self.error(
                status: .badRequest, message: "invalid name/role",
                type: "invalid_request_error", code: "invalid_name")
        }
        guard await store.getUser(username: username) != nil else {
            return Self.error(
                status: .notFound, message: "no user '\(username)'",
                type: "invalid_request_error", code: "not_found")
        }
        if role == "admin" {
            let onlyAdmin: Bool
            do {
                onlyAdmin =
                    try await store.usersWithRole("admin") == [username]
            } catch {
                // NB11: can't read the admin set ⇒ fail closed (refuse),
                // never assume "not the last admin" on a query error.
                await audit(
                    request, action: "role.revoke",
                    target: "\(username):\(role)", result: "denied",
                    detail: "admin-set check failed")
                return Self.error(
                    status: .internalServerError,
                    message: "could not verify the admin set",
                    type: "server_error", code: "store_error")
            }
            if onlyAdmin {
                await audit(
                    request, action: "role.revoke",
                    target: "\(username):\(role)", result: "denied",
                    detail: "only admin")
                return Self.deny403(
                    "'\(username)' is the only admin — refusing to "
                        + "revoke admin")
            }
        }
        let ok = await store.revokeRole(
            username: username, role: role)
        await audit(
            request, action: "role.revoke",
            target: "\(username):\(role)", result: ok ? "ok" : "denied")
        return Self.json(OkResponse(ok: ok))
    }

    func handleTokensList() async -> Response {
        let toks = await store.listTokens()
        return Self.json(
            TokenListResponse(
                tokens: toks.map {
                    TokenSummaryDTO(
                        username: $0.username, scope: $0.scoped,
                        hash_prefix: $0.hashPrefix,
                        label: $0.label, expires: $0.expires)
                }))
    }

    func handleTokenCreate(_ request: Request) async
        -> Response
    {
        let decoded = await decodeJSON(
            request, CreateTokenRequest.self)
        guard case .ok(let body) = decoded else {
            return decoded.orFail
        }
        guard let user = Self.safeIdent(body.user) else {
            return Self.error(
                status: .badRequest, message: "invalid user",
                type: "invalid_request_error", code: "invalid_name")
        }
        guard await store.getUser(username: user) != nil else {
            return Self.error(
                status: .notFound, message: "no user '\(user)'",
                type: "invalid_request_error", code: "not_found")
        }
        let caller = await callerPermissions(request)
        let scoped =
            (body.role?.isEmpty == false) ? body.role : nil
        if let scoped {
            for r in scoped where !RBAC.isValidRole(r) {
                return Self.error(
                    status: .badRequest,
                    message: "unknown role '\(r)'",
                    type: "invalid_request_error",
                    code: "unknown_role")
            }
            guard Self.canGrantAll(scoped, caller) else {
                await audit(
                    request, action: "token.create", target: user,
                    result: "denied", detail: "scope exceeds caller")
                return Self.deny403(
                    "token scope exceeds your permissions")
            }
        } else {
            // Unscoped ⇒ inherits the user's full roles; refuse to
            // mint a token more powerful than the caller.
            let inherited = RBAC.permissions(
                forRoles: await store.rolesForUser(username: user))
            guard inherited.isSubset(of: caller) else {
                await audit(
                    request, action: "token.create", target: user,
                    result: "denied",
                    detail: "unscoped token exceeds caller")
                return Self.deny403(
                    "an unscoped token for '\(user)' would exceed "
                        + "your permissions — pass an explicit role "
                        + "scope you can grant")
            }
        }
        let (key, hash) = AuthConfig.mintToken()
        let expires = body.ttl_secs.flatMap {
            $0 > 0 ? Date().timeIntervalSince1970 + Double($0) : nil
        }
        do {
            try await store.putToken(
                hash: hash, username: user, scopedRoles: scoped,
                label: body.label, expires: expires)
        } catch {
            return Self.storeError(error)
        }
        let hashPrefix = String(
            AuthConfig.hex(Array(hash)).prefix(12))
        await audit(
            request, action: "token.create", target: user,
            result: "ok",
            detail: "hash=\(hashPrefix) scope="
                + (scoped?.joined(separator: ",") ?? "inherit"))
        return Self.json(
            CreateTokenResponse(
                user: user, scope: scoped, token: key,
                hash_prefix: hashPrefix),
            status: .accepted)
    }

    /// 64-char hex → 32 bytes (token-hash reconstruction).
    private static func hexData(_ s: String) -> Data? {
        guard s.count == 64 else { return nil }
        var out = [UInt8]()
        out.reserveCapacity(32)
        var i = s.startIndex
        while i < s.endIndex {
            let j = s.index(i, offsetBy: 2)
            guard let b = UInt8(s[i..<j], radix: 16) else {
                return nil
            }
            out.append(b)
            i = j
        }
        return Data(out)
    }

    func handleTokenDelete(
        _ prefix: String?, _ request: Request
    ) async -> Response {
        guard let prefix, prefix.count >= 6,
            prefix.allSatisfy({ $0.isHexDigit })
        else {
            return Self.error(
                status: .badRequest,
                message: "prefix must be >= 6 hex chars",
                type: "invalid_request_error", code: "invalid_prefix")
        }
        let matches = await store.tokensMatchingHashPrefix(prefix)
        if matches.isEmpty {
            return Self.error(
                status: .notFound,
                message: "no token matched \(prefix)",
                type: "invalid_request_error", code: "not_found")
        }
        var removed = 0
        for m in matches {
            if let d = Self.hexData(m.hex),
                await store.deleteToken(hash: d)
            {
                removed += 1
            }
        }
        await audit(
            request, action: "token.delete", target: prefix,
            result: "ok", detail: "removed=\(removed)")
        return Self.json(TokensRemovedResponse(removed: removed))
    }

    /// `POST /api/tokens/{prefix}/rotate` (M36.2) — revoke + reissue.
    /// The prefix must match EXACTLY one token; its owner/scope/label
    /// carry to a fresh secret (returned once), the old hash is deleted.
    /// `ttl_secs` sets the new token's lifetime (absent ⇒ no expiry).
    /// Same scope guard as create: the caller may not re-mint a token
    /// more powerful than they can grant.
    private func handleTokenRotate(
        _ prefix: String?, _ request: Request
    ) async -> Response {
        guard let prefix, prefix.count >= 6,
            prefix.allSatisfy({ $0.isHexDigit })
        else {
            return Self.error(
                status: .badRequest,
                message: "prefix must be >= 6 hex chars",
                type: "invalid_request_error", code: "invalid_prefix")
        }
        let decoded = await decodeJSON(request, RotateTokenRequest.self)
        guard case .ok(let body) = decoded else {
            return decoded.orFail
        }
        let matches = await store.tokensMatchingHashPrefix(prefix)
        guard matches.count == 1, let m = matches.first,
            let oldHash = Self.hexData(m.hex)
        else {
            return Self.error(
                status: matches.count > 1 ? .conflict : .notFound,
                message: matches.count > 1
                    ? "prefix \(prefix) matched \(matches.count) "
                        + "tokens — use a longer, unambiguous prefix"
                    : "no token matched \(prefix)",
                type: "invalid_request_error",
                code: matches.count > 1 ? "ambiguous_prefix"
                    : "not_found")
        }
        // Same scope guard as create — rotation must not re-mint a token
        // more powerful than the caller can grant.
        let caller = await callerPermissions(request)
        if let scoped = m.scoped {
            guard Self.canGrantAll(scoped, caller) else {
                await audit(
                    request, action: "token.rotate", target: m.username,
                    result: "denied", detail: "scope exceeds caller")
                return Self.deny403("token scope exceeds your permissions")
            }
        } else {
            let inherited = RBAC.permissions(
                forRoles: await store.rolesForUser(username: m.username))
            guard inherited.isSubset(of: caller) else {
                await audit(
                    request, action: "token.rotate", target: m.username,
                    result: "denied",
                    detail: "unscoped token exceeds caller")
                return Self.deny403(
                    "rotating this unscoped token would exceed your "
                        + "permissions")
            }
        }
        let expires = body.ttl_secs.flatMap {
            $0 > 0 ? Date().timeIntervalSince1970 + Double($0) : nil
        }
        // NA4 (M66.2): persist the NEW token BEFORE revoking the old one.
        // The previous order (delete-then-put) left the principal with no
        // working token if putToken threw — a lockout on partial failure.
        // New + old briefly coexist (distinct hashes); we revoke the old
        // only once the new is durably stored.
        let (key, hash) = AuthConfig.mintToken()
        do {
            try await store.putToken(
                hash: hash, username: m.username, scopedRoles: m.scoped,
                label: m.label, expires: expires)
        } catch {
            return Self.storeError(error)
        }
        _ = await store.deleteToken(hash: oldHash)
        let newPrefix = String(AuthConfig.hex(Array(hash)).prefix(12))
        await audit(
            request, action: "token.rotate", target: m.username,
            result: "ok",
            detail: "old=\(prefix) new=\(newPrefix)")
        return Self.json(
            CreateTokenResponse(
                user: m.username, scope: m.scoped, token: key,
                hash_prefix: newPrefix),
            status: .accepted)
    }

    /// Shared remediation for the in-handler RBAC/privilege guards
    /// (#12 / M43.4). Every `deny403` is either an "insufficient
    /// privilege" or a "safety guard" refusal, so one hint covers them;
    /// surfaced as `error.hint` and rendered by the CLI client. Mirrors
    /// the auth-middleware forbidden hint for envelope parity.
    static let privilegedActionHint =
        "Refused: this action needs elevated privileges or trips a "
        + "safety guard (e.g. removing the last admin). Use an admin "
        + "token, have an admin perform it, or adjust roles with "
        + "`athena auth role grant <user> <role>`."

    private static func deny403(
        _ msg: String, hint: String? = nil
    ) -> Response {
        Self.error(
            status: .forbidden, message: msg,
            type: "auth_error", code: "forbidden",
            hint: hint ?? privilegedActionHint)
    }

    /// The compiled-in RBAC role→perms catalog. Shared by
    /// `GET /api/roles` (bearer) and `/ui/api/roles` (cookie).
    static func rolesCatalogResponse() -> Response {
        Self.json(
            RolesResponse(
                roles: RBAC.roleNames.map { r in
                    RoleCatalogEntry(
                        role: r,
                        permissions: (RBAC.catalog[r] ?? [])
                            .map(\.rawValue).sorted())
                }))
    }}
