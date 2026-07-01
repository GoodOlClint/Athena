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

// The WebUI surface: the cookie-authed JSON wrappers behind the dashboard,
// model/daemon/RBAC consoles, and session login. Each re-checks the logged-in
// user's RBAC permission (+ CSRF on mutations) then reuses the same handlers
// the token-authed `/api/*` routes call. Page rendering lives in `WebUI.swift`.
extension AthenaServer {
    /// Register every `/ui*` route (dashboard, consoles, session login).
    /// Called from `run()`.
    func registerUIRoutes(_ router: Router<AppRequestContext>) {
        // Monitoring dashboard (M11.2) — inbound-only.
        router.get("/ui") { request, _ -> Response in
            await handleUIDashboard(request)
        }
        router.get("/ui/api/state") { _, _ -> Response in
            await handleUIState()
        }
        router.get("/ui/config") { request, _ -> Response in
            await handleUIConfigPage(request)
        }
        router.get("/ui/api/config") { _, _ -> Response in
            await handleUIConfigGet()
        }
        router.post("/ui/api/config") { request, _ -> Response in
            await handleUIConfigPost(request)
        }

        // Model console (M18.2). SESSION-cookie authed (AuthPolicy
        // gates /ui* on daemonAdmin); each handler ALSO re-checks the
        // logged-in user's model.read/model.write and (mutations) the
        // CSRF token, then REUSES the M16 op methods (ModelStoreOps /
        // performModelOp) — no self-HTTP, no duplication. All static
        // literals ⇒ no Hummingbird trie-order hazard.
        router.get("/ui/models") { request, _ -> Response in
            await handleUIModelsPage(request)
        }
        router.get("/ui/api/models") { request, _ -> Response in
            await uiModelsList(request)
        }
        router.get("/ui/api/models/show") { request, _
            -> Response in
            await uiModelShow(request)
        }
        router.get("/ui/api/models/default") { request, _
            -> Response in
            await uiDefaultGet(request)
        }
        router.post("/ui/api/models/default") { request, _
            -> Response in
            await uiModelMutate(request) {
                await self.handleDefaultModelSet($0)
            }
        }
        router.post("/ui/api/models/rm") { request, _
            -> Response in
            await uiModelRemove(request)
        }
        router.post("/ui/api/models/copy") { request, _
            -> Response in
            await uiModelMutate(request) {
                await self.handleModelCopy($0)
            }
        }
        // ADR 025 S2 — model ops run synchronously now (no async queue).
        // The browser console isn't an SSE consumer (EventSource is
        // GET-only), so these POSTs BLOCK until the op completes and return
        // the terminal result JSON (or the error envelope). The CLI gets
        // streamed SSE progress on the same `/api/models/*` routes.
        router.post("/ui/api/models/pull") { request, _
            -> Response in
            await uiModelMutate(request) { r in
                await self.uiModelOp(kind: "model_pull", r)
            }
        }
        router.post("/ui/api/models/convert") { request, _
            -> Response in
            await uiModelMutate(request) { r in
                await self.uiModelOp(kind: "model_convert", r)
            }
        }
        router.post("/ui/api/models/prune") { request, _
            -> Response in
            await uiModelMutate(request) { r in
                await self.uiModelOp(kind: "model_prune", r)
            }
        }

        // Daemon control console (M18.3). The WebUI runs INSIDE the
        // daemon ⇒ it controls a RUNNING daemon (warm/unload the
        // model, read posture); it CANNOT cold-start a stopped
        // daemon (nothing serves /ui then — use launchd/CLI). Cookie
        // + per-action daemonAdmin re-check; CSRF on the mutations.
        router.get("/ui/daemon") { request, _ -> Response in
            await handleUIDaemonPage(request)
        }
        router.get("/ui/api/admin/status") { request, _
            -> Response in
            await uiAdminStatus(request)
        }
        router.post("/ui/api/admin/load") { request, _
            -> Response in
            await uiAdminLoad(request)
        }
        router.post("/ui/api/admin/stop") { request, _
            -> Response in
            await uiAdminStop(request)
        }

        // ADR 026 — the allowlist is retired; availability IS the model store
        // (classified by ModelSupport), so the `/ui/allowlist` console and its
        // `/ui/api/allowlist*` mutators are gone. The read-only resident/store
        // projection stays via `/ui` + `/api/models/resident`.

        // RBAC admin console (M18.4). Cookie + per-action
        // users.admin/tokens.admin re-check + CSRF, then REUSE the
        // M16.4 handlers (now cookie-aware via callerPermissions) so
        // server-side canGrant / last-admin / cross-rank guards bind
        // to the logged-in user. All static literals (name/role/
        // prefix ride the JSON body, not a :param) ⇒ no trie hazard.
        router.get("/ui/users") { request, _ -> Response in
            await handleUIUsersPage(request)
        }
        router.get("/ui/api/users") { request, _ -> Response in
            await uiUsersList(request)
        }
        router.post("/ui/api/users") { request, _ -> Response in
            await uiUserCreate(request)
        }
        router.post("/ui/api/users/delete") { request, _
            -> Response in
            await uiUserDelete(request)
        }
        router.post("/ui/api/users/role/grant") { request, _
            -> Response in
            await uiRoleGrant(request)
        }
        router.post("/ui/api/users/role/revoke") { request, _
            -> Response in
            await uiRoleRevoke(request)
        }
        router.get("/ui/api/roles") { request, _ -> Response in
            await uiRolesList(request)
        }
        router.get("/ui/api/tokens") { request, _ -> Response in
            await uiTokensList(request)
        }
        router.post("/ui/api/tokens") { request, _ -> Response in
            await uiTokenCreate(request)
        }
        router.post("/ui/api/tokens/delete") { request, _
            -> Response in
            await uiTokenDelete(request)
        }

        // WebUI session login (M12.2). /ui/login + /ui/logout are
        // open (AuthPolicy); the rest of /ui* needs the cookie.
        router.get("/ui/login") { _, _ -> Response in
            Self.html(Self.loginPage(error: nil))
        }
        router.post("/ui/login") { request, context -> Response in
            // A3: throttle by the TCP peer address only (ADR 004 — never
            // trust X-Forwarded-For). `remoteAddress` comes off the channel
            // via AppRequestContext.
            await handleUILoginPost(
                request, peerIP: context.remoteAddress?.ipAddress)
        }
        router.get("/ui/logout") { _, _ -> Response in
            Self.logoutResponse(secure: tlsEnabled)
        }
        router.post("/ui/logout") { _, _ -> Response in
            Self.logoutResponse(secure: tlsEnabled)
        }
    }


    /// `/ui` console model op (ADR 025 S2). EventSource is GET-only, so the
    /// browser POST BLOCKS until the op completes and returns the terminal
    /// result JSON (or the error envelope) — no streaming, no job poll.
    /// `.modelWrite` + CSRF are checked by `uiModelMutate`; audited here.
    private func uiModelOp(kind: String, _ r: Request) async -> Response {
        let body: Data
        do {
            let buf = try await r.body.collect(upTo: 1 * 1024 * 1024)
            body = Data(buffer: buf)
        } catch {
            return Self.error(
                status: .badRequest,
                message: "Invalid request body: \(error)",
                type: "invalid_request_error", code: "invalid_body")
        }
        let action = "model." + kind.replacingOccurrences(
            of: "model_", with: "")
        await audit(r, action: action, target: nil, result: "ok")
        let (result, error) = await performModelOp(kind: kind, body: body)
        if let result {
            return Self.jsonString(
                String(data: result, encoding: .utf8) ?? "{}")
        }
        let detail =
            error
            ?? .init(
                message: "model op failed", type: "server_error",
                code: "model_op_failed")
        return Self.error(
            status: HTTPResponse.Status(
                code: detail.type == "invalid_request_error" ? 400 : 500),
            message: detail.message, type: detail.type, code: detail.code)
    }

    // MARK: - WebUI model console reuse (M18.2)

    /// 403 JSON for a /ui/api/* action the logged-in user's perms
    /// (or CSRF) reject. AuthPolicy already gated /ui* on
    /// daemonAdmin; this is the per-action defense-in-depth
    /// re-check (the page is never trusted). Lives here (not in
    /// WebUI.swift) so it can reuse the file-private M16 op methods.
    private static func uiDeny(_ msg: String) -> Response {
        Self.error(
            status: .forbidden, message: msg,
            type: "auth_error", code: "forbidden")
    }

    /// First raw value of `key` in the query string. Our model
    /// names are `[A-Za-z0-9._-]` (validated downstream by
    /// `safeModelName`) so no percent-decode is needed.
    private static func uiQuery(
        _ r: Request, _ key: String
    ) -> String? {
        guard let q = r.uri.query else { return nil }
        for kv in q.split(separator: "&") {
            let p = kv.split(separator: "=", maxSplits: 1)
            if p.first.map(String.init) == key {
                return p.count == 2 ? String(p[1]) : ""
            }
        }
        return nil
    }

    private func uiModelsList(_ r: Request) async -> Response {
        guard await uiCaller(r).perms.contains(.modelRead) else {
            return Self.uiDeny("need model.read")
        }
        return handleModelsList()
    }

    private func uiModelShow(_ r: Request) async -> Response {
        guard await uiCaller(r).perms.contains(.modelRead) else {
            return Self.uiDeny("need model.read")
        }
        return handleModelShow(Self.uiQuery(r, "name"))
    }

    private func uiDefaultGet(_ r: Request) async -> Response {
        guard await uiCaller(r).perms.contains(.modelRead) else {
            return Self.uiDeny("need model.read")
        }
        return handleDefaultModelGet()
    }

    /// CSRF + per-action `.modelWrite` re-check, THEN delegate to
    /// the shared M16 op (ModelStoreOps / performModelOp). Mutations
    /// only. `op` is non-escaping (invoked here, never stored).
    private func uiModelMutate(
        _ r: Request, _ op: (Request) async -> Response
    ) async -> Response {
        guard csrfOK(r) else {
            return Self.uiDeny("csrf token missing or invalid")
        }
        guard await uiCaller(r).perms.contains(.modelWrite) else {
            return Self.uiDeny("need model.write")
        }
        return await op(r)
    }

    private func uiModelRemove(_ r: Request) async -> Response {
        await uiModelMutate(r) { req in
            let decoded = await self.decodeJSON(
                req, SetDefaultModelRequest.self)
            guard case .ok(let body) = decoded else {
                return decoded.orFail
            }
            return await self.handleModelRemove(body.name, req)
        }
    }

    // MARK: - WebUI daemon console reuse (M18.3)

    private func uiAdminStatus(_ r: Request) async -> Response {
        guard await uiCaller(r).perms.contains(.daemonAdmin) else {
            return Self.uiDeny("need daemon.admin")
        }
        return await adminStatus()
    }

    /// Destructive: CSRF + per-action `.daemonAdmin` re-check on the
    /// logged-in user, then the shared unload op. The UI also gates
    /// this behind an explicit confirm step (the page is not
    /// trusted; the server decides).
    private func uiAdminStop(_ r: Request) async -> Response {
        guard csrfOK(r) else {
            return Self.uiDeny("csrf token missing or invalid")
        }
        guard await uiCaller(r).perms.contains(.daemonAdmin) else {
            return Self.uiDeny("need daemon.admin")
        }
        return await adminUnloadLLM(r)
    }

    private func uiAdminLoad(_ r: Request) async -> Response {
        guard csrfOK(r) else {
            return Self.uiDeny("csrf token missing or invalid")
        }
        guard await uiCaller(r).perms.contains(.daemonAdmin) else {
            return Self.uiDeny("need daemon.admin")
        }
        return await adminLoadLLM(r)
    }

    // MARK: - WebUI RBAC admin reuse (M18.4)

    /// Small `{key:value}` JSON body → map (for the name/role/prefix
    /// the M16.4 handlers take as path params over `/api/*`; the
    /// cookie /ui passes them in the body instead). Only consumed
    /// for delete/grant/revoke — handlers that DON'T re-read the
    /// body; create/token-create are passed the Request untouched so
    /// they decode their own typed DTO.
    private func uiJSONMap(_ r: Request) async -> [String: String] {
        guard
            let buf = try? await r.body.collect(upTo: 64 * 1024),
            let m = try? JSONDecoder().decode(
                [String: String].self, from: Data(buffer: buf))
        else { return [:] }
        return m
    }

    /// Cookie + per-action perm re-check (+ CSRF when `mutating`),
    /// then run `op`. Reuses the M16.4 handlers as-is: their inner
    /// `callerPermissions` is now cookie-aware (M18.4) so canGrant /
    /// last-admin / cross-rank guards bind to the LOGGED-IN user.
    private func uiRBAC(
        _ r: Request, _ need: Permission, mutating: Bool,
        _ op: (Request) async -> Response
    ) async -> Response {
        if mutating, !csrfOK(r) {
            return Self.uiDeny("csrf token missing or invalid")
        }
        guard await uiCaller(r).perms.contains(need) else {
            return Self.uiDeny("need \(need.rawValue)")
        }
        return await op(r)
    }

    private func uiUsersList(_ r: Request) async -> Response {
        await uiRBAC(r, .usersRead, mutating: false) { _ in
            await self.handleUsersList()
        }
    }
    private func uiRolesList(_ r: Request) async -> Response {
        await uiRBAC(r, .usersRead, mutating: false) { _ in
            Self.rolesCatalogResponse()
        }
    }
    private func uiUserCreate(_ r: Request) async -> Response {
        await uiRBAC(r, .usersAdmin, mutating: true) { req in
            await self.handleUserCreate(req)
        }
    }
    private func uiUserDelete(_ r: Request) async -> Response {
        await uiRBAC(r, .usersAdmin, mutating: true) { req in
            let f = await self.uiJSONMap(req)
            return await self.handleUserDelete(f["name"], req)
        }
    }
    private func uiRoleGrant(_ r: Request) async -> Response {
        await uiRBAC(r, .usersAdmin, mutating: true) { req in
            let f = await self.uiJSONMap(req)
            return await self.handleRoleGrant(
                f["name"], f["role"], req)
        }
    }
    private func uiRoleRevoke(_ r: Request) async -> Response {
        await uiRBAC(r, .usersAdmin, mutating: true) { req in
            let f = await self.uiJSONMap(req)
            return await self.handleRoleRevoke(
                f["name"], f["role"], req)
        }
    }
    private func uiTokensList(_ r: Request) async -> Response {
        await uiRBAC(r, .tokensAdmin, mutating: false) { _ in
            await self.handleTokensList()
        }
    }
    private func uiTokenCreate(_ r: Request) async -> Response {
        await uiRBAC(r, .tokensAdmin, mutating: true) { req in
            await self.handleTokenCreate(req)
        }
    }
    private func uiTokenDelete(_ r: Request) async -> Response {
        await uiRBAC(r, .tokensAdmin, mutating: true) { req in
            let f = await self.uiJSONMap(req)
            return await self.handleTokenDelete(f["prefix"], req)
        }
    }}
