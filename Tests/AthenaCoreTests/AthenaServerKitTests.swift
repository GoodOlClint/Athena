import Foundation
import HTTPTypes
import Logging
import NIOCore
import NIOEmbedded
import NIOHTTPTypes
import XCTest

import AthenaCore
import AthenaStore
@testable import AthenaServerKit

/// M70.1 (audit NA2) — the daemon's HTTP-server security boundary, now a
/// testable library (`AthenaServerKit`) instead of unreachable executable
/// code. Table-driven coverage of the pure seams the audit named:
/// constant-time compare, bearer resolve + token expiry, the
/// route→permission map, the rate-limit + concurrency token buckets, the
/// WebUI session/CSRF HMAC, the multipart reader, and the Prometheus
/// metrics + nearest-rank percentile math. All CI-safe (no MLX, no
/// metallib, tmp SQLite only).
///
/// (XCTAssert autoclosures can't `await`, and an actor-isolated call in an
/// autoclosure fails the same way — awaited values are hoisted to `let`s.)
final class AthenaServerKitTests: XCTestCase {

    // MARK: - constantTimeEqual

    func testConstantTimeEqual() {
        XCTAssertTrue(AuthConfig.constantTimeEqual([], []))
        XCTAssertTrue(AuthConfig.constantTimeEqual([1, 2, 3], [1, 2, 3]))
        XCTAssertFalse(AuthConfig.constantTimeEqual([1, 2, 3], [1, 2, 4]))
        // Length mismatch fails closed (no out-of-bounds, no partial match).
        XCTAssertFalse(AuthConfig.constantTimeEqual([1, 2], [1, 2, 3]))
        XCTAssertFalse(AuthConfig.constantTimeEqual([1], []))
    }

    // MARK: - mintToken format

    func testMintTokenFormatAndHash() {
        let (key, hash) = AuthConfig.mintToken()
        XCTAssertTrue(
            key.hasPrefix("sk-athena-"), "minted key carries the prefix")
        // base64url alphabet only after the prefix (no +/= padding).
        let body = String(key.dropFirst("sk-athena-".count))
        XCTAssertFalse(body.contains("+"))
        XCTAssertFalse(body.contains("/"))
        XCTAssertFalse(body.contains("="))
        // The at-rest hash is exactly SHA-256(key).
        XCTAssertEqual(hash, Data(AuthConfig.sha(key)))
        // Distinct mints differ (256-bit entropy).
        let (k2, _) = AuthConfig.mintToken()
        XCTAssertNotEqual(key, k2)
    }

    // MARK: - resolve(bearer:) — bootstrap hashes (no DB)

    func testResolveBootstrapHashMatchAndMiss() async {
        let adminKey = "sk-athena-bootstrap-admin"
        let cfg = AuthConfig(hashes: [AuthConfig.sha(adminKey): ["admin"]])
        XCTAssertTrue(cfg.isEnabled, "any bootstrap hash enables auth")

        let hit = await cfg.resolve(bearer: adminKey)
        XCTAssertNotNil(hit)
        // Synthetic principal = "t:" + hex(sha(key)) (stable, DB-less).
        XCTAssertEqual(
            hit?.principal, "t:" + AuthConfig.hex(AuthConfig.sha(adminKey)))
        XCTAssertEqual(
            hit?.permissions, RBAC.permissions(forRoles: ["admin"]))

        let miss = await cfg.resolve(bearer: "sk-athena-not-a-key")
        XCTAssertNil(miss, "unknown token resolves to nil")
    }

    // MARK: - resolve(bearer:) — DB token + per-token expiry

    func testResolveDBTokenExpiry() async throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("athena-serverkit-\(UUID()).sqlite")
        defer { try? FileManager.default.removeItem(at: url) }
        let store = try AthenaStore(path: url)

        try await store.putUser(
            username: "alice",
            salt: Passwords.randomSalt(),
            hash: Data(repeating: 0, count: 32),
            iters: Passwords.defaultIterations)
        try await store.grantRole(username: "alice", role: "admin")

        let key = "sk-athena-alice-token"
        let hash = Data(AuthConfig.sha(key))
        let cfg = AuthConfig(store: store, enabled: true)

        // Future expiry → resolves to the owning user with the role's perms.
        try await store.putToken(
            hash: hash, username: "alice", scopedRoles: nil, label: "t",
            expires: Date().timeIntervalSince1970 + 3600)
        let live = await cfg.resolve(bearer: key)
        XCTAssertEqual(live?.principal, "u:alice")
        XCTAssertEqual(
            live?.permissions, RBAC.permissions(forRoles: ["admin"]))

        // Past expiry → nil, indistinguishable from an unknown token.
        try await store.putToken(
            hash: hash, username: "alice", scopedRoles: nil, label: "t",
            expires: Date().timeIntervalSince1970 - 1)
        let expired = await cfg.resolve(bearer: key)
        XCTAssertNil(expired, "an expired token must not resolve")
    }

    // MARK: - validateStartup fail-safe

    func testValidateStartupFailsafe() {
        let open = AuthConfig()  // no credentials
        XCTAssertFalse(open.isEnabled)
        // Open mode is confined to loopback.
        XCTAssertNoThrow(try open.validateStartup(listenHost: "127.0.0.1"))
        XCTAssertNoThrow(try open.validateStartup(listenHost: "::1"))
        XCTAssertNoThrow(try open.validateStartup(listenHost: "localhost"))
        XCTAssertThrowsError(
            try open.validateStartup(listenHost: "0.0.0.0"))
        // With credentials, a non-loopback bind is allowed.
        let enabled = AuthConfig(hashes: [AuthConfig.sha("k"): ["admin"]])
        XCTAssertNoThrow(
            try enabled.validateStartup(listenHost: "0.0.0.0"))
    }

    // MARK: - AuthConfig.load (env)

    func testLoadFromEnv() {
        let cfg = AuthConfig.load(
            file: nil,
            env: [
                "ATHENA_ADMIN_KEYS": "sk-admin-1,sk-admin-2",
                "ATHENA_INFERENCE_KEYS": "sk-infer-1",
            ],
            log: Logger(label: "test"))
        XCTAssertTrue(cfg.isEnabled)
    }

    // MARK: - AuthPolicy.required (route → permission map)

    func testAuthPolicyRequiredTable() {
        let cases: [(method: String, path: String, want: Permission?)] = [
            ("GET", "/healthz", nil),
            ("GET", "/openapi.json", nil),
            ("POST", "/ui/login", nil),
            ("GET", "/ui/logout", nil),
            ("GET", "/metrics", .metricsRead),
            ("GET", "/ui", .daemonAdmin),
            ("GET", "/ui/users", .daemonAdmin),
            ("GET", "/v1/models", .modelRead),
            ("GET", "/v1/models/abc", .modelRead),
            ("GET", "/api/admin", .daemonAdmin),
            ("GET", "/api/audit", .daemonAdmin),
            ("GET", "/api/logs", .daemonAdmin),
            ("GET", "/api/logs/stream", .daemonAdmin),
            ("GET", "/api/cache/prompt", .daemonAdmin),
            ("GET", "/api/usage", .inference),
            ("GET", "/api/models", .modelRead),
            ("POST", "/api/models/load", .modelWrite),
            ("GET", "/api/roles", .usersRead),
            ("GET", "/api/users", .usersRead),
            ("POST", "/api/users", .usersAdmin),
            ("GET", "/api/tokens", .tokensAdmin),
            ("POST", "/api/tokens", .tokensAdmin),
            ("POST", "/v1/queue", .queueSubmit),
            // Inference catch-all (unlisted routes fail closed to .inference).
            ("POST", "/v1/chat/completions", .inference),
            ("POST", "/v1/embeddings", .inference),
            // /v1/video/* (ADR 022) is inference-tier like /v1/audio/*.
            ("POST", "/v1/video/transcriptions", .inference),
            ("POST", "/api/chat", .inference),
            ("GET", "/some/unknown/route", .inference),
        ]
        for c in cases {
            XCTAssertEqual(
                AuthPolicy.required(method: c.method, path: c.path),
                c.want,
                "\(c.method) \(c.path)")
        }
    }

    // MARK: - RateLimiter.take (token bucket)

    func testRateLimiterTakeBurstRefillRetryAfter() async {
        let rl = RateLimiter(rate: 1, burst: 2)
        // Two tokens admitted at t=0.
        let a = await rl.take("p", now: 0)
        let b = await rl.take("p", now: 0)
        XCTAssertNil(a)
        XCTAssertNil(b)
        // Third is rejected with a whole-second Retry-After (>= 1).
        let c = await rl.take("p", now: 0)
        XCTAssertEqual(c, 1)
        // After 2s of refill (rate 1/s, cap 2) a token is available again.
        let d = await rl.take("p", now: 2)
        XCTAssertNil(d)
        // A different principal has its own independent bucket.
        let other = await rl.take("q", now: 0)
        XCTAssertNil(other)
    }

    // MARK: - ConcurrencyLimiter (global + per-principal)

    func testConcurrencyLimiterCaps() async {
        // global 2, perPrincipal 1.
        let cl = ConcurrencyLimiter(global: 2, perPrincipal: 1)
        let a1 = await cl.acquire("a")
        XCTAssertTrue(a1)
        // Same principal exceeds its per-principal cap of 1.
        let a2 = await cl.acquire("a")
        XCTAssertFalse(a2)
        // A second principal still fits under the global cap of 2.
        let b1 = await cl.acquire("b")
        XCTAssertTrue(b1)
        // A third distinct principal is rejected — global cap full.
        let c1 = await cl.acquire("c")
        XCTAssertFalse(c1)
        // Releasing frees a slot in both dimensions.
        await cl.release("a")
        let a3 = await cl.acquire("a")
        XCTAssertTrue(a3)
    }

    func testConcurrencyLimiterZeroIsUnlimited() async {
        let cl = ConcurrencyLimiter(global: 0, perPrincipal: 0)
        var allAdmitted = true
        for _ in 0..<50 {
            let ok = await cl.acquire("p")
            allAdmitted = allAdmitted && ok
        }
        XCTAssertTrue(allAdmitted, "0 means unlimited for that dimension")
    }

    // MARK: - Session validate + CSRF (HMAC + expiry)

    func testSessionMintValidateExpiry() {
        let s = Session()
        let now = Date(timeIntervalSince1970: 1_000_000)
        let token = s.mint(user: "alice", now: now)
        let ok = s.validate(token, now: now)
        XCTAssertEqual(ok, "alice")
        // Just before expiry still valid; after the TTL it is not.
        XCTAssertEqual(
            s.validate(token, now: now.addingTimeInterval(Session.ttl - 1)),
            "alice")
        XCTAssertNil(
            s.validate(token, now: now.addingTimeInterval(Session.ttl + 1)))
        // Tampered payload fails the HMAC.
        XCTAssertNil(s.validate(token + "x", now: now))
        XCTAssertNil(s.validate("garbage", now: now))
        // A different Session (different per-process secret) rejects it.
        XCTAssertNil(Session().validate(token, now: now))
    }

    func testSessionCSRF() {
        let s = Session()
        let csrf = s.csrf(user: "alice")
        XCTAssertTrue(s.validateCSRF(csrf, user: "alice"))
        // Bound to the user — another account's name won't validate.
        XCTAssertFalse(s.validateCSRF(csrf, user: "bob"))
        XCTAssertFalse(s.validateCSRF(nil, user: "alice"))
        XCTAssertFalse(s.validateCSRF("not-base64url-hmac", user: "alice"))
    }

    func testSessionCookieParsing() {
        let header = "foo=bar; \(Session.cookieName)=tok123; baz=qux"
        XCTAssertEqual(Session.token(fromCookieHeader: header), "tok123")
        XCTAssertNil(Session.token(fromCookieHeader: "foo=bar"))
        XCTAssertNil(Session.token(fromCookieHeader: nil))
        // Secure attribute only when the daemon serves TLS (A12).
        XCTAssertTrue(Session.setCookie("v", secure: true).contains("Secure"))
        XCTAssertFalse(
            Session.setCookie("v", secure: false).contains("Secure"))
    }

    // MARK: - MultipartForm

    func testMultipartFormParse() throws {
        let boundary = "X-BOUND-123"
        var body = Data()
        func add(_ s: String) { body.append(Data(s.utf8)) }
        add("--\(boundary)\r\n")
        add("Content-Disposition: form-data; name=\"model\"\r\n\r\n")
        add("whisper-1\r\n")
        add("--\(boundary)\r\n")
        add(
            "Content-Disposition: form-data; name=\"file\"; "
                + "filename=\"a.wav\"\r\n")
        add("Content-Type: application/octet-stream\r\n\r\n")
        body.append(Data([0x01, 0x02, 0x03]))
        add("\r\n")
        add("--\(boundary)--\r\n")

        let form = try XCTUnwrap(
            MultipartForm(body: body, boundary: boundary))
        XCTAssertEqual(form.text("model"), "whisper-1")
        let file = try XCTUnwrap(form.first("file"))
        XCTAssertEqual(file.filename, "a.wav")
        XCTAssertEqual(file.data, Data([0x01, 0x02, 0x03]))
        XCTAssertNil(form.first("missing"))
    }

    func testMultipartBoundaryFromContentType() {
        XCTAssertEqual(
            MultipartForm.boundary(
                fromContentType: "multipart/form-data; boundary=abc123"),
            "abc123")
        XCTAssertEqual(
            MultipartForm.boundary(
                fromContentType: "multipart/form-data; boundary=\"q d\""),
            "q d")
        XCTAssertNil(
            MultipartForm.boundary(fromContentType: "application/json"))
    }

    // MARK: - AthenaMetrics.percentile (nearest-rank) + prometheus

    func testPercentileNearestRank() {
        XCTAssertEqual(AthenaMetrics.percentile([], 0.5), 0)
        let four: [Double] = [10, 20, 30, 40]
        // rank = ceil(0.5*4)=2 → index 1 → 20; ceil(0.95*4)=4 → index 3 → 40.
        XCTAssertEqual(AthenaMetrics.percentile(four, 0.5), 20)
        XCTAssertEqual(AthenaMetrics.percentile(four, 0.95), 40)
        // n=2: p95 picks the larger sample, p50 the smaller (M69.2 A25).
        XCTAssertEqual(AthenaMetrics.percentile([5, 9], 0.95), 9)
        XCTAssertEqual(AthenaMetrics.percentile([5, 9], 0.5), 5)
        XCTAssertEqual(AthenaMetrics.percentile([7], 0.99), 7)
    }

    func testPercentileMatchesSnapshot() async {
        let m = AthenaMetrics()
        for v in [10.0, 20, 30, 40] {
            await m.record(kind: "chat", ms: v, isError: false)
        }
        let snap = await m.snapshot()
        XCTAssertEqual(snap.p50Ms, AthenaMetrics.percentile([10, 20, 30, 40], 0.5))
        XCTAssertEqual(snap.p95Ms, AthenaMetrics.percentile([10, 20, 30, 40], 0.95))
        XCTAssertEqual(snap.totalRequests, 4)
        XCTAssertEqual(snap.latencyWindow, 4)
    }

    func testPrometheusExposition() {
        let snap = AthenaMetrics.Snapshot(
            totalRequests: 5, totalErrors: 1,
            byKind: ["chat": 3, "embeddings": 2],
            avgMs: 12.5, p50Ms: 10, p95Ms: 40, llmTokens: 123,
            sinceEpoch: 1000, latencyWindow: 4, auditWriteFailures: 2)
        let out = AthenaMetrics.prometheus(snap, now: 1100)
        XCTAssertTrue(out.contains("athena_requests_total 5"))
        XCTAssertTrue(out.contains("athena_request_errors_total 1"))
        XCTAssertTrue(out.contains("athena_llm_tokens_total 123"))
        XCTAssertTrue(
            out.contains("athena_audit_write_failures_total 2"))
        XCTAssertTrue(
            out.contains(
                "athena_request_latency_ms{quantile=\"0.5\"} 10.0"))
        XCTAssertTrue(
            out.contains(
                "athena_request_latency_ms{quantile=\"0.95\"} 40.0"))
        XCTAssertTrue(
            out.contains(
                "athena_requests_by_kind_total{kind=\"chat\"} 3"))
        // uptime = now - sinceEpoch.
        XCTAssertTrue(out.contains("athena_uptime_seconds 100.0"))
        XCTAssertTrue(out.contains("# TYPE athena_requests_total counter"))
    }

    func testPrometheusEscapesLabel() {
        let snap = AthenaMetrics.Snapshot(
            totalRequests: 1, totalErrors: 0,
            byKind: ["a\"b\\c\nd": 1],
            avgMs: 0, p50Ms: 0, p95Ms: 0, llmTokens: 0,
            sinceEpoch: 0, latencyWindow: 0, auditWriteFailures: 0)
        let out = AthenaMetrics.prometheus(snap, now: 0)
        // Quote, backslash, and newline are escaped in the label value.
        XCTAssertTrue(out.contains(#"kind="a\"b\\c\nd""#))
    }

    // MARK: - Passwords PBKDF2 round-trip

    func testPasswordsDeriveVerify() {
        let salt = Passwords.randomSalt()
        XCTAssertEqual(salt.count, Passwords.saltLen)
        let hash = Passwords.derive(
            password: "correct horse", salt: salt,
            iters: Passwords.defaultIterations)
        XCTAssertEqual(hash.count, Passwords.hashLen)
        XCTAssertTrue(
            Passwords.verify(
                password: "correct horse", salt: salt, hash: hash,
                iters: Passwords.defaultIterations))
        XCTAssertFalse(
            Passwords.verify(
                password: "wrong", salt: salt, hash: hash,
                iters: Passwords.defaultIterations))
    }

    // MARK: - UploadLimit (ADR 017)

    func testUploadLimitContentLengthDecision() {
        let cap = 104_857_600  // 100 MiB
        // Absent / unparseable Content-Length never rejects up front — the
        // streamed collect backstop enforces the cap as the body arrives.
        XCTAssertEqual(
            UploadLimit.check(contentLength: nil, cap: cap), .proceed)
        // Within cap proceeds; exactly at the cap proceeds (cap is inclusive).
        XCTAssertEqual(
            UploadLimit.check(contentLength: 1, cap: cap), .proceed)
        XCTAssertEqual(
            UploadLimit.check(contentLength: cap, cap: cap), .proceed)
        // One byte over the cap rejects up front.
        XCTAssertEqual(
            UploadLimit.check(contentLength: cap + 1, cap: cap),
            .rejectTooLarge)
        XCTAssertEqual(
            UploadLimit.check(contentLength: 200_000_000, cap: cap),
            .rejectTooLarge)
    }

    func testUploadLimitMessageStatesCapAndLeaksNoType() {
        let msg = UploadLimit.tooLargeMessage(cap: 104_857_600)
        XCTAssertTrue(msg.contains("104857600"), "states the cap in bytes")
        // Must not leak the internal NIO error type the old 400 exposed.
        XCTAssertFalse(msg.contains("NIOTooManyBytesError"))
        XCTAssertFalse(msg.contains("Optional"))
    }

    // MARK: - ExpectContinueHandler (ADR 017)

    func testExpectContinuePredicate() {
        var withExpect = HTTPFields()
        withExpect[.expect] = "100-continue"
        XCTAssertTrue(
            ExpectContinueHandler.expectsContinue(
                HTTPRequest(
                    method: .post, scheme: "http", authority: "h",
                    path: "/", headerFields: withExpect)))
        // Case-insensitive value (RFC 9110 §10.1.1).
        var mixed = HTTPFields()
        mixed[.expect] = "100-Continue"
        XCTAssertTrue(
            ExpectContinueHandler.expectsContinue(
                HTTPRequest(
                    method: .post, scheme: "http", authority: "h",
                    path: "/", headerFields: mixed)))
        // Absent ⇒ false (no interim emitted).
        XCTAssertFalse(
            ExpectContinueHandler.expectsContinue(
                HTTPRequest(
                    method: .post, scheme: "http", authority: "h",
                    path: "/", headerFields: HTTPFields())))
    }

    func testExpectContinueHandlerEmitsInterimAndForwardsHead() throws {
        let channel = EmbeddedChannel(handler: ExpectContinueHandler())
        defer { _ = try? channel.finish() }
        var fields = HTTPFields()
        fields[.expect] = "100-continue"
        let req = HTTPRequest(
            method: .post, scheme: "http", authority: "h",
            path: "/v1/audio/transcriptions", headerFields: fields)
        try channel.writeInbound(HTTPRequestPart.head(req))

        // The interim 100 Continue is written outbound …
        let out = try channel.readOutbound(as: HTTPResponsePart.self)
        guard case .head(let resp)? = out else {
            return XCTFail("expected an outbound 100 Continue head")
        }
        XCTAssertEqual(resp.status, .continue)
        XCTAssertNil(
            try channel.readOutbound(as: HTTPResponsePart.self),
            "exactly one interim head")
        // … and the request head is forwarded inbound, unchanged.
        let fwd = try channel.readInbound(as: HTTPRequestPart.self)
        guard case .head(let head)? = fwd else {
            return XCTFail("request head not forwarded")
        }
        XCTAssertEqual(head.path, "/v1/audio/transcriptions")
    }

    func testExpectContinueHandlerPassthroughWithoutExpect() throws {
        let channel = EmbeddedChannel(handler: ExpectContinueHandler())
        defer { _ = try? channel.finish() }
        let req = HTTPRequest(
            method: .post, scheme: "http", authority: "h",
            path: "/v1/chat/completions", headerFields: HTTPFields())
        try channel.writeInbound(HTTPRequestPart.head(req))

        // No interim response when the client didn't ask for one …
        XCTAssertNil(try channel.readOutbound(as: HTTPResponsePart.self))
        // … and the head still flows through.
        let fwd = try channel.readInbound(as: HTTPRequestPart.self)
        guard case .head? = fwd else {
            return XCTFail("request head not forwarded")
        }
    }
}
