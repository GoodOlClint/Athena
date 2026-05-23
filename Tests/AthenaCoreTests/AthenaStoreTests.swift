import Foundation
import XCTest

@testable import AthenaStore

/// M7.1 — embedded SQLite store. Pure, CI-safe (tmp DB file).
/// (XCTAssert autoclosures can't contain `await`, so awaited values
/// are hoisted to `let`s first.)
final class AthenaStoreTests: XCTestCase {

    private func tmpURL() -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("athena-store-\(UUID()).sqlite")
    }

    func testVectorRoundTripDeleteCount() async throws {
        let url = tmpURL()
        defer { try? FileManager.default.removeItem(at: url) }
        let s = try AthenaStore(path: url)

        let v = (0..<8).map { Float($0) * 0.5 }
        let meta = Data(#"{"src":"unit"}"#.utf8)
        try await s.putVector(id: "a", vector: v, metadata: meta)
        try await s.putVector(id: "b", vector: [1, 2, 3], metadata: nil)

        let a = await s.getVector(id: "a")
        XCTAssertEqual(a?.vector, v, "exact float round-trip")
        XCTAssertEqual(a?.metadata, meta)
        let bMeta = await s.getVector(id: "b")?.metadata
        XCTAssertNil(bMeta)
        var cnt = await s.vectorCount()
        XCTAssertEqual(cnt, 2)

        // INSERT OR REPLACE.
        try await s.putVector(id: "a", vector: [9], metadata: nil)
        let aReplaced = await s.getVector(id: "a")?.vector
        XCTAssertEqual(aReplaced, [9])
        cnt = await s.vectorCount()
        XCTAssertEqual(cnt, 2)

        let ids = await s.allVectors().map { $0.id }
        XCTAssertEqual(ids, ["a", "b"])  // ORDER BY id

        let delA = await s.deleteVector(id: "a")
        XCTAssertTrue(delA)
        let delMissing = await s.deleteVector(id: "missing")
        XCTAssertFalse(delMissing)
        cnt = await s.vectorCount()
        XCTAssertEqual(cnt, 1)
        let gone = await s.getVector(id: "a")
        XCTAssertNil(gone)
    }

    func testJobLifecycle() async throws {
        let url = tmpURL()
        defer { try? FileManager.default.removeItem(at: url) }
        let s = try AthenaStore(path: url)

        let req = Data("{\"prompt\":\"hi\"}".utf8)
        try await s.insertJob(
            id: "j1", kind: "conversation", request: req,
            owner: "u:alice")
        let q = await s.getJob(id: "j1")
        XCTAssertEqual(q?.status, "queued")
        XCTAssertEqual(q?.owner, "u:alice")
        XCTAssertEqual(q?.kind, "conversation")
        XCTAssertEqual(q?.request, req)
        XCTAssertNil(q?.result)
        XCTAssertNil(q?.error)
        XCTAssertGreaterThan(q?.created ?? 0, 0)

        let res = Data("{\"text\":\"ok\"}".utf8)
        try await s.updateJob(
            id: "j1", status: "done", result: res, error: nil)
        let done = await s.getJob(id: "j1")
        XCTAssertEqual(done?.status, "done")
        XCTAssertEqual(done?.result, res)
        XCTAssertGreaterThanOrEqual(
            done?.updated ?? 0,
            done?.created ?? .greatestFiniteMagnitude)

        try await s.insertJob(
            id: "j2", kind: "embeddings", request: Data(),
            owner: nil)
        let count = await s.listJobs().count
        XCTAssertEqual(count, 2)
        let queued = await s.listJobs(status: "queued").map { $0.id }
        XCTAssertEqual(queued, ["j2"])

        let d1 = await s.deleteJob(id: "j1")
        XCTAssertTrue(d1)
        let dNope = await s.deleteJob(id: "nope")
        XCTAssertFalse(dNope)
        let j1Gone = await s.getJob(id: "j1")
        XCTAssertNil(j1Gone)
        let after = await s.listJobs().count
        XCTAssertEqual(after, 1)
    }

    func testUserRolesAndCascadingDelete() async throws {
        let url = tmpURL()
        defer { try? FileManager.default.removeItem(at: url) }
        let s = try AthenaStore(path: url)

        let salt = Data(repeating: 7, count: 16)
        let hash = Data(repeating: 9, count: 32)
        try await s.putUser(
            username: "alice", salt: salt, hash: hash, iters: 1000)
        try await s.grantRole(username: "alice", role: "admin")
        try await s.grantRole(username: "alice", role: "operator")
        // INSERT OR IGNORE — re-grant is a no-op, no duplicate.
        try await s.grantRole(username: "alice", role: "admin")

        let roles = await s.rolesForUser(username: "alice")
        XCTAssertEqual(roles, ["admin", "operator"])  // ORDER BY role

        try await s.putUser(
            username: "bob", salt: salt, hash: hash, iters: 1000)
        try await s.grantRole(username: "bob", role: "admin")
        let admins = await s.usersWithRole("admin")
        XCTAssertEqual(admins, ["alice", "bob"])

        let revoked = await s.revokeRole(
            username: "alice", role: "operator")
        XCTAssertTrue(revoked)
        let revMissing = await s.revokeRole(
            username: "alice", role: "operator")
        XCTAssertFalse(revMissing)
        let after = await s.rolesForUser(username: "alice")
        XCTAssertEqual(after, ["admin"])

        // Token bound to alice + a scoped narrowing.
        try await s.putToken(
            hash: Data(repeating: 1, count: 32), username: "alice",
            scopedRoles: ["member"], label: "scoped")
        try await s.putToken(
            hash: Data(repeating: 2, count: 32), username: "alice",
            scopedRoles: nil, label: nil)
        let scoped = await s.tokenPrincipal(
            hash: Data(repeating: 1, count: 32))
        XCTAssertEqual(scoped?.username, "alice")
        XCTAssertEqual(scoped?.scopedRoles, ["member"])
        let unscoped = await s.tokenPrincipal(
            hash: Data(repeating: 2, count: 32))
        XCTAssertEqual(unscoped?.username, "alice")
        XCTAssertNil(unscoped?.scopedRoles)  // NULL ⇒ inherit
        let toks = await s.listTokens()
        XCTAssertEqual(toks.count, 2)
        XCTAssertTrue(toks.allSatisfy { $0.username == "alice" })

        // Deleting alice cascades: her roles AND her tokens vanish;
        // bob (also admin) is untouched.
        let delAlice = await s.deleteUser(username: "alice")
        XCTAssertTrue(delAlice)
        let aliceRoles = await s.rolesForUser(username: "alice")
        XCTAssertEqual(aliceRoles, [])
        let aliceTok = await s.tokenPrincipal(
            hash: Data(repeating: 1, count: 32))
        XCTAssertNil(aliceTok)
        let tokCount = await s.tokenCount()
        XCTAssertEqual(tokCount, 0)
        let adminsLeft = await s.usersWithRole("admin")
        XCTAssertEqual(adminsLeft, ["bob"])
        let delMissing = await s.deleteUser(username: "ghost")
        XCTAssertFalse(delMissing)
    }

    func testTokenExpiryRoundTrip() async throws {
        let url = tmpURL()
        defer { try? FileManager.default.removeItem(at: url) }
        let s = try AthenaStore(path: url)
        try await s.putUser(
            username: "carol", salt: Data([1]), hash: Data([2]),
            iters: 1)
        let future = Date().timeIntervalSince1970 + 3600
        // One token with an explicit expiry, one without (never).
        try await s.putToken(
            hash: Data(repeating: 7, count: 32), username: "carol",
            scopedRoles: nil, label: "ttl", expires: future)
        try await s.putToken(
            hash: Data(repeating: 8, count: 32), username: "carol",
            scopedRoles: nil, label: "forever")
        let withExp = await s.tokenPrincipal(
            hash: Data(repeating: 7, count: 32))
        XCTAssertEqual(withExp?.username, "carol")
        XCTAssertEqual(withExp?.expires, future)
        XCTAssertGreaterThan(withExp?.created ?? 0, 0)  // mint time set
        let noExp = await s.tokenPrincipal(
            hash: Data(repeating: 8, count: 32))
        XCTAssertNil(noExp?.expires)  // NULL ⇒ never expires
        XCTAssertGreaterThan(noExp?.created ?? 0, 0)
    }

    func testErrorStatusAndPersistenceAcrossOpen() async throws {
        let url = tmpURL()
        defer { try? FileManager.default.removeItem(at: url) }
        do {
            let s = try AthenaStore(path: url)
            try await s.insertJob(
                id: "e", kind: "x", request: Data("q".utf8),
                owner: nil)
            try await s.updateJob(
                id: "e", status: "error", result: nil, error: "boom")
            try await s.putVector(
                id: "persist", vector: [3, 1, 4], metadata: nil)
        }
        let s2 = try AthenaStore(path: url)  // reopen same file
        let j = await s2.getJob(id: "e")
        XCTAssertEqual(j?.status, "error")
        XCTAssertEqual(j?.error, "boom")
        let pv = await s2.getVector(id: "persist")?.vector
        XCTAssertEqual(pv, [3, 1, 4])
    }

    /// M27.2 — per-principal usage counters accumulate, stay keyed by
    /// principal, persist across reopen, and order by total tokens.
    func testUsageCountersAccumulatePerPrincipal() async throws {
        let url = tmpURL()
        defer { try? FileManager.default.removeItem(at: url) }
        do {
            let s = try AthenaStore(path: url)
            try await s.addUsage(
                principal: "u:alice", promptTokens: 10,
                completionTokens: 4)
            try await s.addUsage(
                principal: "u:alice", promptTokens: 6,
                completionTokens: 2)
            try await s.addUsage(
                principal: "u:bob", promptTokens: 1, completionTokens: 1)

            let a = await s.usage(principal: "u:alice")
            XCTAssertEqual(a?.requests, 2)
            XCTAssertEqual(a?.promptTokens, 16)
            XCTAssertEqual(a?.completionTokens, 6)
            XCTAssertEqual(a?.totalTokens, 22)
            // bob's request must NOT have leaked into alice's row.
            let b = await s.usage(principal: "u:bob")
            XCTAssertEqual(b?.requests, 1)
            XCTAssertEqual(b?.totalTokens, 2)
            let nobody = await s.usage(principal: "u:nobody")
            XCTAssertNil(nobody)
        }
        // Persist across reopen; allUsage orders by total desc.
        let s2 = try AthenaStore(path: url)
        let all = await s2.allUsage()
        XCTAssertEqual(all.map(\.principal), ["u:alice", "u:bob"])
        XCTAssertEqual(all.first?.totalTokens, 22)
    }

    func testAuditLogAppendFilterAndPersist() async throws {
        let url = tmpURL()
        defer { try? FileManager.default.removeItem(at: url) }
        do {
            let s = try AthenaStore(path: url)
            try await s.addAudit(
                principal: "u:admin", action: "user.create",
                target: "alice", result: "ok", detail: "role=member")
            try await s.addAudit(
                principal: "u:admin", action: "user.delete",
                target: "admin", result: "denied",
                detail: "only admin")
            try await s.addAudit(
                principal: "u:bob", action: "model.remove",
                target: "m1", result: "ok", detail: nil)

            let count = await s.auditCount()
            XCTAssertEqual(count, 3)
            // Most-recent-first (descending id).
            let recent = await s.listAudit(limit: 10)
            XCTAssertEqual(recent.first?.action, "model.remove")
            XCTAssertEqual(recent.first?.target, "m1")
            XCTAssertNil(recent.first?.detail)
            // Filter by principal.
            let admin = await s.listAudit(principal: "u:admin")
            XCTAssertEqual(admin.count, 2)
            XCTAssertTrue(admin.allSatisfy { $0.principal == "u:admin" })
            // Filter by action.
            let deletes = await s.listAudit(action: "user.delete")
            XCTAssertEqual(deletes.count, 1)
            XCTAssertEqual(deletes.first?.result, "denied")
            XCTAssertEqual(deletes.first?.detail, "only admin")
        }
        // Append-only rows persist across reopen.
        let s2 = try AthenaStore(path: url)
        let persisted = await s2.auditCount()
        XCTAssertEqual(persisted, 3)
    }

    func testAuditPruneByAge() async throws {
        let url = tmpURL()
        defer { try? FileManager.default.removeItem(at: url) }
        let s = try AthenaStore(path: url)
        for i in 0..<3 {
            try await s.addAudit(
                principal: "u:admin", action: "user.create",
                target: "u\(i)", result: "ok", detail: nil)
        }
        let now = Date().timeIntervalSince1970
        // A cutoff well in the past removes nothing (all rows are ~now).
        let none = try await s.pruneAudit(olderThan: now - 10_000)
        XCTAssertEqual(none, 0)
        let stillThere = await s.auditCount()
        XCTAssertEqual(stillThere, 3)
        // A cutoff in the future removes everything.
        let all = try await s.pruneAudit(olderThan: now + 10_000)
        XCTAssertEqual(all, 3)
        let empty = await s.auditCount()
        XCTAssertEqual(empty, 0)
    }

    func testPruneJobsByAgeSkipsPending() async throws {
        let url = tmpURL()
        defer { try? FileManager.default.removeItem(at: url) }
        let s = try AthenaStore(path: url)
        // Two terminal jobs + one still-queued.
        try await s.insertJob(
            id: "d1", kind: "conversation", request: Data(), owner: nil)
        try await s.updateJob(
            id: "d1", status: "done", result: Data("r".utf8), error: nil)
        try await s.insertJob(
            id: "e1", kind: "conversation", request: Data(), owner: nil)
        try await s.updateJob(
            id: "e1", status: "error", result: nil, error: "boom")
        try await s.insertJob(
            id: "q1", kind: "conversation", request: Data(), owner: nil)
        let now = Date().timeIntervalSince1970
        // Past cutoff: nothing is old enough yet.
        let none = try await s.pruneJobs(olderThan: now - 10_000)
        XCTAssertEqual(none, 0)
        // Future cutoff: both terminal results go, the queued one stays.
        let removed = try await s.pruneJobs(olderThan: now + 10_000)
        XCTAssertEqual(removed, 2)
        let left = await s.listJobs().map { $0.id }
        XCTAssertEqual(left, ["q1"], "pending job is never pruned")
    }

    func testTrimJobsToMaxRowsOldestTerminalFirst() async throws {
        let url = tmpURL()
        defer { try? FileManager.default.removeItem(at: url) }
        let s = try AthenaStore(path: url)
        // Three terminal (done) jobs, completed in id order, + a queued one.
        for id in ["a", "b", "c"] {
            try await s.insertJob(
                id: id, kind: "conversation", request: Data(), owner: nil)
            try await s.updateJob(
                id: id, status: "done", result: Data(id.utf8), error: nil)
        }
        try await s.insertJob(
            id: "q", kind: "conversation", request: Data(), owner: nil)
        // Cap at 2 total rows ⇒ delete 2 oldest terminal (a, b);
        // c (terminal) + q (pending) remain.
        let trimmed = try await s.trimJobs(maxRows: 2)
        XCTAssertEqual(trimmed, 2)
        let left = await s.listJobs().map { $0.id }.sorted()
        XCTAssertEqual(left, ["c", "q"])
        // maxRows 0 disables; never deletes.
        let none = try await s.trimJobs(maxRows: 0)
        XCTAssertEqual(none, 0)
    }

    func testTrimJobsNeverDropsPendingBelowCap() async throws {
        let url = tmpURL()
        defer { try? FileManager.default.removeItem(at: url) }
        let s = try AthenaStore(path: url)
        // Two pending + one terminal; cap of 1 can only reclaim the
        // terminal one — pending work is never trimmed.
        try await s.insertJob(
            id: "p1", kind: "conversation", request: Data(), owner: nil)
        try await s.insertJob(
            id: "p2", kind: "conversation", request: Data(), owner: nil)
        try await s.insertJob(
            id: "t1", kind: "conversation", request: Data(), owner: nil)
        try await s.updateJob(
            id: "t1", status: "done", result: Data(), error: nil)
        let trimmed = try await s.trimJobs(maxRows: 1)
        XCTAssertEqual(trimmed, 1)
        let left = await s.listJobs().map { $0.id }.sorted()
        XCTAssertEqual(left, ["p1", "p2"], "stays above cap, keeps pending")
    }

    // MARK: At-rest encryption (M34.3b)

    func testEncryptedStoreRoundTripAndWrongKey() async throws {
        let url = tmpURL()
        defer {
            for ext in ["", "-wal", "-shm"] {
                try? FileManager.default.removeItem(
                    at: URL(fileURLWithPath: url.path + ext))
            }
        }
        let key = "0123456789abcdef0123456789abcdef"
        do {
            let s = try AthenaStore(path: url, key: key)
            try await s.putVector(id: "v", vector: [1, 2, 3], metadata: nil)
            await s.close()
        }
        // The on-disk file must NOT be a plaintext SQLite database.
        XCTAssertFalse(
            AthenaStore.isPlaintextDatabase(at: url),
            "keyed store must be ciphertext on disk")
        // Correct key reopens and reads.
        do {
            let s = try AthenaStore(path: url, key: key)
            let got = await s.getVector(id: "v")?.vector
            XCTAssertEqual(got, [1, 2, 3])
            await s.close()
        }
        // Wrong key fails fast with an encryption error.
        XCTAssertThrowsError(try AthenaStore(path: url, key: "wrongkey")) {
            guard case AthenaStore.StoreError.encryption = $0 else {
                return XCTFail("expected .encryption, got \($0)")
            }
        }
        // No key against an encrypted file also fails (can't read it).
        XCTAssertThrowsError(try AthenaStore(path: url))
    }

    func testPlaintextToEncryptedMigrationPreservesData() async throws {
        let url = tmpURL()
        defer {
            for ext in ["", "-wal", "-shm"] {
                try? FileManager.default.removeItem(
                    at: URL(fileURLWithPath: url.path + ext))
            }
        }
        // Seed a plaintext store, then close it so the file is free for
        // the migration swap (the daemon migrates BEFORE opening too).
        do {
            let s = try AthenaStore(path: url)
            try await s.putVector(id: "keep", vector: [7, 8], metadata: nil)
            await s.close()
        }
        XCTAssertTrue(
            AthenaStore.isPlaintextDatabase(at: url),
            "store starts plaintext")
        // Migrate in place.
        let key = "feedfacefeedfacefeedfacefeedface"
        try AthenaStore.migrateToEncrypted(at: url, key: key)
        XCTAssertFalse(
            AthenaStore.isPlaintextDatabase(at: url),
            "store is encrypted after migration")
        // Data survives, readable only with the key.
        let s = try AthenaStore(path: url, key: key)
        let got = await s.getVector(id: "keep")?.vector
        XCTAssertEqual(got, [7, 8])
    }

    func testPruneVectorsByAge() async throws {
        let url = tmpURL()
        defer { try? FileManager.default.removeItem(at: url) }
        let s = try AthenaStore(path: url)
        try await s.putVector(id: "a", vector: [1, 2, 3], metadata: nil)
        try await s.putVector(id: "b", vector: [4, 5, 6], metadata: nil)
        let now = Date().timeIntervalSince1970
        // Past cutoff: both were just written, nothing is old enough.
        let none = try await s.pruneVectors(olderThan: now - 10_000)
        XCTAssertEqual(none, 0)
        var cnt = await s.vectorCount()
        XCTAssertEqual(cnt, 2)
        // Future cutoff: both are "older", both go.
        let all = try await s.pruneVectors(olderThan: now + 10_000)
        XCTAssertEqual(all, 2)
        cnt = await s.vectorCount()
        XCTAssertEqual(cnt, 0)
    }

    func testClearJobRequestEmptiesPrompt() async throws {
        let url = tmpURL()
        defer { try? FileManager.default.removeItem(at: url) }
        let s = try AthenaStore(path: url)
        let req = Data("{\"prompt\":\"secret\"}".utf8)
        try await s.insertJob(
            id: "j", kind: "conversation", request: req, owner: nil)
        try await s.updateJob(
            id: "j", status: "done", result: Data("r".utf8), error: nil)
        try await s.clearJobRequest(id: "j")
        let after = await s.getJob(id: "j")
        XCTAssertEqual(after?.request, Data(), "prompt blob emptied")
        XCTAssertEqual(after?.result, Data("r".utf8), "result untouched")
        XCTAssertEqual(after?.status, "done")
    }
}
