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
}
