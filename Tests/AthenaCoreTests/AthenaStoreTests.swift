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
}
