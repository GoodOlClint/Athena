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

    /// H5 (M66.6 / ADR 006) — the vectors `owner` column: round-trip via
    /// `allVectors`, and the optional owner filter on `getVector`/
    /// `deleteVector` (a NULL/legacy row never matches a scoped owner).
    func testVectorOwnerColumnH5() async throws {
        let url = tmpURL()
        defer { try? FileManager.default.removeItem(at: url) }
        let s = try AthenaStore(path: url)
        try await s.putVector(
            id: "av", vector: [1, 0], metadata: nil, owner: "u:alice")
        try await s.putVector(
            id: "legacy", vector: [0, 1], metadata: nil)  // NULL owner

        let owners = Dictionary(
            uniqueKeysWithValues:
                await s.allVectors().map { ($0.id, $0.owner) })
        XCTAssertEqual(owners["av"], "u:alice")
        XCTAssertNil(owners["legacy"] ?? nil)

        // Owner filter: alice sees her row; the NULL row never matches a
        // scoped owner; unfiltered (nil) sees both.
        let avAlice = await s.getVector(id: "av", owner: "u:alice")
        XCTAssertNotNil(avAlice)
        let avBob = await s.getVector(id: "av", owner: "u:bob")
        XCTAssertNil(avBob)
        let legacyScoped = await s.getVector(id: "legacy", owner: "u:alice")
        XCTAssertNil(legacyScoped)
        let legacyUnfiltered = await s.getVector(id: "legacy")
        XCTAssertNotNil(legacyUnfiltered)

        // Scoped delete confines to the owner; the NULL row resists a
        // scoped delete but yields to an unscoped (admin) one.
        let crossDel = await s.deleteVector(id: "av", owner: "u:bob")
        XCTAssertFalse(crossDel)
        let legacyScopedDel =
            await s.deleteVector(id: "legacy", owner: "u:alice")
        XCTAssertFalse(legacyScopedDel)
        let ownDel = await s.deleteVector(id: "av", owner: "u:alice")
        XCTAssertTrue(ownDel)
        let legacyAdminDel = await s.deleteVector(id: "legacy")
        XCTAssertTrue(legacyAdminDel)
        let cnt2 = await s.vectorCount()
        XCTAssertEqual(cnt2, 0)
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

    /// H6 (M65.6) — the optional store-layer owner filter on
    /// getJob/listJobs/allUsage. nil keeps the legacy unfiltered behavior;
    /// a non-nil owner confines the result, and an ownerless (NULL) row
    /// never matches a scoped query (so a tenant can't read pre-auth jobs).
    func testOwnerScopedQueriesH6() async throws {
        let url = tmpURL()
        defer { try? FileManager.default.removeItem(at: url) }
        let s = try AthenaStore(path: url)

        try await s.insertJob(
            id: "a", kind: "conversation", request: Data(),
            owner: "u:alice")
        try await s.insertJob(
            id: "b", kind: "conversation", request: Data(),
            owner: "u:bob")
        try await s.insertJob(
            id: "legacy", kind: "conversation", request: Data(),
            owner: nil)

        // getJob: scoped to the owner; cross-owner and ownerless both miss.
        let aScoped = await s.getJob(id: "a", owner: "u:alice")
        XCTAssertEqual(aScoped?.id, "a")
        let aWrong = await s.getJob(id: "a", owner: "u:bob")
        XCTAssertNil(aWrong)
        let legacyScoped = await s.getJob(id: "legacy", owner: "u:alice")
        XCTAssertNil(legacyScoped)
        // nil = unfiltered: the ownerless row is reachable (admin path).
        let legacyUnfiltered = await s.getJob(id: "legacy")
        XCTAssertEqual(legacyUnfiltered?.id, "legacy")

        // listJobs: owner filter confines; nil sees all three.
        let aliceList = await s.listJobs(owner: "u:alice").map(\.id)
        XCTAssertEqual(aliceList, ["a"])
        let allCount = await s.listJobs().count
        XCTAssertEqual(allCount, 3)
        // status + owner compose.
        let bobQueued =
            await s.listJobs(status: "queued", owner: "u:bob").map(\.id)
        XCTAssertEqual(bobQueued, ["b"])
        // a scoped query never returns the NULL-owner row.
        let nobody = await s.listJobs(owner: "u:nobody")
        XCTAssertTrue(nobody.isEmpty)

        // allUsage: principal filter scopes to one row; nil = full table.
        try await s.addUsage(
            principal: "u:alice", promptTokens: 5, completionTokens: 1)
        try await s.addUsage(
            principal: "u:bob", promptTokens: 2, completionTokens: 1)
        let aliceUsage =
            await s.allUsage(principal: "u:alice").map(\.principal)
        XCTAssertEqual(aliceUsage, ["u:alice"])
        let allUsageCount = await s.allUsage().count
        XCTAssertEqual(allUsageCount, 2)
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
        let admins = try await s.usersWithRole("admin")
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
        let delAlice = try await s.deleteUser(username: "alice")
        XCTAssertTrue(delAlice)
        let aliceRoles = await s.rolesForUser(username: "alice")
        XCTAssertEqual(aliceRoles, [])
        let aliceTok = await s.tokenPrincipal(
            hash: Data(repeating: 1, count: 32))
        XCTAssertNil(aliceTok)
        let tokCount = await s.tokenCount()
        XCTAssertEqual(tokCount, 0)
        let adminsLeft = try await s.usersWithRole("admin")
        XCTAssertEqual(adminsLeft, ["bob"])
        let delMissing = try await s.deleteUser(username: "ghost")
        XCTAssertFalse(delMissing)
    }

    /// H12 (M66.2) — `listTokens` exposes only a 12-hex display prefix
    /// (never the full SHA-256 digest), while `tokensMatchingHashPrefix`
    /// returns the full hash for the rm/rotate paths and matches a
    /// case-insensitive hex prefix.
    func testTokenHashPrefixMatchingH12() async throws {
        let url = tmpURL()
        defer { try? FileManager.default.removeItem(at: url) }
        let s = try AthenaStore(path: url)
        try await s.putUser(
            username: "u", salt: Data([1]), hash: Data([2]), iters: 1000)
        let h1 = Data(repeating: 0xAB, count: 32)  // hex "abab…"
        let h2 = Data(repeating: 0xCD, count: 32)  // hex "cdcd…"
        try await s.putToken(
            hash: h1, username: "u", scopedRoles: nil, label: "one")
        try await s.putToken(
            hash: h2, username: "u", scopedRoles: ["member"], label: "two")

        // Display listing: 12-hex prefix only, no full digest leaked.
        let list = await s.listTokens()
        XCTAssertEqual(list.count, 2)
        XCTAssertTrue(list.allSatisfy { $0.hashPrefix.count == 12 })
        XCTAssertTrue(list.contains { $0.hashPrefix == "abababababab" })

        // Matching path: full hash + case-insensitive prefix match.
        let m = await s.tokensMatchingHashPrefix("ABABAB")
        XCTAssertEqual(m.count, 1)
        XCTAssertEqual(m.first?.hex.count, 64)
        XCTAssertEqual(m.first?.label, "one")
        XCTAssertTrue(m.first?.hex.hasPrefix("abab") ?? false)
        let none = await s.tokensMatchingHashPrefix("ffff")
        XCTAssertTrue(none.isEmpty)
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
        // H13: the cap governs RETAINED RESULTS (terminal rows), not total
        // rows. 3 terminal, cap 2 ⇒ delete 1 oldest terminal (a); b, c
        // (terminal) + q (pending, never counted) remain.
        let trimmed = try await s.trimJobs(maxRows: 2)
        XCTAssertEqual(trimmed, 1)
        let left = await s.listJobs().map { $0.id }.sorted()
        XCTAssertEqual(left, ["b", "c", "q"])
        // maxRows 0 disables; never deletes.
        let none = try await s.trimJobs(maxRows: 0)
        XCTAssertEqual(none, 0)
    }

    func testTrimJobsNeverDropsPendingBelowCap() async throws {
        let url = tmpURL()
        defer { try? FileManager.default.removeItem(at: url) }
        let s = try AthenaStore(path: url)
        // H13: a large pending backlog must NOT cause terminal results to
        // be over-pruned. Two pending + one terminal, cap 1: the single
        // terminal result is within the 1-result budget, so nothing is
        // trimmed — pending rows don't count toward (or trigger) the cap.
        try await s.insertJob(
            id: "p1", kind: "conversation", request: Data(), owner: nil)
        try await s.insertJob(
            id: "p2", kind: "conversation", request: Data(), owner: nil)
        try await s.insertJob(
            id: "t1", kind: "conversation", request: Data(), owner: nil)
        try await s.updateJob(
            id: "t1", status: "done", result: Data(), error: nil)
        let trimmed = try await s.trimJobs(maxRows: 1)
        XCTAssertEqual(trimmed, 0, "terminal count within cap ⇒ no prune")
        let left = await s.listJobs().map { $0.id }.sorted()
        XCTAssertEqual(left, ["p1", "p2", "t1"])
        // Add a second terminal result: now 2 terminal > cap 1 ⇒ drop the
        // oldest terminal (t1), keep t2 + both pending.
        try await s.insertJob(
            id: "t2", kind: "conversation", request: Data(), owner: nil)
        try await s.updateJob(
            id: "t2", status: "done", result: Data(), error: nil)
        let trimmed2 = try await s.trimJobs(maxRows: 1)
        XCTAssertEqual(trimmed2, 1)
        let left2 = await s.listJobs().map { $0.id }.sorted()
        XCTAssertEqual(left2, ["p1", "p2", "t2"])
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

    /// NH1 (M66.1) — `recoverInterruptedMigration` finishes or rolls back
    /// a migration crash mid-swap. Exercised at the file level (the swap is
    /// pure file moves); no SQLCipher needed.
    func testRecoverInterruptedMigrationNH1() async throws {
        let fm = FileManager.default
        // State 1: plaintext moved aside, encrypted copy not yet in place
        // (path missing, enc-migrate present) ⇒ complete the swap.
        do {
            let url = tmpURL()
            let enc = url.appendingPathExtension("enc-migrate")
            let bak = url.appendingPathExtension("migrate-bak")
            defer {
                for u in [url, enc, bak] { try? fm.removeItem(at: u) }
            }
            try Data("ENCRYPTED".utf8).write(to: enc)
            try Data("OLDPLAINTEXT".utf8).write(to: bak)
            XCTAssertFalse(fm.fileExists(atPath: url.path))
            try AthenaStore.recoverInterruptedMigration(at: url)
            XCTAssertEqual(
                try String(contentsOf: url, encoding: .utf8), "ENCRYPTED",
                "encrypted copy completes the swap into place")
            XCTAssertFalse(
                fm.fileExists(atPath: enc.path), "enc-migrate consumed")
            XCTAssertFalse(
                fm.fileExists(atPath: bak.path), "backup dropped")
        }
        // State 2: a stale encrypted orphan with the real db still present
        // (crash before the swap began) ⇒ discard the orphan, keep the db.
        do {
            let url = tmpURL()
            let enc = url.appendingPathExtension("enc-migrate")
            defer {
                for u in [url, enc] { try? fm.removeItem(at: u) }
            }
            try Data("REALDB".utf8).write(to: url)
            try Data("STALE".utf8).write(to: enc)
            try AthenaStore.recoverInterruptedMigration(at: url)
            XCTAssertEqual(
                try String(contentsOf: url, encoding: .utf8), "REALDB",
                "live db is untouched")
            XCTAssertFalse(
                fm.fileExists(atPath: enc.path), "stale orphan removed")
        }
        // State 3: swap completed but the backup deletion didn't
        // (encrypted in place, .migrate-bak leftover) ⇒ drop the backup.
        do {
            let url = tmpURL()
            let bak = url.appendingPathExtension("migrate-bak")
            defer {
                for u in [url, bak] { try? fm.removeItem(at: u) }
            }
            try Data("ENCRYPTED".utf8).write(to: url)
            try Data("OLDPLAINTEXT".utf8).write(to: bak)
            try AthenaStore.recoverInterruptedMigration(at: url)
            XCTAssertEqual(
                try String(contentsOf: url, encoding: .utf8), "ENCRYPTED")
            XCTAssertFalse(
                fm.fileExists(atPath: bak.path), "orphan backup removed")
        }
    }

    /// H2 (M66.1) — the allowlist default swap is transactional and keeps
    /// EXACTLY one default per module across repeated re-points, and a
    /// rejected (unknown-id) re-point leaves the existing default intact.
    func testAllowlistDefaultSingleSwapH2() async throws {
        let url = tmpURL()
        defer { try? FileManager.default.removeItem(at: url) }
        let s = try AthenaStore(path: url)
        try await s.addModelAllowlist(
            module: "llm", id: "a", isDefault: true)
        try await s.addModelAllowlist(module: "llm", id: "b")
        try await s.addModelAllowlist(module: "llm", id: "c")
        func defaults() async -> [String] {
            await s.listModelAllowlist(module: "llm")
                .filter(\.isDefault).map(\.id)
        }
        let d0 = await defaults()
        XCTAssertEqual(d0, ["a"])
        try await s.setModelAllowlistDefault(module: "llm", id: "b")
        let d1 = await defaults()
        XCTAssertEqual(d1, ["b"], "exactly one default, swapped")
        try await s.setModelAllowlistDefault(module: "llm", id: "c")
        let d2 = await defaults()
        XCTAssertEqual(d2, ["c"])
        // A re-point to an unknown id is rejected and changes nothing.
        do {
            try await s.setModelAllowlistDefault(module: "llm", id: "zzz")
            XCTFail("expected a throw for an unknown default id")
        } catch {
            // expected
        }
        let d3 = await defaults()
        XCTAssertEqual(d3, ["c"], "rejected swap left default intact")
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
