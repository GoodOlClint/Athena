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

    // ADR 025 S2 — the jobs table (queue) was removed entirely, so its
    // lifecycle / cancel / FIFO-access tests are gone. Owner-scoping now
    // only applies to the surviving usage table (below).

    /// H6 (M65.6) — the optional store-layer principal filter on `allUsage`:
    /// nil = the full table; a non-nil principal confines the result.
    func testOwnerScopedUsageQueriesH6() async throws {
        let url = tmpURL()
        defer { try? FileManager.default.removeItem(at: url) }
        let s = try AthenaStore(path: url)

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

    func testPersistenceAcrossOpen() async throws {
        let url = tmpURL()
        defer { try? FileManager.default.removeItem(at: url) }
        do {
            let s = try AthenaStore(path: url)
            try await s.addUsage(
                principal: "u:e", promptTokens: 9, completionTokens: 4)
        }
        let s2 = try AthenaStore(path: url)  // reopen same file
        let row = await s2.usage(principal: "u:e")
        XCTAssertEqual(row?.promptTokens, 9)
        XCTAssertEqual(row?.completionTokens, 4)
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

    // ADR 025 S2 — the queue (jobs table) and its retention/trim helpers
    // were removed; their `testPruneJobsByAgeSkipsPending` /
    // `testTrimJobs…` coverage went with them. Audit retention is still
    // pinned by `testAuditPruneByAge`.

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
            try await s.addUsage(
                principal: "u:v", promptTokens: 7, completionTokens: 3)
            await s.close()
        }
        // The on-disk file must NOT be a plaintext SQLite database.
        XCTAssertFalse(
            AthenaStore.isPlaintextDatabase(at: url),
            "keyed store must be ciphertext on disk")
        // Correct key reopens and reads.
        do {
            let s = try AthenaStore(path: url, key: key)
            let got = await s.usage(principal: "u:v")?.promptTokens
            XCTAssertEqual(got, 7)
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
            try await s.addUsage(
                principal: "u:keep", promptTokens: 11, completionTokens: 2)
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
        let got = await s.usage(principal: "u:keep")?.promptTokens
        XCTAssertEqual(got, 11)
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

    // ADR 026 — the `model_allowlist` table is retired; its
    // `testAllowlistDefaultSingleSwapH2` coverage is gone with it. Selection
    // resolution + the ambiguity rule are unit-pinned in ModelSelectionTests.

}
