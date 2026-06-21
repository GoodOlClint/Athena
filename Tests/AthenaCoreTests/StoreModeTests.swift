import XCTest

@testable import AthenaCore

/// ADR 025 S4 — the store-persistence decision (stateless loopback vs. an
/// on-disk `athena.sqlite`) is MLX-free and unit-pinned (ADR 008/009): the
/// rule is the slice's behavioral contract, so each branch is asserted.
final class StoreModeTests: XCTestCase {

    private func resolve(
        keys: Bool = false, dbExists: Bool = false, loopback: Bool = true,
        encrypt: Bool = false, override: Bool = false
    ) -> StorePersistence {
        StoreMode.resolve(
            hasBootstrapKeys: keys, dbFileExists: dbExists,
            isLoopback: loopback, encryptStore: encrypt,
            persistOverride: override)
    }

    func testLoopbackNoCredentialsIsEphemeral() {
        // The motivating case: a loopback dev daemon with nothing to
        // authenticate writes no file.
        XCTAssertEqual(resolve(), .ephemeral)
    }

    func testBootstrapKeysForcePersistent() {
        // File/env auth keys ⇒ auth is on ⇒ the DB is needed.
        XCTAssertEqual(resolve(keys: true), .persistent)
    }

    func testExistingDbFileForcesPersistent() {
        // A store already on disk (e.g. a prior authed run) is respected.
        XCTAssertEqual(resolve(dbExists: true), .persistent)
    }

    func testEncryptStoreForcesPersistent() {
        // Opting into at-rest encryption implies wanting a durable file.
        XCTAssertEqual(resolve(encrypt: true), .persistent)
    }

    func testNonLoopbackForcesPersistent() {
        // A non-loopback bind is auth-required (fail-closed elsewhere), so it
        // is never stateless.
        XCTAssertEqual(resolve(loopback: false), .persistent)
    }

    func testPersistOverrideWinsOverEphemeralConditions() {
        // The `persist_store` switch keeps audit/usage on disk even in the
        // otherwise-stateless loopback case.
        XCTAssertEqual(resolve(override: true), .persistent)
    }

    func testOverrideTrumpsEveryStatelessSignalTogether() {
        XCTAssertEqual(
            resolve(
                keys: false, dbExists: false, loopback: true,
                encrypt: false, override: true),
            .persistent)
    }

    func testIsLoopbackRecognizesTheStandardHosts() {
        for h in ["127.0.0.1", "::1", "localhost"] {
            XCTAssertTrue(StoreMode.isLoopback(h), h)
        }
        for h in ["0.0.0.0", "10.0.0.5", "example.com", "192.168.1.2"] {
            XCTAssertFalse(StoreMode.isLoopback(h), h)
        }
    }
}
