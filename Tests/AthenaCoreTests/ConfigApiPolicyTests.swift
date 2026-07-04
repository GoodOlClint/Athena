import XCTest

import AthenaServerKit

/// ADR 037 — the `PUT /api/config` security deny-list. These keys govern
/// auth/TLS/encryption/data-dir/debugger posture; config takeover would be
/// daemon takeover (and loopback dev mode has no auth), so they stay
/// TOML-plus-sudo. Pure + unit-pinned (ADR 008/009).
final class ConfigApiPolicyTests: XCTestCase {
    func testDeniedKeysAreNotSettable() {
        for k in [
            "auth_keys_file", "tls_cert", "tls_key", "encrypt_store",
            "data_dir", "deny_debugger_attach",
        ] {
            XCTAssertFalse(
                ConfigApiPolicy.isSettable(k), "\(k) must be deny-listed")
            XCTAssertTrue(ConfigApiPolicy.deniedKeys.contains(k))
        }
    }

    func testOrdinaryKeysAreSettable() {
        for k in [
            "max_tokens", "temperature", "rate_limit", "log_level",
            "max_concurrency", "cold_load_wait_secs",
        ] {
            XCTAssertTrue(
                ConfigApiPolicy.isSettable(k), "\(k) must be settable via API")
        }
    }

    func testDeniedMessageNamesTheKeyAndTheRemedy() {
        let m = ConfigApiPolicy.deniedMessage("tls_key")
        XCTAssertTrue(m.contains("tls_key"))
        XCTAssertTrue(m.lowercased().contains("sudo"))
    }
}
