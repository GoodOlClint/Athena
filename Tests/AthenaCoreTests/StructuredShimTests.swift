import XCTest

@testable import AthenaStructured

/// FFI bring-up: proves the Rust llguidance staticlib links and the C ABI
/// is callable from Swift (M53 replaced the outlines-core engine behind the
/// same ABI). The Guide/index behaviour itself is covered by the Rust
/// `cargo test` suite and the Swift wrapper tests.
final class StructuredShimTests: XCTestCase {
    func testShimLinksAndReportsVersion() {
        // L11 (M70.3): assert the OBSERVABLE shape — the shim links and
        // self-reports a well-formed llguidance version — instead of pinning
        // an exact string that breaks on every (intended) engine bump while
        // proving nothing more. Catches a missing/garbage version and a
        // wrong-engine swap (via the prefix) without the brittle exact pin.
        let v = StructuredShim.version
        XCTAssertTrue(
            v.hasPrefix("llguidance-"),
            "expected the llguidance engine, got '\(v)'")
        let semver = v.dropFirst("llguidance-".count)
        let parts = semver.split(separator: ".")
        XCTAssertGreaterThanOrEqual(
            parts.count, 2, "version '\(v)' is not dotted major.minor[.patch]")
        XCTAssertTrue(
            parts.allSatisfy { Int($0) != nil },
            "version components must be numeric, got '\(semver)'")
    }

    func testLastErrorEmptyWhenNoFailure() {
        XCTAssertEqual(StructuredShim.lastError(), "")
    }
}
