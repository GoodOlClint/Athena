import XCTest

@testable import AthenaStructured

/// FFI bring-up: proves the Rust llguidance staticlib links and the C ABI
/// is callable from Swift (M53 replaced the outlines-core engine behind the
/// same ABI). The Guide/index behaviour itself is covered by the Rust
/// `cargo test` suite and the Swift wrapper tests.
final class StructuredShimTests: XCTestCase {
    func testShimLinksAndReportsVersion() {
        XCTAssertEqual(StructuredShim.version, "llguidance-1.7.5")
    }

    func testLastErrorEmptyWhenNoFailure() {
        XCTAssertEqual(StructuredShim.lastError(), "")
    }
}
