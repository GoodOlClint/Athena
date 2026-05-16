import XCTest

@testable import AthenaStructured

/// M3.1 FFI bring-up: proves the Rust outlines-core staticlib links and
/// the C ABI is callable from Swift. The Guide/index behaviour itself is
/// covered by the Rust `cargo test` suite (and the Swift wrapper tests
/// land in M3.2).
final class StructuredShimTests: XCTestCase {
    func testShimLinksAndReportsVersion() {
        XCTAssertEqual(StructuredShim.version, "0.2.14")
    }

    func testLastErrorEmptyWhenNoFailure() {
        XCTAssertEqual(StructuredShim.lastError(), "")
    }
}
