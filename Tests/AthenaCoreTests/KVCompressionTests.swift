import AthenaCore  // NB4 (M70.1b): the KVCompression enum + resolve moved here.
import XCTest

@testable import AthenaLLM  // the MLX-coupled .kvScheme/.servesArch extension

/// The shared `kv_compression` knob: precedence (env > TOML > none),
/// fail-closed validation, and the eviction seam. Pure logic — no MLX,
/// always runs in CI. M20.2 introduced this contract; M21 added the
/// `triattention` case (the M20 `turboquant` case has since been retired).
final class KVCompressionTests: XCTestCase {

    func testDefaultsToNoneWhenUnsetOrBlank() throws {
        XCTAssertEqual(try KVCompression.resolve(env: nil, toml: nil), .none)
        XCTAssertEqual(try KVCompression.resolve(env: "", toml: "  \n"), .none)
    }

    func testTomlUsedWhenNoEnv() throws {
        XCTAssertEqual(
            try KVCompression.resolve(env: nil, toml: "triattention"),
            .triattention)
    }

    func testEnvOverridesToml() throws {
        XCTAssertEqual(
            try KVCompression.resolve(env: "none", toml: "triattention"),
            .none)
        XCTAssertEqual(
            try KVCompression.resolve(env: "triattention", toml: "none"),
            .triattention)
    }

    func testCaseInsensitiveAndTrimmed() throws {
        XCTAssertEqual(
            try KVCompression.resolve(env: "  TriAttention\n", toml: nil),
            .triattention)
    }

    func testUnknownValueFailsClosed() {
        XCTAssertThrowsError(
            try KVCompression.resolve(env: "lz4", toml: nil))
        XCTAssertThrowsError(
            try KVCompression.resolve(env: nil, toml: "bogus"))
    }

    /// M21: `triattention` now resolves (its eviction path exists).
    func testTriattentionResolves() throws {
        XCTAssertEqual(
            try KVCompression.resolve(env: nil, toml: "triattention"),
            .triattention)
        XCTAssertEqual(
            try KVCompression.resolve(env: "TriAttention ", toml: "none"),
            .triattention)
    }

    /// The substrate `kvScheme` seam is non-nil only for `triattention`.
    func testKVSchemeAccessor() {
        XCTAssertNil(KVCompression.none.kvScheme)
        XCTAssertEqual(KVCompression.triattention.kvScheme, "triattention")
    }
}
