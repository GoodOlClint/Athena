import XCTest

@testable import AthenaLLM

/// The shared `kv_compression` knob: precedence (env > TOML > none),
/// fail-closed validation, and the codec→scheme mapping. Pure logic —
/// no MLX, always runs in CI. M20.2 owns this contract; M21 extends it
/// with the `triattention` case.
final class KVCompressionTests: XCTestCase {

    func testDefaultsToNoneWhenUnsetOrBlank() throws {
        XCTAssertEqual(try KVCompression.resolve(env: nil, toml: nil), .none)
        XCTAssertEqual(try KVCompression.resolve(env: "", toml: "  \n"), .none)
    }

    func testTomlUsedWhenNoEnv() throws {
        XCTAssertEqual(
            try KVCompression.resolve(env: nil, toml: "turboquant"),
            .turboquant)
    }

    func testEnvOverridesToml() throws {
        XCTAssertEqual(
            try KVCompression.resolve(env: "none", toml: "turboquant"),
            .none)
        XCTAssertEqual(
            try KVCompression.resolve(env: "turboquant", toml: "none"),
            .turboquant)
    }

    func testCaseInsensitiveAndTrimmed() throws {
        XCTAssertEqual(
            try KVCompression.resolve(env: "  TurboQuant\n", toml: nil),
            .turboquant)
    }

    func testUnknownValueFailsClosed() {
        XCTAssertThrowsError(
            try KVCompression.resolve(env: "lz4", toml: nil))
        XCTAssertThrowsError(
            try KVCompression.resolve(env: nil, toml: "bogus"))
    }

    /// `triattention` is a documented future value but its codec does
    /// not exist until M21 — selecting it now must fail closed, not
    /// silently degrade to `none`.
    func testTriattentionNotYetAvailableFailsClosed() {
        XCTAssertThrowsError(
            try KVCompression.resolve(env: nil, toml: "triattention"))
    }

    func testGenerationMapping() {
        XCTAssertNil(KVCompression.none.generation.kvBits)
        XCTAssertEqual(KVCompression.none.generation.scheme, .uniform)
        XCTAssertEqual(KVCompression.turboquant.generation.kvBits, 4.0)
        XCTAssertEqual(
            KVCompression.turboquant.generation.scheme, .turboQuant)
    }
}
