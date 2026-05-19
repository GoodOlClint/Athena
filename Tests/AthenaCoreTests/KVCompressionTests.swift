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

    /// M21: `triattention` now resolves (its eviction path exists).
    func testTriattentionResolves() throws {
        XCTAssertEqual(
            try KVCompression.resolve(env: nil, toml: "triattention"),
            .triattention)
        XCTAssertEqual(
            try KVCompression.resolve(env: "TriAttention ", toml: "none"),
            .triattention)
    }

    func testGenerationMapping() {
        XCTAssertNil(KVCompression.none.generation.kvBits)
        XCTAssertEqual(KVCompression.none.generation.scheme, .uniform)
        XCTAssertEqual(KVCompression.turboquant.generation.kvBits, 4.0)
        XCTAssertEqual(
            KVCompression.turboquant.generation.scheme, .turboQuant)
        // TriAttention evicts, it does not quantize KV: no kvBits, the
        // plain (uniform) scheme — the eviction lives on `eviction`.
        XCTAssertNil(KVCompression.triattention.generation.kvBits)
        XCTAssertEqual(
            KVCompression.triattention.generation.scheme, .uniform)
    }

    /// The eviction seam is distinct from the quant tuple: non-nil only
    /// for `triattention`.
    func testEvictionAccessor() {
        XCTAssertNil(KVCompression.none.eviction)
        XCTAssertNil(KVCompression.turboquant.eviction)
        XCTAssertNotNil(KVCompression.triattention.eviction)
    }
}
