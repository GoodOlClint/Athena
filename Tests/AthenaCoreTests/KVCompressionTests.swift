import AthenaCore  // NB4 (M70.1b): the KVCompression enum + resolve moved here.
import XCTest

@testable import AthenaLLM  // the MLX-coupled .kvScheme/.servesArch extension

/// The shared `kv_compression` knob: precedence (env > TOML > none),
/// normalization (case-folding and trimming), fail-closed validation, and the
/// `kvScheme` accessor the MLX-coupled extension exposes. Those are the
/// behaviours the tests below cover, not a running order — one test exercises
/// precedence and trimming together, so the categories do not map onto the
/// file top to bottom. Pure logic — no MLX, always runs in CI. M20.2 introduced
/// this contract; M21 added the `triattention` case (the M20 `turboquant` case
/// has since been retired).
///
/// **Not covered here: the eviction seam.** This header used to claim it, and
/// none of the seven tests below touches eviction — they are knob resolution
/// and one accessor. TriAttention's geometry and eviction behaviour are MLX
/// numerics and are out of reach of this tier (ADR 009: a control-flow tier,
/// not a numeric one). The mirror-image claim in ADR 009's NF3 row was struck
/// by #100; this is the same claim in the file it was made about (#101).
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
