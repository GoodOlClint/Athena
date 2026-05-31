import XCTest

@testable import AthenaCore

/// M54 — model-id resolution by store-dir identity, so a request can name
/// a model by either its full HuggingFace id (`org/name`) or its bare
/// store-dir name (the form `athena pull` creates) and resolve the same
/// allowlist row, regardless of which form the allowlist stores.
final class ModelAllowlistTests: XCTestCase {

    func testModelStoreIdentity() {
        XCTAssertEqual(
            "Qwen/Qwen3-Embedding-4B".modelStoreIdentity, "Qwen3-Embedding-4B")
        XCTAssertEqual(
            "Qwen3-Embedding-4B".modelStoreIdentity, "Qwen3-Embedding-4B")
        XCTAssertEqual("a/b/c".modelStoreIdentity, "c")
        XCTAssertEqual(
            "/abs/store/My-Model".modelStoreIdentity, "My-Model")
        XCTAssertEqual("".modelStoreIdentity, "")
    }

    func testEitherFormMatchesOrgAllowlist() {
        let allow = ["Qwen/Qwen3-Embedding-4B", "BAAI/bge-small-en-v1.5"]
        // The canonical (stored) form is returned, drives load + echo.
        XCTAssertEqual(
            allow.canonicalByStoreIdentity("Qwen/Qwen3-Embedding-4B"),
            "Qwen/Qwen3-Embedding-4B")
        XCTAssertEqual(
            allow.canonicalByStoreIdentity("Qwen3-Embedding-4B"),
            "Qwen/Qwen3-Embedding-4B")
        XCTAssertEqual(
            allow.canonicalByStoreIdentity("qwen3-embedding-4b"),
            "Qwen/Qwen3-Embedding-4B")
        XCTAssertNil(allow.canonicalByStoreIdentity("nope/other"))
    }

    func testEitherFormMatchesBareAllowlist() {
        let allow = ["Qwen3-Embedding-4B"]
        XCTAssertEqual(
            allow.canonicalByStoreIdentity("Qwen/Qwen3-Embedding-4B"),
            "Qwen3-Embedding-4B")
        XCTAssertEqual(
            allow.canonicalByStoreIdentity("Qwen3-Embedding-4B"),
            "Qwen3-Embedding-4B")
    }
}
