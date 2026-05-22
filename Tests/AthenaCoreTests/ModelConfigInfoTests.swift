import XCTest

@testable import AthenaLLM

/// Pure config.json parsing + the generalized per-token KV estimate
/// (M23 fork C). No MLX — always runs in CI.
final class ModelConfigInfoTests: XCTestCase {

    private func info(_ json: String) -> ModelConfigInfo {
        ModelConfigInfo.parse(configJSON: Data(json.utf8))
    }

    func testFlatLlamaConfig() {
        let i = info(
            """
            {"model_type":"llama","vocab_size":128256,
             "num_hidden_layers":16,"num_attention_heads":32,
             "num_key_value_heads":8,"hidden_size":2048,"head_dim":64}
            """)
        XCTAssertEqual(i.modelType, "llama")
        XCTAssertEqual(i.vocabSize, 128256)
        XCTAssertEqual(i.numHiddenLayers, 16)
        XCTAssertEqual(i.effectiveKVHeads, 8)
        XCTAssertEqual(i.effectiveHeadDim, 64)
        // 2 (K+V) · 16 layers · 8 kv-heads · 64 head-dim · 2 bytes (fp16)
        XCTAssertEqual(i.perTokenKVBytes(bytesPerElement: 2), 32 * 1024)
    }

    /// The Qwen3.5-27B geometry the old constant was tuned for: the
    /// formula must reproduce exactly 256 KiB (proves it is a faithful
    /// generalization, not a behavior change for that model).
    func test27BGeometryReproduces256KiB() {
        let i = ModelConfigInfo(
            numHiddenLayers: 64, numAttentionHeads: 64,
            numKeyValueHeads: 8, headDim: 128)
        XCTAssertEqual(i.perTokenKVBytes(bytesPerElement: 2), 256 * 1024)
    }

    func testKVHeadsFallBackToAttentionHeads() {
        let i = info(
            """
            {"num_hidden_layers":4,"num_attention_heads":16,
             "hidden_size":1024,"head_dim":64}
            """)
        // No num_key_value_heads ⇒ multi-head: kv heads = attention heads.
        XCTAssertEqual(i.effectiveKVHeads, 16)
    }

    func testHeadDimDerivedFromHiddenSize() {
        let i = info(
            """
            {"num_hidden_layers":4,"num_attention_heads":16,
             "num_key_value_heads":4,"hidden_size":1024}
            """)
        // No head_dim ⇒ hidden_size / num_attention_heads = 1024/16 = 64.
        XCTAssertEqual(i.effectiveHeadDim, 64)
        XCTAssertEqual(i.perTokenKVBytes(bytesPerElement: 2), 2 * 4 * 4 * 64 * 2)
    }

    func testNestedTextConfig() {
        let i = info(
            """
            {"model_type":"gemma3","text_config":{
              "vocab_size":262144,"num_hidden_layers":26,
              "num_attention_heads":4,"num_key_value_heads":1,
              "head_dim":256}}
            """)
        XCTAssertEqual(i.modelType, "gemma3")
        XCTAssertEqual(i.vocabSize, 262144)
        XCTAssertEqual(i.numHiddenLayers, 26)
        XCTAssertEqual(i.effectiveKVHeads, 1)
        XCTAssertEqual(i.effectiveHeadDim, 256)
    }

    func testTopLevelWinsOverNested() {
        let i = info(
            """
            {"num_hidden_layers":10,"text_config":{"num_hidden_layers":99}}
            """)
        XCTAssertEqual(i.numHiddenLayers, 10)
    }

    func testMissingDimsYieldNilEstimate() {
        // Only vocab present — cannot size KV; caller keeps the constant.
        let i = info(#"{"vocab_size":50000}"#)
        XCTAssertEqual(i.vocabSize, 50000)
        XCTAssertNil(i.perTokenKVBytes(bytesPerElement: 2))
        XCTAssertNil(i.effectiveHeadDim)
    }

    func testGarbagePayloadIsAllNil() {
        let i = info("not json at all {{{")
        XCTAssertNil(i.modelType)
        XCTAssertNil(i.vocabSize)
        XCTAssertNil(i.perTokenKVBytes(bytesPerElement: 2))
    }

    func testZeroDimsRejected() {
        let i = ModelConfigInfo(
            numHiddenLayers: 0, numAttentionHeads: 8,
            numKeyValueHeads: 8, headDim: 64)
        XCTAssertNil(i.perTokenKVBytes(bytesPerElement: 2))
    }
}
