import XCTest

@testable import AthenaCore

/// Pure model-class classification (ADR 016 / M73). No MLX — the detector is
/// MLX-free so it always runs in CI (ADR 009). Pins the decision table that
/// routes `athena convert`: generative/vision load; embedding redirects.
final class ModelClassTests: XCTestCase {

    private func info(_ json: String) -> ModelConfigInfo {
        ModelConfigInfo.parse(configJSON: Data(json.utf8))
    }

    // MARK: decision table

    func testBertIsEmbeddingByTypeAlone() {
        // mxbai-embed-large-v1 — embedder-only `model_type`, no ST markers
        // needed. (The real failure was the LLM factory's `unsupportedModelType("bert")`.)
        let c = ModelClass.classify(
            info: info(#"{"model_type":"bert"}"#),
            hasSentenceTransformerMarkers: false)
        XCTAssertEqual(c, .embedding)
    }

    func testEmbeddingGemmaNeedsSTMarkersToOverrideGenerativeType() {
        // embeddinggemma-300m — `gemma3_text` is ALSO a generative arch, so the
        // sentence-transformers markers are what make it embedding. Without
        // them the same type is generative (an ordinary Gemma3 text LLM).
        let json = #"{"model_type":"gemma3_text"}"#
        XCTAssertEqual(
            ModelClass.classify(
                info: info(json), hasSentenceTransformerMarkers: true),
            .embedding)
        XCTAssertEqual(
            ModelClass.classify(
                info: info(json), hasSentenceTransformerMarkers: false),
            .generative)
    }

    func testVisionWinsFirst() {
        let c = ModelClass.classify(
            info: info(
                #"{"model_type":"gemma4","vision_config":{"x":1}}"#),
            hasSentenceTransformerMarkers: false)
        XCTAssertEqual(c, .vision)
    }

    func testGenerativeQwen35() {
        XCTAssertEqual(
            ModelClass.classify(
                info: info(#"{"model_type":"qwen3_5"}"#),
                hasSentenceTransformerMarkers: false),
            .generative)
    }

    func testUnknownWhenNoModelType() {
        XCTAssertEqual(
            ModelClass.classify(
                info: info(#"{}"#), hasSentenceTransformerMarkers: false),
            .unknown)
    }

    func testCaseInsensitiveEmbedderType() {
        XCTAssertEqual(
            ModelClass.classify(
                info: info(#"{"model_type":"DistilBERT"}"#),
                hasSentenceTransformerMarkers: false),
            .embedding)
    }

    // MARK: filesystem probe

    func testDetectFromDirectoryWithSTMarker() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("mc-test-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        try Data(#"{"model_type":"gemma3_text"}"#.utf8).write(
            to: dir.appendingPathComponent("config.json"))
        XCTAssertEqual(ModelClass.detect(in: dir), .generative)

        // Adding a sentence-transformers marker flips it to embedding.
        try Data("[]".utf8).write(
            to: dir.appendingPathComponent("modules.json"))
        XCTAssertTrue(ModelClass.hasSentenceTransformerMarkers(in: dir))
        XCTAssertEqual(ModelClass.detect(in: dir), .embedding)
    }

    // MARK: error mapping

    func testUnsupportedConvertClassIs400() {
        let e = AthenaError.unsupportedConvertClass(
            model: "x/y", detected: "embedding", guidance: "do z")
        XCTAssertEqual(e.httpStatus, 400)
        XCTAssertEqual(e.code, "unsupported_convert_class")
        XCTAssertEqual(e.type, "invalid_request_error")
        XCTAssertTrue(e.message.contains("x/y"))
        XCTAssertTrue(e.message.contains("do z"))
    }

    func testLooksLikeUnsupportedArchMatchesSubstrateStrings() {
        struct Sub: Error { let s: String; var localizedDescription: String { s } }
        XCTAssertTrue(
            AthenaError.looksLikeUnsupportedArch(
                Sub(s: #"unsupportedModelType("bert")"#)))
        XCTAssertTrue(
            AthenaError.looksLikeUnsupportedArch(
                Sub(s: #"keyNotFound(path: ["model","norm","weight"])"#)))
        XCTAssertFalse(
            AthenaError.looksLikeUnsupportedArch(Sub(s: "some other error")))
        // An AthenaError must never be re-classified by the string matcher.
        XCTAssertFalse(
            AthenaError.looksLikeUnsupportedArch(
                AthenaError.requestTimedOut(seconds: 1)))
    }
}
