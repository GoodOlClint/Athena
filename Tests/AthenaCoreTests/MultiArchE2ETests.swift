import AthenaCore
import Foundation
import XCTest

@testable import AthenaLLM

/// M23 — model-on validation that a NON-Qwen architecture (Llama, Gemma,
/// Phi, …) loads through the substrate factory and generates, and that
/// structured output is genuinely schema-guided on the substrate path
/// (fork A), not silently dropped. Heavy: gated behind
/// ATHENA_RUN_MODEL_TESTS=1 (CI never runs it).
///
/// Model: `ATHENA_TEST_MODEL_NONQWEN` (default
/// `Llama-3.2-1B-Instruct-4bit`) under the model store. The test asserts
/// the model is actually non-Qwen3.5, so it provably exercises the
/// substrate path and not the vendored Qwen3.5 model.
final class MultiArchE2ETests: XCTestCase {

    private func skipUnlessEnabled() throws -> URL {
        let env = ProcessInfo.processInfo.environment
        guard env["ATHENA_RUN_MODEL_TESTS"] == "1" else {
            throw XCTSkip("set ATHENA_RUN_MODEL_TESTS=1 to run (heavy)")
        }
        let name =
            env["ATHENA_TEST_MODEL_NONQWEN"] ?? "Llama-3.2-1B-Instruct-4bit"
        let modelURL = ModelStore().resolve(name)
        guard
            FileManager.default.fileExists(
                atPath: modelURL.appendingPathComponent("config.json").path)
        else {
            throw XCTSkip("model not present at \(modelURL.path)")
        }
        // Provably the substrate path: the test is meaningless on Qwen3.5.
        let info = ModelConfigInfo.read(modelDirectory: modelURL)
        if SupportedModels.isQwen35(info?.modelType) {
            throw XCTSkip(
                "ATHENA_TEST_MODEL_NONQWEN is a Qwen3.5 model "
                    + "(\(info?.modelType ?? "?")) — needs a non-Qwen arch")
        }
        return modelURL
    }

    private func makeLLM(
        _ model: URL, kv: KVCompression = .none
    ) async throws -> MLXLLMModule {
        let llm = MLXLLMModule(
            modelDirectory: model,
            parameters: .init(
                maxTokens: 64, temperature: 0, kvCompression: kv))
        let gov = MemoryGovernor(totalBudgetBytes: Int(64) << 30)
        await gov.register(llm, evictable: false)
        try await gov.ensureLoaded(.llm)
        return llm
    }

    private func assertCoherent(_ text: String, _ label: String) {
        XCTAssertFalse(text.isEmpty, "\(label): empty output")
        XCTAssertGreaterThan(
            text.count, 3, "\(label): implausibly short output")
        let distinct = Set(text.replacingOccurrences(of: " ", with: ""))
        XCTAssertGreaterThan(
            distinct.count, 2,
            "\(label): degenerate output: \(text)")
    }

    /// Baseline: a non-Qwen arch loads + does plain generation (the M23.1
    /// finding — non-Qwen falls through to the substrate stream).
    func testNonQwenLoadsAndGeneratesPlain() async throws {
        let model = try skipUnlessEnabled()
        let llm = try await makeLLM(model)
        var out = ""
        for await chunk in llm.generate(
            prompt: "List three primary colors, comma separated.")
        {
            out += chunk
        }
        assertCoherent(
            out.trimmingCharacters(in: .whitespacesAndNewlines),
            "plain")
    }

    /// Fork A: structured output is schema-guided on the substrate path —
    /// the result must parse as JSON with the required integer field
    /// (before M23 this silently produced unconstrained prose).
    func testNonQwenStructuredOutputIsValidJSON() async throws {
        let model = try skipUnlessEnabled()
        let llm = try await makeLLM(model)
        let schema = """
            {"type":"object","properties":{"answer":{"type":"integer"}},
             "required":["answer"],"additionalProperties":false}
            """
        let out = await llm.generatedText(
            prompt: "What is 2 + 2? Reply as JSON.", schemaJSON: schema)
        let text = out.trimmingCharacters(in: .whitespacesAndNewlines)
        XCTAssertFalse(text.isEmpty, "structured: empty output")
        guard let data = text.data(using: .utf8),
            let obj = try? JSONSerialization.jsonObject(with: data)
                as? [String: Any]
        else {
            return XCTFail("structured: not valid JSON object: \(text)")
        }
        XCTAssertNotNil(
            obj["answer"] as? Int,
            "structured: missing integer 'answer': \(text)")
    }

    /// Fork B (model-on): selecting `triattention` on a non-Qwen arch is
    /// inert — generation still succeeds (runs uncompressed), it does not
    /// crash or degenerate.
    func testTriattentionInertOnNonQwenStillGenerates() async throws {
        let model = try skipUnlessEnabled()
        let llm = try await makeLLM(model, kv: .triattention)
        var out = ""
        for await chunk in llm.generate(
            prompt: "Name one ocean.")
        {
            out += chunk
        }
        assertCoherent(
            out.trimmingCharacters(in: .whitespacesAndNewlines),
            "triattention-inert")
    }
}
