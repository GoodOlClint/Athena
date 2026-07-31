import AthenaCore
import Foundation
import XCTest

@testable import AthenaLLM

/// M64.3 — model-on coherence for the Gemma4 MoE target (gemma-4-26b-a4b-it-4bit:
/// 128 experts, top-8, mixed 4/8-bit quant) through Athena's actual generation
/// path (`MLXLLMModule.generate` → `MemoryGovernor` → substrate stream), not just
/// a raw forward. Confirms the MoE arch loads under the governor and serves
/// coherent text, and that constrained (schema-guided) decoding works on it — the
/// behaviours the `.validated` tier advertises. Heavy: gated behind
/// ATHENA_RUN_MODEL_TESTS=1 (CI never runs it).
///
/// Model dir: ATHENA_TEST_GEMMA4_MOE_DIR, else the HF cache snapshot, else the
/// model store. The test asserts the loaded config is actually the MoE arch
/// (enable_moe_block) so it provably exercises the M64.1 expert path.
final class Gemma4MoEE2ETests: XCTestCase {

    private func skipUnlessEnabled() throws -> URL {
        let env = ProcessInfo.processInfo.environment
        guard env["ATHENA_RUN_MODEL_TESTS"] == "1" else {
            throw XCTSkip("set ATHENA_RUN_MODEL_TESTS=1 to run (heavy)")
        }
        if let p = env["ATHENA_TEST_GEMMA4_MOE_DIR"], !p.isEmpty {
            return URL(fileURLWithPath: p)
        }
        let home = FileManager.default.homeDirectoryForCurrentUser
        let snaps = home.appending(
            path:
                ".cache/huggingface/hub/models--mlx-community--gemma-4-26b-a4b-it-4bit/snapshots")
        if let entries = try? FileManager.default.contentsOfDirectory(
            at: snaps, includingPropertiesForKeys: nil),
            let first = entries.first(where: {
                FileManager.default.fileExists(
                    atPath: $0.appending(component: "config.json").path)
            })
        {
            return first
        }
        let store = home.appending(path: ".athena/models/gemma-4-26b-a4b-it-4bit")
        if FileManager.default.fileExists(
            atPath: store.appending(component: "config.json").path)
        {
            return store
        }
        throw XCTSkip("no gemma-4-26b-a4b-it-4bit checkpoint present")
    }

    /// Provably the MoE arch: enable_moe_block must be true, else the test would
    /// pass on a dense Gemma4 and not exercise the expert path.
    private func assertIsMoE(_ dir: URL) throws {
        let cfg = dir.appending(component: "config.json")
        let data = try Data(contentsOf: cfg)
        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        let text = (json?["text_config"] as? [String: Any]) ?? json
        guard text?["enable_moe_block"] as? Bool == true else {
            throw XCTSkip(
                "checkpoint is not a Gemma4 MoE target (enable_moe_block != true)")
        }
    }

    private func makeLLM(_ model: URL) async throws -> MLXLLMModule {
        let llm = MLXLLMModule(
            modelDirectory: model,
            parameters: .init(maxTokens: 64, temperature: 0))
        let gov = MemoryGovernor(totalBudgetBytes: Int(96) << 30)
        await gov.register(llm, evictable: false)
        try await gov.ensureLoaded(.llm)
        return llm
    }

    private func assertCoherent(_ text: String, _ label: String) {
        XCTAssertFalse(text.isEmpty, "\(label): empty output")
        XCTAssertGreaterThan(text.count, 3, "\(label): implausibly short: \(text)")
        let distinct = Set(text.replacingOccurrences(of: " ", with: ""))
        XCTAssertGreaterThan(
            distinct.count, 2, "\(label): degenerate output: \(text)")
    }

    /// Coherence: the MoE target serves a correct, non-degenerate answer.
    func testMoELoadsAndGeneratesCoherent() async throws {
        let model = try skipUnlessEnabled()
        try assertIsMoE(model)
        let llm = try await makeLLM(model)

        var out = ""
        for await chunk in llm.generate(
            prompt: "What is the capital of France? Answer in one word.")
        {
            out += chunk
        }
        let text = out.trimmingCharacters(in: .whitespacesAndNewlines)
        assertCoherent(text, "moe-plain")
        XCTAssertTrue(
            text.lowercased().contains("paris"),
            "moe-plain: expected 'Paris' in the answer, got: \(text)")
        print("Gemma4 MoE e2e coherence: \(text)")
    }

    /// Validated-tier capability: schema-guided decoding works on the MoE arch
    /// (the output parses as JSON with the required integer field).
    func testMoEStructuredOutputIsValidJSON() async throws {
        let model = try skipUnlessEnabled()
        try assertIsMoE(model)
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
            obj["answer"] as? Int, "structured: missing integer 'answer': \(text)")
        print("Gemma4 MoE e2e structured: \(text)")
    }
}
