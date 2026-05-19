import AthenaCore
import Foundation
import XCTest

@testable import AthenaLLM

/// M20.4 — model-on numeric/quality sanity for the TurboQuant KV-cache
/// codec. Heavy: loads a real MTP model, so gated behind
/// ATHENA_RUN_MODEL_TESTS=1 (CI never runs it). Validates that
/// `kv_compression = turboquant` (keys=Prod / values=MSE) produces
/// coherent, non-degenerate greedy output and does not crash the
/// governed path — i.e. the deliberate keys=Prod choice is sane.
final class TurboQuantE2ETests: XCTestCase {

    private func skipUnlessEnabled() throws -> URL {
        let env = ProcessInfo.processInfo.environment
        guard env["ATHENA_RUN_MODEL_TESTS"] == "1" else {
            throw XCTSkip("set ATHENA_RUN_MODEL_TESTS=1 to run (heavy)")
        }
        let modelURL = ModelStore().resolve(env["ATHENA_TEST_MODEL"])
        guard
            FileManager.default.fileExists(
                atPath: modelURL.appendingPathComponent("config.json").path)
        else {
            throw XCTSkip("model not present at \(modelURL.path)")
        }
        return modelURL
    }

    private func generate(
        model: URL, kv: KVCompression
    ) async throws -> String {
        let llm = MLXLLMModule(
            modelDirectory: model,
            parameters: .init(
                maxTokens: 48, temperature: 0, kvCompression: kv))
        let gov = MemoryGovernor(totalBudgetBytes: Int(64) << 30)
        await gov.register(llm, evictable: false)
        try await gov.ensureLoaded(.llm)
        var out = ""
        for await chunk in llm.generate(
            prompt: "List three primary colors, comma separated.")
        {
            out += chunk
        }
        return out.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Output must be non-empty and not degenerate (a single character
    /// or token repeated is the classic signature of a broken KV codec
    /// — e.g. wrong rotation transpose or biased keys).
    private func assertCoherent(_ text: String, _ label: String) {
        XCTAssertFalse(text.isEmpty, "\(label): empty output")
        XCTAssertGreaterThan(
            text.count, 3, "\(label): implausibly short output")
        let distinct = Set(text.replacingOccurrences(of: " ", with: ""))
        XCTAssertGreaterThan(
            distinct.count, 2,
            "\(label): degenerate output (\(distinct.count) distinct "
                + "chars): \(text)")
    }

    func testTurboQuantProducesCoherentOutput() async throws {
        let model = try skipUnlessEnabled()
        let turbo = try await generate(model: model, kv: .turboquant)
        assertCoherent(turbo, "turboquant")
    }

    /// Sanity A/B: the uncompressed path and the TurboQuant path both
    /// produce coherent text. Outputs are NOT required to be identical
    /// — TurboQuant changes cache numerics by design — only that
    /// enabling it does not collapse generation.
    func testUncompressedVsTurboQuantBothCoherent() async throws {
        let model = try skipUnlessEnabled()
        let plain = try await generate(model: model, kv: .none)
        assertCoherent(plain, "none")
        let turbo = try await generate(model: model, kv: .turboquant)
        assertCoherent(turbo, "turboquant")
    }
}
