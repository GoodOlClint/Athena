import AthenaCore
import Foundation
import XCTest

@testable import AthenaLLM

/// ModelStore path resolution — pure logic, no MLX, always runs in CI.
final class ModelStoreTests: XCTestCase {

    func testDefaultResolvesToExternalSSDModel() {
        let url = ModelStore().resolve(nil)
        XCTAssertEqual(
            url.path, "/Volumes/SB-XTM5/mlx-models/Qwen3.5-27B-4bit-mtp")
    }

    func testEmptyStringResolvesToDefault() {
        XCTAssertEqual(
            ModelStore().resolve(""),
            ModelStore().resolve(nil))
    }

    func testAbsolutePathUsedVerbatim() {
        let url = ModelStore().resolve("/tmp/some-model")
        XCTAssertEqual(url.path, "/tmp/some-model")
    }

    func testBareNameResolvedUnderStoreRoot() {
        let store = ModelStore(
            rootDirectory: URL(fileURLWithPath: "/models", isDirectory: true))
        XCTAssertEqual(store.resolve("Qwen3.6-27B-8bit-mtp").path,
                       "/models/Qwen3.6-27B-8bit-mtp")
    }
}

/// Governor admission estimate = on-disk safetensors footprint. Builds a
/// fake model dir so this runs in CI without a multi-GB model.
final class MLXLLMModuleEstimateTests: XCTestCase {

    func testEstimateSumsOnlySafetensors() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("athena-est-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        try Data(count: 1_000).write(
            to: dir.appendingPathComponent("model-00001.safetensors"))
        try Data(count: 2_500).write(
            to: dir.appendingPathComponent("model-00002.safetensors"))
        try Data(count: 9_999).write(
            to: dir.appendingPathComponent("tokenizer.json"))

        XCTAssertEqual(
            MLXLLMModule.estimateBytes(forModelAt: dir), 3_500)
    }

    func testEstimateMissingDirIsZero() {
        let missing = URL(
            fileURLWithPath: "/nonexistent/athena/\(UUID().uuidString)")
        XCTAssertEqual(MLXLLMModule.estimateBytes(forModelAt: missing), 0)
    }
}

/// Real end-to-end generation through the governor. Gated: loading a 27B
/// model is far too heavy for CI, so it runs only when the model is present
/// AND opted in via ATHENA_RUN_MODEL_TESTS=1.
final class MLXLLMGenerationIntegrationTests: XCTestCase {

    func testGovernedGenerationProducesText() async throws {
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

        let llm = MLXLLMModule(
            modelDirectory: modelURL,
            parameters: .init(maxTokens: 24, temperature: 0))
        let gov = MemoryGovernor(totalBudgetBytes: Int(64) << 30)
        await gov.register(llm, evictable: false)

        try await gov.ensureLoaded(.llm)
        let reserved = await gov.snapshot().reservedBytes
        XCTAssertGreaterThan(reserved, 0)

        var out = ""
        for await chunk in llm.generate(prompt: "Reply with exactly: ok") {
            out += chunk
        }
        XCTAssertFalse(
            out.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
    }
}
