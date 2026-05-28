import AthenaCore
import Foundation
import MLXLMCommon
import XCTest

@testable import AthenaLLM

/// MTP speculative decoding eligibility (M40 lifted the temp==0 gate;
/// both greedy and sampling speculative paths are now in production).
/// Pure, CI-safe.
final class SpeculativeGateTests: XCTestCase {
    func testEligibleAtTempZero() {
        XCTAssertTrue(
            LLMGenerationParameters(temperature: 0, speculative: true)
                .speculativeEligible)
    }
    func testEligibleAtNonZeroTemp() {
        XCTAssertTrue(
            LLMGenerationParameters(temperature: 0.7, speculative: true)
                .speculativeEligible)
    }
    func testSpeculativeOffNeverEngages() {
        XCTAssertFalse(
            LLMGenerationParameters(temperature: 0, speculative: false)
                .speculativeEligible)
        XCTAssertFalse(
            LLMGenerationParameters(temperature: 0.7, speculative: false)
                .speculativeEligible)
    }
}

/// ModelStore path resolution — pure logic, no MLX, always runs in CI.
final class ModelStoreTests: XCTestCase {

    func testNilReferenceResolvesToDefaultModelUnderStoreRoot() {
        let store = ModelStore(
            rootDirectory: URL(fileURLWithPath: "/tmp/store"))
        XCTAssertEqual(
            store.resolve(nil).path,
            "/tmp/store/" + ModelStore.defaultModelName)
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

    /// `pull`'s HF-cache layout points each shard at ../../blobs/<sha>.
    /// The estimate must follow the symlink to the real blob size — not
    /// the ~tens-of-bytes link path — or the governor's pre-load OOM
    /// admission gate sees ~0 B for every pulled model.
    func testEstimateFollowsBlobSymlinks() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("athena-est-sym-\(UUID().uuidString)")
        let blobs = root.appendingPathComponent("blobs", isDirectory: true)
        let snap = root.appendingPathComponent(
            "snapshots/rev", isDirectory: true)
        try FileManager.default.createDirectory(
            at: blobs, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(
            at: snap, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        try Data(count: 4_000).write(
            to: blobs.appendingPathComponent("aaa"))
        try Data(count: 6_000).write(
            to: blobs.appendingPathComponent("bbb"))
        try FileManager.default.createSymbolicLink(
            atPath: snap.appendingPathComponent("model-00001.safetensors").path,
            withDestinationPath: "../../blobs/aaa")
        try FileManager.default.createSymbolicLink(
            atPath: snap.appendingPathComponent("model-00002.safetensors").path,
            withDestinationPath: "../../blobs/bbb")

        XCTAssertEqual(
            MLXLLMModule.estimateBytes(forModelAt: snap), 10_000)
    }

    /// The store entry itself is a symlink for pulled models
    /// (~/.athena/models/<name> → snapshot). The estimate must follow that
    /// ROOT symlink too — `contentsOfDirectory` won't traverse a symlinked
    /// root, which left the estimate at 0 B for every pulled model.
    func testEstimateFollowsRootSymlink() throws {
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("athena-est-root-\(UUID().uuidString)")
        let real = base.appendingPathComponent("real", isDirectory: true)
        try FileManager.default.createDirectory(
            at: real, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: base) }
        try Data(count: 7_000).write(
            to: real.appendingPathComponent("model-00001.safetensors"))
        let link = base.appendingPathComponent("entry")
        try FileManager.default.createSymbolicLink(at: link, withDestinationURL: real)

        XCTAssertEqual(
            MLXLLMModule.estimateBytes(forModelAt: link), 7_000)
    }
}

/// M24.1 chat role fidelity — the model input must carry the FULL
/// conversation (system/user/assistant/tool), not a user-only join. Pure
/// mapping logic, no MLX/model, always runs in CI.
final class ChatTurnMappingTests: XCTestCase {
    func testRolesMapToSubstrateRoles() {
        let turns = [
            ChatTurn(role: "system", content: "sys"),
            ChatTurn(role: "user", content: "u"),
            ChatTurn(role: "assistant", content: "a"),
            ChatTurn(role: "tool", content: "t"),
        ]
        let msgs = MLXLLMModule.chatMessages(turns)
        XCTAssertEqual(
            msgs.map(\.role),
            [.system, .user, .assistant, .tool])
        XCTAssertEqual(msgs.map(\.content), ["sys", "u", "a", "t"])
    }

    func testUnknownRoleFallsBackToUser() {
        let msgs = MLXLLMModule.chatMessages([
            ChatTurn(role: "function", content: "x")
        ])
        XCTAssertEqual(msgs.map(\.role), [.user])
    }

    /// The substrate requires at least one message; an empty turn list
    /// must not crash `UserInput(chat:)`.
    func testEmptyTurnsBecomeSingleEmptyUser() {
        let msgs = MLXLLMModule.chatMessages([])
        XCTAssertEqual(msgs.count, 1)
        XCTAssertEqual(msgs[0].role, .user)
        XCTAssertEqual(msgs[0].content, "")
    }

    /// The protocol's default bridge (used by the stub and any non-role
    /// conformer) flattens turns in order — system + prior turns included,
    /// not dropped.
    func testFlattenedPromptKeepsAllTurnsInOrder() {
        let turns = [
            ChatTurn(role: "system", content: "S"),
            ChatTurn(role: "user", content: "U"),
            ChatTurn(role: "assistant", content: "A"),
        ]
        XCTAssertEqual(turns.flattenedPrompt(), "S\nU\nA")
    }
}

/// M24.3 per-request max_tokens override — a positive value wins over the
/// loaded default; 0/negative is ignored so a bad override can't truncate
/// to nothing. Pure, CI-safe.
final class EffectiveMaxTokensTests: XCTestCase {
    func testPositiveOverrideWins() {
        XCTAssertEqual(
            MLXLLMModule.effectiveMaxTokens(4096, 1024), 4096)
    }
    func testNilFallsBackToDefault() {
        XCTAssertEqual(
            MLXLLMModule.effectiveMaxTokens(nil, 1024), 1024)
    }
    func testZeroOrNegativeIgnored() {
        XCTAssertEqual(MLXLLMModule.effectiveMaxTokens(0, 1024), 1024)
        XCTAssertEqual(MLXLLMModule.effectiveMaxTokens(-5, 1024), 1024)
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
        let reserved = await gov.snapshot().residentBytes
        XCTAssertGreaterThan(reserved, 0)

        var out = ""
        for await chunk in llm.generate(prompt: "Reply with exactly: ok") {
            out += chunk
        }
        XCTAssertFalse(
            out.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
    }

    /// Regression for F10: `athena pull` lands a model as a SYMLINK
    /// (~/.athena/models/<name> → HF snapshot). The substrate weight loader
    /// doesn't follow a symlinked root dir, so a pulled model loaded ZERO
    /// shards and failed with keyNotFound. Load via a symlink and assert it
    /// resolves + loads + generates. (Heavy — gated like the test above.)
    func testLoadsViaStoreSymlink() async throws {
        let env = ProcessInfo.processInfo.environment
        guard env["ATHENA_RUN_MODEL_TESTS"] == "1" else {
            throw XCTSkip("set ATHENA_RUN_MODEL_TESTS=1 to run (heavy)")
        }
        let realURL = ModelStore().resolve(env["ATHENA_TEST_MODEL"])
            .resolvingSymlinksInPath()
        guard
            FileManager.default.fileExists(
                atPath: realURL.appendingPathComponent("config.json").path)
        else {
            throw XCTSkip("model not present at \(realURL.path)")
        }
        // Mimic the pull layout: a symlink store entry → the real model dir.
        let link = FileManager.default.temporaryDirectory
            .appendingPathComponent("athena-symlink-\(UUID().uuidString)")
        try FileManager.default.createSymbolicLink(
            at: link, withDestinationURL: realURL)
        defer { try? FileManager.default.removeItem(at: link) }

        let llm = MLXLLMModule(
            modelDirectory: link,
            parameters: .init(maxTokens: 16, temperature: 0))
        let gov = MemoryGovernor(totalBudgetBytes: Int(96) << 30)
        await gov.register(llm, evictable: false)
        try await gov.ensureLoaded(.llm)  // would throw keyNotFound pre-fix

        var out = ""
        for await chunk in llm.generate(prompt: "Reply with exactly: ok") {
            out += chunk
        }
        XCTAssertFalse(
            out.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
            "model loaded via a symlinked store entry must generate")
    }
}
