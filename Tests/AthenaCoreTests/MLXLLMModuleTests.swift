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

/// ADR 032 S4 — the pure MTP acceptance-rate helper (the substrate `.info`
/// aggregate → an operability rate). CI-safe.
final class MTPAcceptanceRateTests: XCTestCase {
    func testNilWhenDrafterDidNotRun() {
        // No proposed tokens ⇒ the plain non-speculative path; nothing to report.
        XCTAssertNil(
            MLXLLMModule.mtpAcceptanceRate(proposed: nil, accepted: nil))
        XCTAssertNil(
            MLXLLMModule.mtpAcceptanceRate(proposed: 0, accepted: 0))
    }
    func testRate() {
        XCTAssertEqual(
            MLXLLMModule.mtpAcceptanceRate(proposed: 4, accepted: 3) ?? -1,
            0.75, accuracy: 1e-9)
        // accepted defaults to 0 when absent.
        XCTAssertEqual(
            MLXLLMModule.mtpAcceptanceRate(proposed: 8, accepted: nil) ?? -1,
            0.0, accuracy: 1e-9)
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
        XCTAssertEqual(
            store.resolve("Qwen3.6-27B-8bit-mtp").path,
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

/// `max_prompt_tokens` prefill guard (ADR 009 decision seam): a positive cap
/// refuses prompts strictly above it; nil/non-positive ⇒ unbounded so the
/// legacy "no cap" behavior is preserved. Pure, CI-safe.
final class PromptExceedsCapTests: XCTestCase {
    func testUnboundedWhenCapNil() {
        XCTAssertFalse(MLXLLMModule.promptExceedsCap(1_000_000, cap: nil))
    }
    func testUnboundedWhenCapNonPositive() {
        XCTAssertFalse(MLXLLMModule.promptExceedsCap(50, cap: 0))
        XCTAssertFalse(MLXLLMModule.promptExceedsCap(50, cap: -1))
    }
    func testOverCapExceeds() {
        XCTAssertTrue(MLXLLMModule.promptExceedsCap(32_769, cap: 32_768))
    }
    func testAtCapAllowed() {
        XCTAssertFalse(MLXLLMModule.promptExceedsCap(32_768, cap: 32_768))
    }
}

/// Stub LLM model selection + governor-accounting discipline. Pure,
/// CI-safe (no MLX / model).
final class StubLLMModuleSelectionTests: XCTestCase {

    /// NE5: a full HF org/name id resolves to the bare store-dir name
    /// (uniform with the embedding/audio modules) instead of 400ing —
    /// pre-fix the LLM path used a full-string case-insensitive match.
    func testFullHFIdResolvesToBareStoreName() async throws {
        let m = StubLLMModule(
            modelIds: ["Qwen3.5-2B-4bit", "Phi-3.5-mini-instruct-4bit"])
        try await m.rebind(to: "mlx-community/Qwen3.5-2B-4bit")
        let r = await m.residentModelId()
        XCTAssertEqual(r, "Qwen3.5-2B-4bit")
    }

    /// ADR 026: load() with >1 selectable model and NO configured default is
    /// ambiguous — it must throw `ambiguous_model` and bind nothing, so the
    /// governor never reserves a slot for an un-decidable default (the NC13
    /// "don't bind a stale default" intent, restated for store-backed
    /// selection).
    func testAmbiguousLoadReservesNothing() async throws {
        let m = StubLLMModule(modelIds: ["a", "b"])  // no configuredDefault
        do {
            try await m.load(
                reservation: MemoryReservation(module: .llm, bytes: 0))
            XCTFail("ambiguous omit-model load should throw")
        } catch let e as AthenaError {
            XCTAssertEqual(e.code, "ambiguous_model")
        }
        let bytes = await m.residentBytes
        XCTAssertEqual(bytes, 0, "no reservation for an ambiguous default")
        let resident = await m.residentModelId()
        XCTAssertNil(resident)
    }

    /// ADR 026: a configured default resolves the omit-model case even with
    /// several selectable models — exactly one is bound.
    func testConfiguredDefaultResolvesOmitModel() async throws {
        let m = StubLLMModule(modelIds: ["a", "b"], configuredDefault: "b")
        try await m.load(
            reservation: MemoryReservation(module: .llm, bytes: 0))
        let resident = await m.residentModelId()
        XCTAssertEqual(resident, "b")
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

/// M48.3 — temperature is inert under a Guide.
///
/// The contract: `GuidedSubstrate` decodes with a hardcoded `ArgMaxSampler()`
/// (`GuidedSubstrate.swift`), so a schema request at temperature 0.1 must emit
/// the byte-identical sequence a temperature-0 request does. The invariant is
/// asserted in prose at `DecodeDispatch.effectiveTemperature` ("under a Guide
/// the decode is masked-argmax and temperature is inert") — this is the only
/// thing that would catch someone swapping a temperature-aware sampler in.
///
/// **Single variable: `temperature`.** `runOnce` hardcodes
/// `speculative: false` and takes no parameter for it, so the arms cannot
/// diverge on anything but temperature. They used to differ in that flag too.
/// It is not a routing input at all — `DecodeDispatch.route`'s signature is
/// `route(hasSchema:hasLogprobSink:)`, so a schema request reaches
/// `GuidedSubstrate` without `speculative` ever being consulted. But the flag
/// was only inert AT DECODE, not on load: `MLXLLMModule.pairMTPDrafter` opens
/// with `guard params.speculative else { return }`, so the old temp-0.1 arm
/// tried to resolve and load a second (drafter) model that the temp-0 arm did
/// not — extra residency and a soft dependency on the test model being
/// MTP-paired. Pinning `false` removes a real asymmetry, not a cosmetic one.
///
/// If ADR 033 ever wires the drafter into the guided path, the fix is a
/// SECOND gate for speculative-guided inertness — do not flip this one to
/// `true`, which would silently change which contract it covers while leaving
/// its name and this doc unchanged.
///
/// This test shared a class with two siblings #47 deleted, and they died of
/// DIFFERENT defects. Distinguished here so this one is not swept up as a
/// third:
///   - `testStructuredGreedyParityAcrossSpeculative` was **vacuous** — it
///     varied ONLY `speculative`, so both arms drove the identical path and
///     its PARITY assertion could not fail. (Its `isEmpty` assertions and the
///     `ensureLoaded` throw could still fail; the parity claim could not.)
///   - `testStructuredSpeculativeAcceptanceRate` was a **guaranteed failure
///     whenever it actually ran**, not a vacuous one — it asserted
///     `XCTAssertGreaterThan(stats.total, 0)` against an observer whose
///     publisher publication S0 had already removed. Being model-gated, it
///     skipped in CI exactly as its vacuous sibling did, so CI history shows
///     no red for either.
///
/// MTP speculative bit-identity is a separate contract with no gate at all;
/// see #64.
///
/// Provenance — this test is neither new nor restored. It landed with M48.3
/// itself as `testStructuredGreedyParityTempIneretUnderGuide` (`033f1b15`,
/// 2026-05-28) and has been in the tree continuously **on `main`** ever
/// since; #47 renamed it in place and rehomed it out of the class whose other
/// two tests it deleted.
///
/// The "restored" wording that used to open this comment was not invented: on
/// the unmerged branch `fix/47-delete-speculative-stats`, `130ffa51` deleted
/// all three tests and `d7da7277` ("fix: restore the temperature-inertness
/// test") added this one back. Neither is an ancestor of `main`, so
/// `git log --all -S testStructuredGreedyParityTempIneretUnderGuide` surfaces
/// a commit that looks like it contradicts the paragraph above. It does not —
/// that history is branch-local, and what landed is a net rename in place.
///
/// Heavy: drives a real MLX model. Gated on ATHENA_RUN_MODEL_TESTS=1.
final class GuidedTemperatureInertnessTests: XCTestCase {

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

    func testTemperatureIsInertUnderGuide() async throws {
        let modelURL = try skipUnlessEnabled()
        let schema = """
            {"type":"object",
             "properties":{
               "answer":{"type":"string","enum":["yes","no"]}
             },
             "required":["answer"]}
            """
        let messages = [
            ChatTurn(
                role: "user",
                content:
                    "Is the sky generally blue on a clear day? "
                    + "Answer in JSON.")
        ]
        // `temperature` is the ONLY difference between the arms; `runOnce`
        // takes no `speculative` parameter, so it cannot become a second one.
        let atTemp = try await runOnce(
            modelURL: modelURL, messages: messages,
            schemaJSON: schema, temperature: 0.1)
        let greedyBaseline = try await runOnce(
            modelURL: modelURL, messages: messages,
            schemaJSON: schema, temperature: 0)
        XCTAssertFalse(atTemp.isEmpty, "temp>0 branch produced empty")
        XCTAssertFalse(
            greedyBaseline.isEmpty, "greedy baseline branch produced empty")
        XCTAssertEqual(
            atTemp, greedyBaseline,
            "a structured request at temp>0 must emit the same "
                + "byte-identical sequence as one at temp=0 — the Guide "
                + "collapses both to masked-argmax, so temperature is inert. "
                + "(M48.3 contract; enforced by GuidedSubstrate's hardcoded "
                + "ArgMaxSampler.)")
    }

    /// No `speculative` parameter BY DESIGN: this class's whole contract is
    /// that `temperature` is the only variable, so the flag is not a knob a
    /// call site can reach. Hardcoding it here makes divergence between the
    /// arms unrepresentable rather than merely discouraged by the class doc.
    private func runOnce(
        modelURL: URL, messages: [ChatTurn],
        schemaJSON: String,
        temperature: Double = 0
    ) async throws -> String {
        let llm = MLXLLMModule(
            modelDirectory: modelURL,
            parameters: .init(
                maxTokens: 64, temperature: Float(temperature),
                speculative: false))
        let gov = MemoryGovernor(totalBudgetBytes: Int(96) << 30)
        await gov.register(llm, evictable: false)
        try await gov.ensureLoaded(.llm)
        var out = ""
        let stream = llm.generateMetered(
            messages: messages, schemaJSON: schemaJSON, tools: nil,
            maxTokens: nil, temperature: temperature,
            topP: nil, seed: nil,
            speculative: false, chatTemplateKwargs: nil)
        for await chunk in stream {
            if case .text(let s) = chunk { out += s }
        }
        return out
    }
}

/// C11 — seeded sampling reproducibility. A per-request seed must make a
/// temperature>0 generation byte-identical across runs (and the same seed
/// twice must agree). Heavy: drives a real MLX model at temp>0, so gated on
/// ATHENA_RUN_MODEL_TESTS=1. NOTE: publication S0 removed the vendored
/// sampling loop, so MTP and non-MTP models alike decode on the substrate
/// stream and the seed rides `GenerateParameters.seed` in both cases.
final class SeededSamplingReproducibilityTests: XCTestCase {

    func testSameSeedTempPositiveIsReproducible() async throws {
        let env = ProcessInfo.processInfo.environment
        guard env["ATHENA_RUN_MODEL_TESTS"] == "1" else {
            throw XCTSkip("set ATHENA_RUN_MODEL_TESTS=1 to run (heavy)")
        }
        let modelURL = ModelStore().resolve(env["ATHENA_TEST_MODEL"])
        guard
            FileManager.default.fileExists(
                atPath: modelURL.appendingPathComponent("config.json").path)
        else { throw XCTSkip("model not present at \(modelURL.path)") }

        func run(seed: Int) async throws -> String {
            let llm = MLXLLMModule(
                modelDirectory: modelURL,
                parameters: .init(maxTokens: 48, temperature: 0.8))
            let gov = MemoryGovernor(totalBudgetBytes: Int(96) << 30)
            await gov.register(llm, evictable: false)
            try await gov.ensureLoaded(.llm)
            var out = ""
            let stream = llm.generateMetered(
                messages: [
                    ChatTurn(role: "user", content: "Write one short sentence.")
                ],
                schemaJSON: nil, tools: nil, maxTokens: nil,
                temperature: 0.8, topP: nil, seed: seed,
                speculative: true, chatTemplateKwargs: nil)
            for await chunk in stream {
                if case .text(let s) = chunk { out += s }
            }
            return out
        }

        let a = try await run(seed: 1234)
        let b = try await run(seed: 1234)
        XCTAssertFalse(a.isEmpty, "seeded temp>0 produced empty")
        XCTAssertEqual(
            a, b,
            "same seed + same prompt + temp>0 must reproduce byte-for-byte "
                + "(C11: per-request GenerateParameters.seed)")
    }
}
