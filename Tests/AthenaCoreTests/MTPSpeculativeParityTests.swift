import AthenaCore
import Foundation
import XCTest

@testable import AthenaLLM

/// #64 — the MTP speculative bit-identical-greedy gate, restored non-vacuously.
///
/// `docs/speculative-decoding.md` calls MTP speculative decoding lossless:
/// temperature-0 speculative decoding must emit the same token sequence as
/// non-speculative greedy for the same model + prompt. The prior test
/// (`testStructuredGreedyParityAcrossSpeculative`, deleted by #47) was vacuous
/// post-S0: both arms carried a schema, so both routed to `GuidedSubstrate`
/// and the flag toggled only a log field. This one carries NO schema, so
/// `DecodeDispatch.route` sends both arms to `beginGeneration` — the substrate
/// stream whose `speculative: true` arm takes the ADR 032 MTP-drafter overload,
/// the path that actually differs.
///
/// Non-vacuity is asserted, not assumed: the speculative arm must report
/// proposed draft tokens > 0 (the drafter provably ran), and the greedy arm
/// must report none. A run where MTP silently degraded to single-token FAILS
/// rather than passing on an inert flag — that silent skip is the failure mode
/// this gate exists to end.
///
/// Heavy: gated on ATHENA_RUN_MODEL_TESTS=1 and a checkpoint whose MTP drafter
/// actually pairs. Default target is the Gemma 4 pair (ADR 032) — the drafter
/// (`gemma4_assistant`) must be in the store. The Qwen3.5 fused-head path
/// cannot currently pair at all: `-mtp` checkpoints converted from the VLM
/// repo carry `language_model.mtp.*` keys while `qwenMTPSanitizeWeights`
/// filters bare `mtp.*`, so the drafter load fails (keyNotFound) and
/// speculative falls back to single-token — at BOTH the 751aaed and
/// 5b892140 pins (whole chain byte-identical across them), i.e. pre-existing,
/// not a #91 regression. Skips cleanly with a stated reason otherwise. This
/// is also ADR 032's heavy E2E DoD, and the gate ADR 028's bit-identity
/// claim rides across substrate bumps.
final class MTPSpeculativeParityTests: XCTestCase {

    private func skipUnlessEnabled() throws -> URL {
        let env = ProcessInfo.processInfo.environment
        guard env["ATHENA_RUN_MODEL_TESTS"] == "1" else {
            throw XCTSkip("set ATHENA_RUN_MODEL_TESTS=1 to run (heavy)")
        }
        let modelURL = ModelStore().resolve(
            env["ATHENA_TEST_MTP_MODEL"] ?? "gemma-4-26b-a4b-it-4bit")
        guard
            FileManager.default.fileExists(
                atPath: modelURL.appendingPathComponent("config.json").path)
        else {
            throw XCTSkip(
                "no MTP-capable checkpoint at \(modelURL.path) "
                    + "(override with ATHENA_TEST_MTP_MODEL)")
        }
        return modelURL
    }

    private struct Arm {
        let text: String
        let proposed: Int?
    }

    private func run(
        _ llm: MLXLLMModule, speculative: Bool
    ) async -> Arm {
        let messages = [
            ChatTurn(
                role: "user",
                content:
                    "List the first eight prime numbers, comma-separated, "
                    + "then name the planet closest to the sun.")
        ]
        var out = ""
        for await chunk in llm.generateMetered(
            messages: messages, schemaJSON: nil, tools: nil,
            maxTokens: 64, temperature: 0, topP: nil, seed: nil,
            speculative: speculative, chatTemplateKwargs: nil)
        {
            if case .text(let t) = chunk { out += t }
        }
        let counts = await llm.lastMTPDraftCounts
        return Arm(text: out, proposed: counts?.proposed)
    }

    /// Same prompt, same maxTokens, temperature 0, no schema — only the
    /// per-request `speculative` flag toggles. The decoded strings must match
    /// byte-for-byte, and the speculative arm must provably have drafted.
    func testUnstructuredGreedyParityAcrossSpeculative() async throws {
        let modelURL = try skipUnlessEnabled()
        // speculative: true at load so the drafter pairs (Qwen3.5: fused
        // `mtp.*` head from the target's own checkpoint; Gemma 4: the seeded
        // `gemma4_assistant` pairing). Per-request flags select the arm.
        let llm = MLXLLMModule(
            modelDirectory: modelURL,
            parameters: .init(maxTokens: 64, temperature: 0, speculative: true))
        let gov = MemoryGovernor(totalBudgetBytes: Int(96) << 30)
        await gov.register(llm, evictable: false)
        try await gov.ensureLoaded(.llm)
        guard await llm.mtpDrafterResident else {
            throw XCTSkip(
                "checkpoint loaded but no MTP drafter paired (no `mtp.*` "
                    + "weights / no drafter in store) — nothing to gate")
        }

        let spec = await run(llm, speculative: true)
        let greedy = await run(llm, speculative: false)

        XCTAssertFalse(spec.text.isEmpty, "speculative arm produced empty")
        XCTAssertFalse(greedy.text.isEmpty, "greedy arm produced empty")
        // Non-vacuity: the speculative arm actually drafted; the greedy arm
        // actually took the plain path.
        XCTAssertGreaterThan(
            spec.proposed ?? 0, 0,
            "speculative arm reported no proposed draft tokens — MTP did not "
                + "engage, the gate would be vacuous (see #64)")
        XCTAssertEqual(
            greedy.proposed ?? 0, 0,
            "greedy arm reported proposed draft tokens — the flag failed to "
                + "disable speculation")
        XCTAssertEqual(
            spec.text, greedy.text,
            "temp-0 MTP speculative diverged from non-speculative greedy — "
                + "the M2-era bit-identical-greedy contract is broken "
                + "(GDN/Mamba recurrent restore on reject, or the verify mask)")
    }
}
