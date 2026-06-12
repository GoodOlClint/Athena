import Foundation
import XCTest

@testable import AthenaCore

/// Brief item 4a — Metal/MLX OOM is classified to a governed 503,
/// never a bare 500 / process abort. Pure, CI-safe.
final class AthenaErrorTests: XCTestCase {

    private struct Fake: Error, CustomStringConvertible {
        let description: String
    }

    func testDetectsMetalOOMVocabulary() {
        for m in [
            "MLX error: [METAL] out of memory",
            "failed to allocate 4096 MB",
            "Insufficient Memory for MTLBuffer",
            "newBufferWithLength returned nil",
            "cannot allocate region / vm_allocate",
        ] {
            XCTAssertTrue(
                AthenaError.isMetalOOM(Fake(description: m)),
                "should flag: \(m)")
        }
    }

    func testIgnoresNonOOMAndAlreadyClassified() {
        XCTAssertFalse(
            AthenaError.isMetalOOM(
                Fake(description: "tokenizer file not found")))
        // An existing AthenaError must not be re-flagged as OOM.
        XCTAssertFalse(
            AthenaError.isMetalOOM(
                AthenaError.moduleNotRegistered(.llm)))
    }

    func testClassifyRoutesOOMTo503AndPassesThrough() {
        let oom = AthenaError.classify(
            Fake(description: "[metal] out of memory"),
            module: .transcription)
        guard case let .metalOutOfMemory(mod, _) = oom else {
            return XCTFail("expected .metalOutOfMemory, got \(oom)")
        }
        XCTAssertEqual(mod, .transcription)
        XCTAssertEqual(oom.httpStatus, 503)
        XCTAssertEqual(oom.code, "metal_oom")

        // Non-OOM substrate failure ⇒ 500 moduleLoadFailed.
        let generic = AthenaError.classify(
            Fake(description: "config.json missing"), module: .llm)
        XCTAssertEqual(generic.httpStatus, 500)
        XCTAssertEqual(generic.code, "module_load_failed")

        // Existing AthenaError passes through untouched.
        let passthrough = AthenaError.classify(
            AthenaError.memoryBudgetExceeded(
                requested: 1, available: 0, module: .llm),
            module: .textEmbedding)
        XCTAssertEqual(passthrough.code, "memory_budget_exceeded")
    }

    func testPromptCacheCapExceededIs503() {
        let e = AthenaError.promptCacheCapExceeded(
            requestedBytes: 9, capBytes: 4)
        XCTAssertEqual(e.httpStatus, 503)
        XCTAssertEqual(e.code, "prompt_cache_cap_exceeded")
        XCTAssertTrue(e.message.contains("9"))
        XCTAssertTrue(e.message.contains("4"))
    }

    // M65.3 — the two input-cap / fail-closed error classes are 400s
    // with stable codes and informative messages.
    func testInputTooLongIs400() {
        let e = AthenaError.inputTooLong(
            module: .textEmbedding, tokens: 99_000, maxTokens: 32_768)
        XCTAssertEqual(e.httpStatus, 400)
        XCTAssertEqual(e.code, "input_too_long")
        XCTAssertTrue(e.message.contains("99000"))
        XCTAssertTrue(e.message.contains("32768"))
    }

    func testStructuredOutputUnavailableIs400() {
        let e = AthenaError.structuredOutputUnavailable(
            detail: "vocab unresolved")
        XCTAssertEqual(e.httpStatus, 400)
        XCTAssertEqual(e.code, "structured_output_unavailable")
        XCTAssertTrue(e.message.contains("vocab unresolved"))
    }

    // NE7 — substrate detail (paths/repo ids/state) must stay OUT of the
    // client-facing `message` and live only in `serverDetail` (server log).
    func testSubstrateDetailNotInClientMessage() {
        let secret = "/Users/op/.athena/models/secret-repo/config.json"
        let load = AthenaError.moduleLoadFailed(.llm, reason: secret)
        XCTAssertFalse(
            load.message.contains(secret),
            "moduleLoadFailed must not leak the substrate reason")
        XCTAssertEqual(load.serverDetail, secret)

        let oom = AthenaError.metalOutOfMemory(
            module: .textEmbedding, detail: secret)
        XCTAssertFalse(
            oom.message.contains(secret),
            "metalOutOfMemory must not leak the detail")
        XCTAssertEqual(oom.serverDetail, secret)

        // Cases that are already client-safe expose no serverDetail.
        XCTAssertNil(AthenaError.requestTimedOut(seconds: 5).serverDetail)
    }
}

/// Brief 4b — the prompt-cache cap is owned by the governor (config
/// default ¼ budget) and surfaced in the snapshot. Pure, CI-safe.
final class PromptCacheCapTests: XCTestCase {

    func testConfigDefaultsToQuarterBudget() {
        let c = GovernorConfig(totalBudgetBytes: 4_000)
        XCTAssertEqual(c.promptCacheCapBytes, 1_000)
        let c2 = GovernorConfig(
            totalBudgetBytes: 4_000, promptCacheCapBytes: 777)
        XCTAssertEqual(c2.promptCacheCapBytes, 777)
    }

    func testGovernorSnapshotExposesCap() async {
        let gov = MemoryGovernor(
            totalBudgetBytes: 8_000, promptCacheCapBytes: 2_500)
        let s = await gov.snapshot()
        XCTAssertEqual(s.promptCacheCapBytes, 2_500)
    }
}
