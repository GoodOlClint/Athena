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
}
