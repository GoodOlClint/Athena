import XCTest

import AthenaCore  // NB4 (M70.1b): the KVCompression enum moved here.
@testable import AthenaLLM  // the .servesArch extension stays in AthenaLLM

/// Validated-architecture tiers (M23 fork D) + the codec/arch "serves"
/// check (M23 fork B). Pure logic — always runs in CI.
final class SupportedModelsTests: XCTestCase {

    func testQwen35FamilyDetected() {
        XCTAssertTrue(SupportedModels.isQwen35("qwen3_5"))
        XCTAssertTrue(SupportedModels.isQwen35("qwen3_5_moe"))
        XCTAssertTrue(SupportedModels.isQwen35("QWEN3_5_TEXT"))
        XCTAssertFalse(SupportedModels.isQwen35("llama"))
        XCTAssertFalse(SupportedModels.isQwen35(nil))
    }

    func testSupportTiers() {
        XCTAssertEqual(SupportedModels.support(for: "qwen3_5"), .qwen35)
        XCTAssertEqual(SupportedModels.support(for: "llama"), .validated)
        XCTAssertEqual(SupportedModels.support(for: "gemma3"), .validated)
        XCTAssertEqual(SupportedModels.support(for: "gemma4_unified"), .validated)
        XCTAssertEqual(SupportedModels.support(for: "GEMMA4_TEXT"), .validated)
        XCTAssertEqual(SupportedModels.support(for: "phi3"), .validated)
        XCTAssertEqual(
            SupportedModels.support(for: "some_exotic_arch"), .bestEffort)
        XCTAssertEqual(SupportedModels.support(for: nil), .bestEffort)
    }

    func testDescribeMentionsTypeAndTier() {
        XCTAssertTrue(
            SupportedModels.describe(modelType: "llama").contains("llama"))
        XCTAssertTrue(
            SupportedModels.describe(modelType: "qwen3_5")
                .contains("MTP speculative"))
        XCTAssertTrue(
            SupportedModels.describe(modelType: "llama")
                .contains("guided structured output"))
        XCTAssertTrue(
            SupportedModels.describe(modelType: nil).contains("unknown"))
    }

    // MARK: - fork B: does the codec serve the arch?

    func testNoneServesEveryArch() {
        for t in ["qwen3_5", "llama", "gemma3", "gemma4_unified", nil] {
            XCTAssertTrue(KVCompression.none.servesArch(modelType: t))
        }
    }

    func testTriattentionServesOnlyQwen35() {
        XCTAssertTrue(
            KVCompression.triattention.servesArch(modelType: "qwen3_5"))
        XCTAssertTrue(
            KVCompression.triattention.servesArch(
                modelType: "qwen3_5_moe"))
        XCTAssertFalse(
            KVCompression.triattention.servesArch(modelType: "llama"))
        XCTAssertFalse(
            KVCompression.triattention.servesArch(modelType: "gemma3"))
        XCTAssertFalse(
            KVCompression.triattention.servesArch(
                modelType: "gemma4_unified"))
        XCTAssertFalse(
            KVCompression.triattention.servesArch(modelType: nil))
    }
}
