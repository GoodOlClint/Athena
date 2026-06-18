import AthenaCore
import XCTest

@testable import AthenaLLM

/// ADR 021 S4 — the `pull` preflight gate decision. MLX-free and config-only:
/// the decision keys on a `ModelSupport` verdict, so it is unit-pinned without
/// a network fetch (ADR 008/009). Pins D2: refuse-early on unsupported,
/// warn-and-proceed on unknown, proceed on loadable.
final class ModelPreflightTests: XCTestCase {

    private func support(_ l: Loadability, _ m: ModelModality = .llm)
        -> ModelSupport
    {
        ModelSupport(modality: m, loadability: l)
    }

    func testLoadableProceeds() {
        XCTAssertEqual(
            ModelPreflight.gate(
                support(.loadable, .transcription(.whisper)), id: "x"),
            .proceed)
    }

    func testUnknownWarnsAndProceeds() {
        let d = ModelPreflight.gate(support(.unknown, .llm), id: "my-llm")
        guard case let .warn(msg) = d else {
            return XCTFail("expected .warn, got \(d)")
        }
        XCTAssertTrue(msg.contains("my-llm"))
        XCTAssertTrue(msg.contains("generative"))
    }

    func testUnsupportedRefusesWithStructuralReason() {
        let d = ModelPreflight.gate(
            support(
                .unsupported(
                    reason: "config has no joint.vocabulary array",
                    guidance: "use a NeMo-format export"),
                .transcription(.parakeet)),
            id: "x")
        guard case let .refuse(reason) = d else {
            return XCTFail("expected .refuse, got \(d)")
        }
        XCTAssertTrue(reason.contains("joint.vocabulary"), reason)
        XCTAssertTrue(reason.contains("NeMo-format export"), reason)
    }

    /// The refuse path must not invent a repo id beyond what the verdict carries
    /// (the verdict strings are already pinned slash-free in ModelSupportTests).
    func testRefuseReasonHardCodesNoRepoId() {
        let d = ModelPreflight.gate(
            support(
                .unsupported(
                    reason: "decoder requires the large-v3 vocabulary (51866)",
                    guidance: "use a large-v3-family checkpoint"),
                .transcription(.whisper)),
            id: "bare-id")
        guard case let .refuse(reason) = d else { return XCTFail() }
        XCTAssertFalse(reason.contains("/"), reason)
    }
}
