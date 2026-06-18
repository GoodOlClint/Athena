import AthenaCore
import XCTest

@testable import AthenaLLM

/// ADR 021 S3 — `convert`'s modality-routing decision. MLX-free: the redirect
/// is decided from a config-only `ModelSupport` verdict, so it is unit-pinned
/// without a network fetch or a model load. Pins that the non-convertible
/// modalities (embedding + the audio classes) redirect to `pull` with a
/// cause-naming error — the M76 fix — while generative/vision/unsupported
/// proceed to the quantizer.
final class ConvertRedirectTests: XCTestCase {

    private func support(_ m: ModelModality) -> ModelSupport {
        // Loadability is irrelevant to the convert routing decision (it keys on
        // modality); use a neutral value.
        ModelSupport(modality: m, loadability: .unknown)
    }

    // MARK: convert targets proceed (nil)

    func testGenerativeProceeds() {
        XCTAssertNil(ModelConvert.convertRedirect(for: support(.llm), id: "x"))
    }

    func testVisionProceeds() {
        XCTAssertNil(
            ModelConvert.convertRedirect(for: support(.vision), id: "x"))
    }

    func testUnsupportedTypeProceedsToSubstrate() {
        // No model_type → let the substrate factory raise the precise arch
        // error (wrapped by looksLikeUnsupportedArch), don't pre-redirect.
        XCTAssertNil(
            ModelConvert.convertRedirect(for: support(.unsupported), id: "x"))
    }

    // MARK: non-convertible modalities redirect

    func testEmbeddingRedirects() {
        let e = ModelConvert.convertRedirect(
            for: support(.embedding), id: "my-embedder")
        XCTAssertEqual(e?.code, "unsupported_convert_class")
        XCTAssertEqual(e?.httpStatus, 400)
        XCTAssertTrue(e?.message.contains("my-embedder") == true)
    }

    func testTranscriptionRedirects() {
        for m in [
            ModelModality.transcription(.whisper),
            .transcription(.parakeet),
            .diarization(.sortformer),
            .diarization(.pyannoteSegmentation),
            .speakerEmbedding,
        ] {
            let e = ModelConvert.convertRedirect(for: support(m), id: "my-model")
            XCTAssertEqual(
                e?.code, "unsupported_convert_class", "\(m) should redirect")
            XCTAssertEqual(e?.httpStatus, 400)
            // Points the operator at `pull`, not the quantizer.
            XCTAssertTrue(e?.message.contains("athena pull") == true, "\(m)")
        }
    }

    /// The M76 incident: a Parakeet checkpoint must redirect cleanly, NOT emit
    /// the old misleading "bump the substrate" message from the generative
    /// mis-route.
    func testParakeetIncidentRedirectsNotBumpSubstrate() {
        let e = ModelConvert.convertRedirect(
            for: support(.transcription(.parakeet)),
            id: "some/parakeet-tdt")
        XCTAssertEqual(e?.code, "unsupported_convert_class")
        XCTAssertTrue(
            e?.message.contains("transcription (parakeet)") == true, e!.message)
        XCTAssertFalse(
            e?.message.lowercased().contains("bump the substrate") == true)
    }

    // MARK: guidance rule (ADR 021 D5) — no HARD-CODED repo id

    /// Convert guidance is operational ("run `athena pull <id>`") and uses `/`
    /// as a separator (`--embedding-model … / config`), so a slash-free pin
    /// doesn't fit. The D5 concern here is that no vendor repo is *hard-coded*:
    /// the only model id the message names is the operator's own echoed `id`.
    func testRedirectGuidanceHardCodesNoRepoId() {
        let knownRepoIds = [
            "nvidia/parakeet-tdt-0.6b-v3", "mlx-community/parakeet-tdt-0.6b-v3",
            "mlx-community/whisper-large-v3-turbo",
            "aufklarer/WeSpeaker-ResNet34-LM-MLX",
        ]
        for m in [
            ModelModality.embedding,
            .transcription(.parakeet),
            .diarization(.sortformer),
            .speakerEmbedding,
        ] {
            let e = ModelConvert.convertRedirect(for: support(m), id: "bare-id")
            for repo in knownRepoIds {
                XCTAssertFalse(
                    e?.message.contains(repo) == true,
                    "convert guidance must not hard-code \(repo): \(e!.message)")
            }
        }
    }
}
