import AthenaClient
import Foundation
import XCTest

/// Usability audit 2026-07-02 §4 — TYPE column + `--type` filter. `ModelTypeFormat`
/// is the single Foundation-only formatter behind both the local and remote
/// renderers, so these pins are what guarantees "identical local and remote".
final class ModelTypeFormatTests: XCTestCase {
    func testColumnForEveryModality() {
        func col(_ m: String?, _ e: String? = nil, draft: Bool = false, mtp: Bool = false)
            -> String
        {
            ModelTypeFormat.column(
                modality: m, engine: e, draft: draft, fusedMTP: mtp)
        }
        XCTAssertEqual(col("llm"), "llm")
        XCTAssertEqual(col("llm", mtp: true), "llm +mtp")
        XCTAssertEqual(col("vision"), "vision")
        XCTAssertEqual(col("embedding"), "embed")
        XCTAssertEqual(col("speaker"), "speaker")
        XCTAssertEqual(col("transcription", "whisper"), "asr:whisper")
        XCTAssertEqual(col("transcription", "parakeet"), "asr:parakeet")
        XCTAssertEqual(col("diarization", "sortformer"), "diar:sortformer")
        XCTAssertEqual(col("diarization", "pyannote"), "diar:pyannote")
        XCTAssertEqual(col("unsupported"), "unsupported")
        // draft wins regardless of modality echo.
        XCTAssertEqual(col("draft", draft: true), "draft")
        // pre-typing daemon (no fields) → empty cell, never a crash.
        XCTAssertEqual(col(nil), "")
    }

    func testTypeFilterMatching() {
        func m(
            _ filter: String, _ mod: String?, _ eng: String? = nil,
            draft: Bool = false, mtp: Bool = false
        ) -> Bool {
            ModelTypeFormat.matches(
                filter: filter, modality: mod, engine: eng, draft: draft,
                fusedMTP: mtp)
        }
        // Exact TYPE column.
        XCTAssertTrue(m("draft", "draft", draft: true))
        XCTAssertTrue(m("asr:whisper", "transcription", "whisper"))
        // Leading token.
        XCTAssertTrue(m("asr", "transcription", "whisper"))
        XCTAssertTrue(m("diar", "diarization", "sortformer"))
        // Bare modality token.
        XCTAssertTrue(m("transcription", "transcription", "whisper"))
        XCTAssertTrue(m("embedding", "embedding"))
        // `llm` matches both plain and fused.
        XCTAssertTrue(m("llm", "llm"))
        XCTAssertTrue(m("llm", "llm", mtp: true))
        // Case-insensitive.
        XCTAssertTrue(m("DRAFT", "draft", draft: true))
        // Negative.
        XCTAssertFalse(m("draft", "llm"))
        XCTAssertFalse(m("vision", "llm"))
    }
}
