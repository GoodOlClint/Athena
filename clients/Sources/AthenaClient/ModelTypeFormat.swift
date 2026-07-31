import Foundation

/// TYPE-column rendering + `--type` filtering for `athena ls` (usability audit
/// 2026-07-02 §4). Pure, Foundation-only, and — because this file compiles into
/// BOTH the portable client package and the macOS `athena` target — it is the
/// single formatter behind the local AND remote renderers, so their TYPE
/// columns are byte-identical without the portable client linking AthenaCore.
///
/// Inputs are the additive `ModelEntryDTO` fields the daemon classifies with
/// `ModelSupport` (AthenaCore): `modality` token, `engine` sub-label, and the
/// `draft`/`fusedMTP` attribute bits. Unit-pinned (ADR 008/009).
public enum ModelTypeFormat {
    /// The TYPE column string: `llm`, `llm +mtp`, `vision`, `draft`,
    /// `asr:whisper`, `diar:sortformer`, `embed`, `speaker`, `unsupported`.
    /// Falls back to the raw modality token for any future modality so an old
    /// client never renders an empty cell.
    public static func column(
        modality: String?, engine: String?, draft: Bool, fusedMTP: Bool
    ) -> String {
        if draft { return "draft" }
        switch modality {
        case "llm": return fusedMTP ? "llm +mtp" : "llm"
        case "vision": return "vision"
        case "embedding": return "embed"
        case "speaker": return "speaker"
        case "transcription": return "asr:\(engine ?? "?")"
        case "diarization": return "diar:\(engine ?? "?")"
        case "unsupported": return "unsupported"
        case let other?: return other
        case nil: return ""  // pre-typing daemon — no classification sent
        }
    }

    /// Does a row match `--type <filter>`? Case-insensitive. Matches the whole
    /// TYPE column (`llm +mtp`, `asr:whisper`), the leading token before `:`
    /// (`asr`, `diar`), or the bare modality (`llm` also matches `llm +mtp`).
    public static func matches(
        filter: String, modality: String?, engine: String?,
        draft: Bool, fusedMTP: Bool
    ) -> Bool {
        let f = filter.lowercased()
        let col = column(
            modality: modality, engine: engine, draft: draft,
            fusedMTP: fusedMTP
        ).lowercased()
        if col == f { return true }
        // Leading token (`asr` matches `asr:whisper`; `llm` matches `llm +mtp`).
        let head = col.split(whereSeparator: { $0 == ":" || $0 == " " }).first
        if head.map(String.init) == f { return true }
        // Bare modality token (`transcription`, `diarization`, `embedding`).
        if (modality ?? "").lowercased() == f { return true }
        return false
    }
}
