import Foundation

/// Wire projection of a `ModelSupport` verdict for the typed model listing
/// (usability audit 2026-07-02 §4). These are the additive `ModelEntryDTO`
/// fields the daemon sends and the local `athena ls` renders — the semantic
/// classification, kept MLX-free in AthenaCore. The TYPE column string is
/// derived from these fields by the Foundation-only formatter in AthenaClient
/// (`ModelTypeFormat`) so local and remote render identically without the
/// portable client linking AthenaCore.
extension ModelSupport {
    /// Coarse modality token: `llm`/`vision`/`embedding`/`transcription`/
    /// `diarization`/`speaker`/`draft`/`unsupported`.
    public var wireModality: String {
        switch modality {
        case .llm: return "llm"
        case .vision: return "vision"
        case .embedding: return "embedding"
        case .transcription: return "transcription"
        case .diarization: return "diarization"
        case .speakerEmbedding: return "speaker"
        case .mtpDrafter: return "draft"
        case .unsupported: return "unsupported"
        }
    }

    /// Sub-engine for the ASR/diarization modalities (`whisper`/`parakeet`/
    /// `sortformer`/`pyannote`), nil otherwise.
    public var wireEngine: String? {
        switch modality {
        case .transcription(let arch): return arch.rawValue
        case .diarization(.sortformer): return "sortformer"
        case .diarization(.pyannoteSegmentation): return "pyannote"
        default: return nil
        }
    }

    /// `loadable`/`unknown`/`unsupported` — the packaging verdict.
    public var wireLoadability: String {
        switch loadability {
        case .loadable: return "loadable"
        case .unknown: return "unknown"
        case .unsupported: return "unsupported"
        }
    }

    /// A speculative drafter (ADR 032) — never independently servable.
    public var isDraft: Bool { modality == .mtpDrafter }
}
