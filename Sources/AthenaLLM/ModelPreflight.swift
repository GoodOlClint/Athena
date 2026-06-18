import AthenaCore
import Foundation
import HuggingFace
import MLXHuggingFace
import MLXLMCommon

/// ADR 021 S4 — the `pull` preflight: answer *"will this model work before I
/// pull it?"* from a **config-only** Hugging Face metadata fetch, for every
/// modality Athena serves. It reuses the same sanctioned, passive-oracle
/// metadata egress `convert` already performs (`matching: ["config.json", …]`)
/// — no new outbound surface — and classifies with the shared `ModelSupport`
/// predicate, so the preflight verdict can never disagree with what the loader
/// or `convert` would decide (ADR 021 D3).
///
/// Honesty boundary (ADR 021 D4): the verdict proves routing + packaging —
/// *"Athena will route this and the loader's required fields are present"* —
/// **not** that the forward pass is numerically correct. Messaging says "Athena
/// can load this," never "this is correct."
public enum ModelPreflight {
    /// What a plain `pull` should do given a support verdict (ADR 021 D2):
    /// refuse early on a known-unsupported packaging (no multi-GB waste),
    /// warn-and-proceed on `.unknown` (let the substrate try a best-effort
    /// generative/vision load), proceed silently on `.loadable`.
    public enum Decision: Equatable, Sendable {
        case proceed
        case warn(String)
        case refuse(reason: String)
    }

    /// The pure, MLX-free gate decision from a verdict (unit-pinned, ADR
    /// 008/009). `id` is the operator's own input, echoed for context — not a
    /// hard-coded repo (D5).
    public static func gate(_ support: ModelSupport, id: String) -> Decision {
        switch support.loadability {
        case .loadable:
            return .proceed
        case .unknown:
            return .warn(
                "Athena cannot confirm from config alone that '\(id)' "
                + "(\(support.modality.label)) is an architecture the substrate "
                + "implements; that is decided when the model loads. Pulling "
                + "anyway.")
        case let .unsupported(reason, guidance):
            return .refuse(reason: "\(reason); \(guidance)")
        }
    }

    /// Config-only HF pre-fetch + classify. Downloads metadata only (no
    /// weights), then returns the `ModelSupport` verdict. The same file set
    /// `convert`'s pre-fetch grabs, so sentence-transformers markers (which
    /// live outside `config.json`) are available to the classifier (R4).
    public static func check(
        id: String, revision: String? = nil
    ) async throws -> ModelSupport {
        let downloader = #hubDownloader(
            HuggingFace.HubClient(
                session: AthenaProxy.proxiedURLSession()))
        let meta = try await downloader.download(
            id: id, revision: revision,
            matching: [
                "config.json", "modules.json",
                "config_sentence_transformers.json",
                "sentence_bert_config.json",
            ],
            useLatest: false,
            progressHandler: { _ in })
        return ModelSupport.detect(in: meta)
    }
}
