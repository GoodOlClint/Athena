import AthenaCore
import Foundation
import HuggingFace
import MLXHuggingFace
import MLXLMCommon

/// ADR 021 S4 — the `pull` preflight: answer *"will this model work before I
/// pull it?"* from a Hugging Face **metadata-only** fetch (config files plus a
/// repo file listing; no weights), for every modality Athena serves. The
/// config fetch is the same passive-oracle egress `convert` performs
/// (`matching: ["config.json", …]`); the listing is a Hugging Face request
/// too. It classifies with the shared `ModelSupport` predicate, so the
/// preflight verdict can never disagree with what the loader or `convert`
/// would decide (ADR 021 D3).
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

    /// Metadata-only HF pre-fetch (config files + repo file listing) +
    /// classify. Downloads no weights, then returns the `ModelSupport`
    /// verdict. The config file set
    /// `convert`'s pre-fetch grabs, so sentence-transformers markers (which
    /// live outside `config.json`) are available to the classifier (R4).
    public static func check(
        id: String, revision: String? = nil,
        listFiles: ModelPull.FileLister = ModelPull.hubFileLister
    ) async throws -> ModelSupport {
        let support = try await classify(id: id, revision: revision)
        guard let files = try? await listFiles(id, revision) else { return support }
        return applyListing(support, files: files)
    }

    /// Config-only fetch + classify (no weights, no file listing).
    public static func classify(
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

    /// Downgrades a verdict to `.unsupported` when the repo listing carries no
    /// weights the loader opens (#247).
    public static func applyListing(
        _ support: ModelSupport, files: [String]
    ) -> ModelSupport {
        if case .unsupported = support.loadability { return support }
        guard
            let problem = WeightFiles.listingProblem(
                files: files, modality: support.modality)
        else { return support }
        return ModelSupport(
            modality: support.modality,
            loadability: .unsupported(
                reason: problem,
                guidance: WeightFiles.guidance(modality: support.modality)))
    }
}
