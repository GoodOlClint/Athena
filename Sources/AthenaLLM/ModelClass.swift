import Foundation

/// Which substrate model-class factory a checkpoint belongs to. Derived purely
/// from on-disk metadata (config.json + sentence-transformers artifacts) —
/// MLX-free, so it is unit-testable under `swift test` (ADR 009).
///
/// The substrate exposes three model-class factories, each with its own
/// `model_type` registry: `LLMModelFactory` (generative), `VLMModelFactory`
/// (vision), and `EmbedderModelFactory` (embedding). `athena convert` is a
/// generative/vision quantization pipeline; it uses this to dispatch the
/// LLM/VLM routes and to REDIRECT embedding models (ADR 016) — an embedder
/// loads in source precision via the serve path and does not need a converted
/// on-disk artifact. The point: convert routing is bounded by the number of
/// classes (three), not the number of models.
public enum ModelClass: String, Sendable, Equatable {
    case generative
    case vision
    case embedding
    /// No `model_type` at all — can't classify. (A *present-but-unrecognized*
    /// generative type still classifies as `.generative`; the substrate's own
    /// factory raises the precise "no architecture" error for it.)
    case unknown

    /// `model_type`s claimed ONLY by the substrate's `EmbedderTypeRegistry`
    /// (encoder embedders with no generative-LM counterpart). The OVERLAP
    /// arches (`gemma3`/`gemma3_text`, `qwen3`, …) live in BOTH the embedder
    /// and LLM registries and are disambiguated by sentence-transformers
    /// artifacts, never by type name. Kept as a small static mirror so the
    /// detector stays MLX-free; mirrors `EmbedderTypeRegistry`'s encoder set.
    public static let embedderOnlyTypes: Set<String> = [
        "bert", "roberta", "xlm-roberta", "distilbert", "nomic_bert",
    ]

    /// Repo-root filenames whose presence marks a sentence-transformers
    /// embedding model (a pooling head over a backbone). Any one is a
    /// sufficient signal — this is what disambiguates an embedding checkpoint
    /// whose `model_type` (e.g. `gemma3_text`) is also a generative arch.
    public static let sentenceTransformerMarkers: Set<String> = [
        "modules.json", "config_sentence_transformers.json",
        "sentence_bert_config.json",
    ]

    /// Classify from parsed config + whether the snapshot carries
    /// sentence-transformers markers. Order matters: vision wins first (an
    /// image tower must load through the VLM factory), then embedding (ST
    /// markers OR an embedder-only `model_type`), then any named generative
    /// type, else unknown.
    public static func classify(
        info: ModelConfigInfo?,
        hasSentenceTransformerMarkers: Bool
    ) -> ModelClass {
        if info?.hasVisionConfig == true { return .vision }
        let type = info?.modelType?.lowercased()
        if hasSentenceTransformerMarkers { return .embedding }
        if let type, embedderOnlyTypes.contains(type) { return .embedding }
        return type == nil ? .unknown : .generative
    }

    /// True when `dir` (a downloaded snapshot; a store-entry symlink is
    /// resolved first, like `ModelConfigInfo.read`) carries any
    /// sentence-transformers marker file.
    public static func hasSentenceTransformerMarkers(in dir: URL) -> Bool {
        let base = dir.resolvingSymlinksInPath()
        let fm = FileManager.default
        return sentenceTransformerMarkers.contains {
            fm.fileExists(atPath: base.appendingPathComponent($0).path)
        }
    }

    /// Convenience: classify directly from a snapshot directory.
    public static func detect(in dir: URL) -> ModelClass {
        classify(
            info: ModelConfigInfo.read(modelDirectory: dir),
            hasSentenceTransformerMarkers:
                hasSentenceTransformerMarkers(in: dir))
    }
}
