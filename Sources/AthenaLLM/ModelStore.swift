import AthenaCore
import Foundation

/// Resolves where LLM weights live on disk. A model is referenced either
/// by an absolute directory path or by a name resolved under the store
/// root. The root defaults to `~/.athena/models` and is overridable via
/// `--model-store` / the `model_store` config key (e.g. to point at an
/// external SSD).
public struct ModelStore: Sendable {
    /// Default model store: `~/.athena/models` (sibling of the data
    /// dir). Self-contained on the boot volume — no external disk
    /// assumed. Models land here via `pull`/`convert`.
    public static let defaultRoot = AthenaEnv.userHome()
        .appendingPathComponent(".athena/models", isDirectory: true)

    /// Default LLM: 4-bit Qwen3.5-27B with `mtp.*` preserved — the brief's
    /// intended M2 default. The earlier "garbage" from these checkpoints
    /// was a one-line norm-shift convention mismatch (stock mlx-swift-lm
    /// double-shifted fork mtp checkpoints), now fixed in
    /// `AthenaQwen35.sanitize`. Validated coherent at greedy; `mtp.*`
    /// retained for M2.2 speculative decoding without a re-pull.
    /// (`mlx-community/Qwen3.5-2B-4bit` remains a known-good lightweight
    /// alternative via `--model`.)
    public static let defaultModelName = "Qwen3.5-27B-4bit-mtp"

    public let rootDirectory: URL

    public init(rootDirectory: URL = ModelStore.defaultRoot) {
        self.rootDirectory = rootDirectory
    }

    /// Resolve a `--model` value: an absolute/existing path is used verbatim,
    /// otherwise it is treated as a model name under the store root.
    public func resolve(_ reference: String?) -> URL {
        guard let reference, !reference.isEmpty else {
            return rootDirectory.appendingPathComponent(
                Self.defaultModelName, isDirectory: true)
        }
        if reference.hasPrefix("/") {
            return URL(fileURLWithPath: reference, isDirectory: true)
        }
        return rootDirectory.appendingPathComponent(
            reference, isDirectory: true)
    }
}
