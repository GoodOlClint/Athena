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

    public let rootDirectory: URL

    public init(rootDirectory: URL = ModelStore.defaultRoot) {
        self.rootDirectory = rootDirectory
    }

    /// Resolve a `--model` value: an absolute/existing path is used verbatim,
    /// otherwise it is treated as a model name under the store root. `nil`
    /// when no reference is given (#203 — no compiled-in model id, per ADR
    /// 021's guidance rule; callers that need a default in that case go
    /// through `ModelSelection`'s ADR 026 ambiguity rule instead).
    public func resolve(_ reference: String?) -> URL? {
        guard let reference, !reference.isEmpty else { return nil }
        if reference.hasPrefix("/") {
            return URL(fileURLWithPath: reference, isDirectory: true)
        }
        return rootDirectory.appendingPathComponent(
            reference, isDirectory: true)
    }
}
