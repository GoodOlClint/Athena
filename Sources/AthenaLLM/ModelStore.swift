import Foundation

/// Resolves where LLM weights live on disk. Athena's model store defaults to
/// the external SSD (the HF cache and pre-converted MLX models live there,
/// not on the boot volume). A model is referenced either by an absolute
/// directory path or by a name resolved under the store root.
public struct ModelStore: Sendable {
    /// External-SSD model store. Pre-converted MLX Qwen models (MTP weights
    /// preserved) live here as plain directories — loadable with no download.
    public static let defaultRoot = URL(
        fileURLWithPath: "/Volumes/SB-XTM5/mlx-models", isDirectory: true)

    /// Default green-field LLM: 4-bit Qwen3.5-27B with `mtp.*` preserved
    /// (M1 generation today, M2 speculative decoding without re-pull).
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
