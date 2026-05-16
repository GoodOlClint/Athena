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

    /// Default LLM: the substrate-vetted `mlx-community/Qwen3.5-2B-4bit`.
    /// Known-good (validated coherent at greedy). The local `*-mtp`
    /// conversions all emit garbage — a broken converter, not Athena
    /// (proven by a byte-identical substrate-vs-vendored A/B). M2
    /// speculative decoding needs `mtp.*`, so it stays gated on a fixed
    /// mtp checkpoint; this default keeps `athena serve` correct today.
    public static let defaultModelName = "Qwen3.5-2B-4bit"

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
