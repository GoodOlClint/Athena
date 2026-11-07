import Foundation

/// The one definition of which weight files a checkpoint may carry, shared by
/// the Whisper loader, `athena verify` and the `pull` preflight so they cannot
/// disagree about what loads.
public enum WeightFiles {
    /// Whisper single-file weights, in preference order. Sharded
    /// `model-*.safetensors` is not supported by the Whisper loader.
    public static let whisperNames = ["weights.safetensors", "model.safetensors"]

    private static let foreignExtensions: Set<String> = [
        "npz", "bin", "pt", "pth", "ckpt", "gguf", "onnx", "h5", "msgpack",
    ]

    public static func whisper(in dir: URL) -> URL? {
        whisperNames.map { dir.appending(component: $0) }
            .first { FileManager.default.fileExists(atPath: $0.path) }
    }

    /// Why a repo file listing carries no weights Athena can load, or nil when
    /// it does. `files` are repo-relative paths.
    public static func listingProblem(
        files: [String], modality: ModelModality?
    ) -> String? {
        let safetensors = files.filter { $0.hasSuffix(".safetensors") }
        if case .transcription(.whisper)? = modality {
            if safetensors.contains(where: whisperNames.contains) { return nil }
            let found =
                safetensors.isEmpty ? "" : "; found \(safetensors.sorted().joined(separator: ", "))"
            return "no \(whisperNames.joined(separator: " or ")) weights"
                + found + foreignSuffix(files, haveSafetensors: !safetensors.isEmpty)
        }
        if !safetensors.isEmpty { return nil }
        return "no *.safetensors weights" + foreignSuffix(files, haveSafetensors: false)
    }

    private static func foreignSuffix(_ files: [String], haveSafetensors: Bool) -> String {
        let foreign = files.filter {
            foreignExtensions.contains(($0 as NSString).pathExtension.lowercased())
        }
        guard !foreign.isEmpty, !haveSafetensors else { return "" }
        return "; found \(foreign.sorted().joined(separator: ", "))"
    }
}
