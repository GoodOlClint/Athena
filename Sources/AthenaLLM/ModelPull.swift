import AthenaCore
import Foundation
import HuggingFace
import MLXHuggingFace
import MLXLMCommon

/// Fetch an HF repo snapshot and link it into the model store as
/// `storeRoot/<name>`, so `serve --model <name>` and `athena list`
/// pick it up without a multi-GB copy. M6-cli-3.
public enum ModelPull {
    /// `progress` (0…1 download fraction) is optional — default nil
    /// keeps the daemon/queue callers unchanged; the CLI passes a
    /// renderer.
    public static func pull(
        id: String, revision: String? = nil, into storeRoot: URL,
        progress: (@Sendable (Double) -> Void)? = nil
    ) async throws -> URL {
        let snapshot = try await #hubDownloader(
            HuggingFace.HubClient(
                session: AthenaProxy.proxiedURLSession())
        ).download(
            id: id, revision: revision,
            matching: [
                "*.json", "*.safetensors", "*.txt", "*.jinja",
                "tokenizer*", "*.model",
            ],
            useLatest: false,
            progressHandler: { p in
                progress?(p.fractionCompleted)
            })

        let name = id.split(separator: "/").last.map(String.init) ?? id
        try FileManager.default.createDirectory(
            at: storeRoot, withIntermediateDirectories: true)
        let dest = storeRoot.appendingPathComponent(
            name, isDirectory: true)
        try? FileManager.default.removeItem(at: dest)
        try FileManager.default.createSymbolicLink(
            at: dest, withDestinationURL: snapshot)
        return dest
    }
}
