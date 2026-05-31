import Foundation

/// M54 — shared local-store-directory resolution, so every module class
/// (LLM, embedding, transcription, diarization, speaker) loads a model
/// the same way: resolve its configured id to a local store directory and
/// load from there when present, rather than going through the Hub by HF
/// id. This is what lets a request name a model by either its full HF id
/// (`org/name`) or its bare store-dir name (the form `athena pull`
/// creates) — both share `modelStoreIdentity` and resolve the same dir.
public enum ModelStoreLayout {
    /// The local store directory for `id` under `storeRoot`, or for an
    /// absolute-path id the path itself, IF it exists on disk. Returns nil
    /// when the model is not materialized locally — the caller then falls
    /// back to loading by HF id via the Hub. Inference never auto-pulls;
    /// pulling a configured-but-missing model is an operator action.
    public static func localDirectory(
        for id: String, storeRoot: URL?
    ) -> URL? {
        if id.hasPrefix("/") {
            let u = URL(fileURLWithPath: id, isDirectory: true)
            return FileManager.default.fileExists(atPath: u.path) ? u : nil
        }
        guard let root = storeRoot else { return nil }
        let u = root.appendingPathComponent(
            id.modelStoreIdentity, isDirectory: true)
        return FileManager.default.fileExists(atPath: u.path) ? u : nil
    }
}
