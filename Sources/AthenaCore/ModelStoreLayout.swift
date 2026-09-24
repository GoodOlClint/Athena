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
        let identity = id.modelStoreIdentity
        // Defense-in-depth path confinement (D6/NC12): `modelStoreIdentity`
        // is `split("/").last`, so a crafted id like `foo/..` yields the
        // identity `..` and `root/..` would resolve to the store root's
        // PARENT — loading config/weights from outside the store. Every
        // module class resolves local models through here, so confining it
        // once covers the LLM, embedding, transcription, diarization, and
        // speaker load paths. Refuse anything that isn't a plain child-dir
        // name.
        guard !identity.isEmpty, identity != "..", identity != ".",
            !identity.contains("/")
        else { return nil }
        let u = root.appendingPathComponent(identity, isDirectory: true)
        return FileManager.default.fileExists(atPath: u.path) ? u : nil
    }

    /// Sum of the `*.safetensors` shard sizes under `directory`, following
    /// the store-entry symlink and each shard's HF-cache blob symlink. A
    /// per-model admission estimate: file bytes ≥ the tensor bytes MLX maps.
    public static func safetensorsBytes(at directory: URL) -> Int {
        let fm = FileManager.default
        // `contentsOfDirectory` does not traverse a symlinked root, and a
        // pulled shard is itself a ~76 B symlink to ../../blobs/<sha>; sizing
        // either link instead of its target yields a ~0 B estimate.
        let dir = directory.resolvingSymlinksInPath()
        guard
            let entries = try? fm.contentsOfDirectory(
                at: dir,
                includingPropertiesForKeys: [.fileSizeKey])
        else { return 0 }
        var total = 0
        for url in entries where url.pathExtension == "safetensors" {
            total +=
                (try? url.resolvingSymlinksInPath()
                .resourceValues(forKeys: [.fileSizeKey]))?.fileSize ?? 0
        }
        return total
    }
}
