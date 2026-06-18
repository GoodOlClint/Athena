import Foundation

/// Detects, from a checkpoint's `config.json`, whether a transcription model is
/// an architecture the Whisper-only transcription engine cannot load — so the
/// daemon can refuse it with a cause-naming 4xx instead of letting the Whisper
/// loader fail deep with an opaque 500. MLX-free, so it is unit-testable under
/// `swift test` (ADR 009), like `DiarizationBackend`.
///
/// Conservative by design: it only flags **known** non-Whisper ASR families
/// (Parakeet/TDT/RNN-T/Conformer-CTC/NeMo/Canary/wav2vec/HuBERT). A Whisper
/// checkpoint matches none of these, so a real Whisper load is never blocked;
/// a genuinely-unknown arch is left to the Whisper loader (which raises its own
/// error). The list grows as Athena learns about more arches it does not yet
/// support.
public enum TranscriptionArch {
    /// Lowercased substrings in `model_type` / `architectures` that mark a
    /// non-Whisper ASR architecture the transcription engine cannot load.
    public static let unsupportedMarkers: [String] = [
        "parakeet", "tdt", "rnnt", "rnn_t", "rnn-t", "conformer",
        "fastconformer", "nemo", "canary", "wav2vec", "hubert", "wavlm",
    ]

    /// True when the config metadata clearly names a non-Whisper ASR arch.
    public static func isUnsupported(
        modelType: String?, architectures: [String]
    ) -> Bool {
        let hay = ([modelType ?? ""] + architectures).map { $0.lowercased() }
        return hay.contains { field in
            unsupportedMarkers.contains { field.contains($0) }
        }
    }

    /// `(model_type, architectures)` from `<dir>/config.json` (store-entry
    /// symlinks resolved first). MLX-free `JSONSerialization`. Returns
    /// `(nil, [])` when absent/unreadable.
    public static func readConfig(
        in dir: URL
    ) -> (modelType: String?, architectures: [String]) {
        let cfg = dir.resolvingSymlinksInPath()
            .appendingPathComponent("config.json")
        guard let data = try? Data(contentsOf: cfg),
            let obj = try? JSONSerialization.jsonObject(with: data),
            let dict = obj as? [String: Any]
        else { return (nil, []) }
        let type = dict["model_type"] as? String
        let arch = (dict["architectures"] as? [String]) ?? []
        return (type, arch)
    }

    /// True when the checkpoint at `dir` is a known-unsupported ASR arch.
    public static func isUnsupported(in dir: URL) -> Bool {
        let (type, arch) = readConfig(in: dir)
        return isUnsupported(modelType: type, architectures: arch)
    }
}
