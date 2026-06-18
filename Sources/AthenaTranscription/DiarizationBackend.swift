import Foundation

/// Which diarization engine a checkpoint drives. Derived purely from the
/// checkpoint's `config.json` `model_type` — MLX-free, so it is unit-testable
/// under `swift test` (ADR 009), exactly like `ModelClass` (ADR 016).
///
/// The `diarization` module is a single governed slot (ADR 011) whose allowlist
/// spans both end-to-end Sortformer checkpoints and pyannote segmentation
/// checkpoints. On load, `MLXDiarizationModule` classifies the resident model
/// with this detector and instantiates the matching backend; the
/// `/v1/audio/diarizations` `method` selector must agree with the resident
/// model's backend or the request is refused with a cause-naming 4xx. The
/// point (as with `ModelClass`): backend routing is bounded by the number of
/// engine families (two), not the number of models.
public enum DiarizationBackend: String, Sendable, Equatable {
    /// NVIDIA Sortformer — end-to-end "who spoke when", architecturally ≤4
    /// speakers. `model_type: sortformer`.
    case sortformer
    /// pyannote PyanNet segmentation (SincNet + BiLSTM + powerset head) — the
    /// learned front-end of the overlap-aware, arbitrary-speaker pyannote
    /// pipeline. `model_type: pyannote-segmentation`.
    case pyannoteSegmentation
    /// No recognizable `model_type` — cannot classify.
    case unknown

    /// `model_type` strings (lowercased) that mark a pyannote segmentation
    /// checkpoint. The canonical aufklarer/MLX mirror uses
    /// `pyannote-segmentation`; `pyannet` is the upstream class name and is
    /// accepted defensively.
    public static let pyannoteTypes: Set<String> = [
        "pyannote-segmentation", "pyannote_segmentation", "pyannet",
    ]

    /// `model_type` strings (lowercased) that mark a Sortformer checkpoint.
    public static let sortformerTypes: Set<String> = ["sortformer"]

    /// Classify from a checkpoint's `model_type` (already lowercased by the
    /// caller is not required — this lowercases). `nil`/empty ⇒ `.unknown`.
    public static func classify(modelType: String?) -> DiarizationBackend {
        guard let t = modelType?.lowercased(),
            !t.trimmingCharacters(in: .whitespaces).isEmpty
        else { return .unknown }
        if sortformerTypes.contains(t) { return .sortformer }
        if pyannoteTypes.contains(t) { return .pyannoteSegmentation }
        return .unknown
    }

    /// Read just `config.json`'s `model_type` from a snapshot directory
    /// (store-entry symlinks resolved first, like `ModelConfigInfo.read`) and
    /// classify. MLX-free; uses `JSONSerialization` so this file carries no
    /// dependency on the LLM config types.
    public static func detect(in dir: URL) -> DiarizationBackend {
        classify(modelType: readModelType(in: dir))
    }

    /// Lowercased `model_type` from `<dir>/config.json`, or nil if the file is
    /// absent/unreadable/has no `model_type`.
    public static func readModelType(in dir: URL) -> String? {
        let base = dir.resolvingSymlinksInPath()
        let cfg = base.appendingPathComponent("config.json")
        guard let data = try? Data(contentsOf: cfg),
            let obj = try? JSONSerialization.jsonObject(with: data),
            let dict = obj as? [String: Any],
            let type = dict["model_type"] as? String
        else { return nil }
        return type
    }
}
