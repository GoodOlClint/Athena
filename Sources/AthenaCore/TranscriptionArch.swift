import Foundation

/// Which transcription engine a checkpoint drives. Derived purely from the
/// checkpoint's `config.json` — MLX-free, so it is unit-testable under
/// `swift test` (ADR 009), exactly like `DiarizationBackend` (ADR 018) and
/// `ModelClass` (ADR 016).
///
/// ADR 020 turns this from the v0.10.170 *denylist* (Whisper-or-bust, flag a
/// few known-bad arches) into a **positive router**: the single governed
/// `transcription` slot (ADR 011) spans Whisper and Parakeet checkpoints, and
/// `MLXTranscriptionModule` dispatches `loadModel` to the matching engine by
/// this classification. Routing is bounded by the number of engine families
/// (two), not the number of models — anything that is neither Whisper nor
/// Parakeet is `.unsupported` and refused with a cause-naming 4xx.
///
/// Two real config shapes must be recognized as Parakeet:
///   - **transformers** (e.g. `nvidia/parakeet-tdt-0.6b-v3`):
///     `model_type: parakeet_tdt`, `architectures: ["ParakeetForTDT"]`.
///   - **NeMo / MLX** (e.g. `mlx-community/parakeet-tdt-0.6b-v3`, the port's
///     load source): NO top-level `model_type`/`architectures`; the signal is
///     the top-level `target` (`nemo.collections.asr…EncDecRNNTBPEModel`) and
///     `decoding.model_type: tdt`. The router reads both fields so the
///     ungated mlx-community checkpoint classifies correctly.
public enum TranscriptionArch: String, Sendable, Equatable {
    /// OpenAI Whisper (the established default engine). `model_type: whisper`.
    case whisper
    /// NVIDIA Parakeet-TDT (RNN-T + token-and-duration). Selected by `model=`;
    /// additive per ADR 020.
    case parakeet
    /// Neither Whisper nor Parakeet — including other recognized-but-unportable
    /// ASR families (Canary/wav2vec/HuBERT/…) and non-ASR configs. Refused with
    /// `unsupported_transcription_arch` (the router's "neither" path).
    case unsupported

    /// Lowercased substrings that mark a Whisper checkpoint.
    public static let whisperMarkers: [String] = ["whisper"]

    /// Lowercased substrings that mark a Parakeet/TDT checkpoint in either
    /// config shape. `decoding.model_type: tdt` is the decisive NeMo signal —
    /// the mlx-community config carries no top-level `model_type`.
    public static let parakeetMarkers: [String] = ["parakeet", "tdt"]

    /// MLX-free extract of the config fields the router classifies on. A small
    /// struct keeps `classify` pure and trivially unit-testable.
    public struct Config: Sendable, Equatable {
        public var modelType: String?
        public var architectures: [String]
        /// NeMo top-level `target` (e.g. `…EncDecRNNTBPEModel`).
        public var target: String?
        /// `decoding.model_type` (e.g. `tdt`) — present in the NeMo config.
        public var decodingModelType: String?

        public init(
            modelType: String? = nil, architectures: [String] = [],
            target: String? = nil, decodingModelType: String? = nil
        ) {
            self.modelType = modelType
            self.architectures = architectures
            self.target = target
            self.decodingModelType = decodingModelType
        }
    }

    /// Classify a checkpoint's config into the engine that can load it.
    /// Whisper-first (a Whisper config never names "tdt"/"parakeet"; a Parakeet
    /// config never names "whisper", so order is unambiguous). Everything that
    /// matches neither family — unknown, absent, or another ASR arch — is
    /// `.unsupported`.
    public static func classify(_ c: Config) -> TranscriptionArch {
        let hay =
            ([c.modelType, c.target, c.decodingModelType].compactMap { $0 }
                + c.architectures)
            .map { $0.lowercased() }
        func matches(_ markers: [String]) -> Bool {
            hay.contains { field in markers.contains { field.contains($0) } }
        }
        if matches(whisperMarkers) { return .whisper }
        if matches(parakeetMarkers) { return .parakeet }
        return .unsupported
    }

    /// Read the routing-relevant fields from `<dir>/config.json` (store-entry
    /// symlinks resolved first, like `DiarizationBackend.readModelType`).
    /// MLX-free `JSONSerialization`. Missing fields ⇒ `nil`/empty.
    public static func readConfig(in dir: URL) -> Config {
        let cfg = dir.resolvingSymlinksInPath()
            .appendingPathComponent("config.json")
        guard let data = try? Data(contentsOf: cfg),
            let obj = try? JSONSerialization.jsonObject(with: data),
            let dict = obj as? [String: Any]
        else { return Config() }
        let type = dict["model_type"] as? String
        let arch = (dict["architectures"] as? [String]) ?? []
        let target = dict["target"] as? String
        let decodingType =
            (dict["decoding"] as? [String: Any])?["model_type"] as? String
        return Config(
            modelType: type, architectures: arch,
            target: target, decodingModelType: decodingType)
    }

    /// Classify the checkpoint at `dir` directly from its `config.json`.
    public static func detect(in dir: URL) -> TranscriptionArch {
        classify(readConfig(in: dir))
    }
}
