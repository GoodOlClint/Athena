import Foundation

/// **The single source of truth for "what is this checkpoint, and can Athena
/// load it?"** (ADR 021 / M77). Derived purely from on-disk `config.json` +
/// sentence-transformers marker files — MLX-free, so it is unit-testable under
/// `swift test` (ADR 008/009), exactly like the detectors it composes.
///
/// `ModelSupport` does **not** fork a fourth classifier. It *composes* the
/// existing focused detectors —
///   - `ModelClass` (ADR 016): generative / vision / embedding,
///   - `TranscriptionArch` (ADR 020): whisper / parakeet,
///   - `DiarizationBackend` (ADR 018): sortformer / pyannote,
/// — into one **modality router** that also covers the modalities none of them
/// did (transcription/diarization/speaker-embedding as first-class verdicts),
/// and adds the **loadability** layer they were missing: a packaging check,
/// distinct from family, that asks whether the fields the *loader* actually
/// requires are present.
///
/// Three consumers share this one predicate so a verdict can never drift from
/// reality (ADR 021 decision 3): the module loaders refuse `.unsupported`
/// packaging with a cause-naming 4xx (not an opaque 500), `convert` redirects
/// the non-convertible audio modalities to `pull`, and the `pull` preflight
/// classifies from a config-only pre-fetch before any multi-GB download.
///
/// **Honesty boundary (ADR 021 decision 4):** a config-only verdict proves
/// routing + packaging — that Athena will route the checkpoint and the loader's
/// required fields are present — **not** that the forward pass is numerically
/// correct. `.loadable` means "Athena can load this," never "this is correct";
/// numeric correctness stays the job of the gated heavy tests.
public struct ModelSupport: Sendable, Equatable {
    public let modality: ModelModality
    public let loadability: Loadability

    public init(modality: ModelModality, loadability: Loadability) {
        self.modality = modality
        self.loadability = loadability
    }
}

/// What modality a checkpoint serves. The two ASR engines and the two
/// diarization backends carry their sub-classification so a consumer can name
/// the exact engine without re-running the detector.
public enum ModelModality: Sendable, Equatable {
    /// A generative text LLM (the substrate's `LLMModelFactory`).
    case llm
    /// A vision-language checkpoint with an image tower (`VLMModelFactory`).
    case vision
    /// A text-embedding model (`EmbedderModelFactory`).
    case embedding
    /// Speech-to-text — Whisper or Parakeet (`.unsupported` never reaches here;
    /// it routes to the `unsupported` modality instead).
    case transcription(TranscriptionArch)
    /// Speaker diarization — Sortformer or pyannote segmentation.
    case diarization(DiarizationBackend)
    /// WeSpeaker-style speaker-embedding (voiceprint) model.
    case speakerEmbedding
    /// A Multi-Token-Prediction speculative **drafter** (e.g. `gemma4_assistant`,
    /// ADR 032) — paired to a generative target, not independently servable.
    /// Loaded via the substrate `MTPDrafterModelFactory`, never the LLM/VLM
    /// factory, and pulled (bf16) rather than converted.
    case mtpDrafter
    /// No modality Athena serves could be identified from the config.
    case unsupported

    /// A short, repo-id-free human label for the modality, used in cause-naming
    /// errors (convert redirect) and the `pull` preflight verdict.
    public var label: String {
        switch self {
        case .llm: return "generative"
        case .vision: return "vision"
        case .embedding: return "embedding"
        case .transcription(let arch): return "transcription (\(arch.rawValue))"
        case .diarization(let backend):
            return "diarization (\(backend.rawValue))"
        case .speakerEmbedding: return "speaker-embedding"
        case .mtpDrafter: return "mtp-drafter"
        case .unsupported: return "unsupported"
        }
    }
}

/// Whether the loader's required packaging is present — a check distinct from
/// modality. `.unknown` is the deliberate "best-effort" verdict for a named
/// generative/vision arch we can't confirm from config alone (the substrate
/// factory is the ground truth and raises the precise error if it lacks that
/// architecture) — the `pull` gate warns-and-proceeds on it rather than
/// blocking a loadable arch.
///
/// Every `.unsupported` `reason`/`guidance` names the **structural
/// requirement** the checkpoint fails and is **free of any hard-coded model id
/// or HF repo** (ADR 021 decision 5) — repos move/vanish; the requirement is
/// stable. A unit test pins that every such string is slash-free (a repo id is
/// always `org/name`).
public enum Loadability: Sendable, Equatable {
    /// The loader's required fields are present; Athena can load this.
    case loadable
    /// Best-effort: the modality is recognized but config alone can't confirm
    /// the substrate implements this architecture. The loader is the authority.
    case unknown
    /// The packaging is missing something the loader requires. `reason` names
    /// what the config lacks; `guidance` names what Athena needs instead.
    case unsupported(reason: String, guidance: String)
}

extension ModelSupport {
    /// Lowercased `model_type` substrings that mark a WeSpeaker-style
    /// speaker-embedding checkpoint (e.g. `wespeaker-resnet34-lm`). A small
    /// static mirror keeps the detector MLX-free, like the other detectors'
    /// marker sets.
    public static let speakerEmbeddingMarkers: [String] = ["wespeaker"]

    /// An MTP speculative drafter (ADR 032) is a distinct substrate-registered
    /// `model_type` that `ModelClass` would otherwise file as `.generative`.
    /// The convention is a `<family>_assistant` suffix — real converts use
    /// both `gemma4_assistant` and `gemma4_unified_assistant`, and no servable
    /// model_type ends in `_assistant`, so a suffix match is the robust rule
    /// (`isMTPDrafterType`). The list is retained for documentation/exact hits.
    public static let mtpDrafterMarkers: [String] = [
        "gemma4_assistant", "gemma4_unified_assistant",
    ]

    /// True iff `type` (lowercased) names an MTP drafter: a known marker or the
    /// `_assistant` suffix convention.
    public static func isMTPDrafterType(_ type: String) -> Bool {
        mtpDrafterMarkers.contains(type) || type.hasSuffix("_assistant")
    }

    /// The MLX-free config fields `classify` decides on. Bundling them in one
    /// struct keeps the decision a pure, trivially unit-testable function with
    /// no filesystem dependency (the `detect(in:)` convenience does the I/O).
    public struct Probe: Sendable, Equatable {
        /// Architecture-agnostic config view (model_type, vision_config, …).
        public var info: ModelConfigInfo
        /// The transcription router's view (NeMo `target`/`decoding` included).
        public var transcription: TranscriptionArch.Config
        /// Whether the snapshot carries any sentence-transformers marker file.
        public var hasSentenceTransformerMarkers: Bool
        /// Whisper's `n_vocab` (the decoder is pinned to the large-v3 family,
        /// vocab 51866). nil when the field is absent.
        public var whisperNVocab: Int?
        /// Whether the NeMo `joint.vocabulary` array is present and non-empty —
        /// the discriminator between a loadable NeMo-format Parakeet export and
        /// a transformers-format Parakeet checkpoint the loader cannot read.
        public var hasJointVocabulary: Bool

        public init(
            info: ModelConfigInfo = ModelConfigInfo(),
            transcription: TranscriptionArch.Config = .init(),
            hasSentenceTransformerMarkers: Bool = false,
            whisperNVocab: Int? = nil,
            hasJointVocabulary: Bool = false
        ) {
            self.info = info
            self.transcription = transcription
            self.hasSentenceTransformerMarkers = hasSentenceTransformerMarkers
            self.whisperNVocab = whisperNVocab
            self.hasJointVocabulary = hasJointVocabulary
        }
    }

    /// Classify a checkpoint from its probed config fields — the pure decision.
    ///
    /// Modality order matters because some audio model types (`parakeet_tdt`,
    /// `sortformer`, `wespeaker-resnet34-lm`) are *named* types that
    /// `ModelClass` would otherwise file as `.generative`. The audio detectors
    /// run first and only claim a checkpoint on a positive ASR/diarization/
    /// speaker signal; everything else falls through to `ModelClass`
    /// (vision → embedding → generative → unknown).
    public static func classify(_ probe: Probe) -> ModelSupport {
        // 1. Transcription — Whisper / Parakeet (positive signal only).
        let arch = TranscriptionArch.classify(probe.transcription)
        switch arch {
        case .whisper:
            return ModelSupport(
                modality: .transcription(.whisper),
                loadability: whisperLoadability(nVocab: probe.whisperNVocab))
        case .parakeet:
            return ModelSupport(
                modality: .transcription(.parakeet),
                loadability: parakeetLoadability(
                    hasJointVocabulary: probe.hasJointVocabulary))
        case .unsupported:
            break  // not an ASR we recognize — keep routing
        }

        // 2. Diarization — Sortformer / pyannote segmentation.
        let backend = DiarizationBackend.classify(
            modelType: probe.info.modelType)
        switch backend {
        case .sortformer, .pyannoteSegmentation:
            // The model_type IS the loader's routing signal and both backends
            // are implemented — loadable on a positive class.
            return ModelSupport(
                modality: .diarization(backend), loadability: .loadable)
        case .unknown:
            break
        }

        // 3. Speaker-embedding — WeSpeaker (voiceprint).
        if let type = probe.info.modelType?.lowercased(),
            speakerEmbeddingMarkers.contains(where: { type.contains($0) })
        {
            return ModelSupport(
                modality: .speakerEmbedding, loadability: .loadable)
        }

        // 3a. MTP drafter — gemma4_assistant et al. (ADR 032). A speculative
        // drafter paired to a target, not a servable model. Detected before
        // ModelClass, which would file its named model_type as `.generative`.
        // The model_type IS the loader's routing signal and the substrate
        // implements it → loadable on a positive match.
        if let type = probe.info.modelType?.lowercased(),
            isMTPDrafterType(type)
        {
            return ModelSupport(modality: .mtpDrafter, loadability: .loadable)
        }

        // 4. Generative / vision / embedding — delegate to ModelClass.
        switch ModelClass.classify(
            info: probe.info,
            hasSentenceTransformerMarkers: probe.hasSentenceTransformerMarkers)
        {
        case .vision:
            // Best-effort: classified by `vision_config`, but the substrate's
            // VLM factory is the ground truth on architecture coverage.
            return ModelSupport(modality: .vision, loadability: .unknown)
        case .embedding:
            // A positive embedding signal (ST markers / embedder-only type) is
            // exactly what the serve path's `EmbedderModelFactory` routes on.
            return ModelSupport(modality: .embedding, loadability: .loadable)
        case .generative:
            // Best-effort: any named `model_type` classifies as generative, but
            // arch coverage is inherited from the substrate (ADR 016) — don't
            // over-promise. The loader raises the precise error if unsupported.
            return ModelSupport(modality: .llm, loadability: .unknown)
        case .unknown:
            // No `model_type` at all — nothing to route on.
            return ModelSupport(
                modality: .unsupported,
                loadability: .unsupported(
                    reason:
                        "the checkpoint's config.json declares no model_type, "
                        + "so its modality cannot be identified",
                    guidance: serveListGuidance))
        }
    }

    /// Whisper loadability: the decoder's special-token ids are pinned to the
    /// large-v3 family (vocab 51866); any other vocab silently mis-decodes, so
    /// it is refused at the packaging layer (ND2 — the loader already enforces
    /// this; ModelSupport surfaces it pre-load too).
    static func whisperLoadability(nVocab: Int?) -> Loadability {
        if nVocab == 51866 { return .loadable }
        let have = nVocab.map(String.init) ?? "an unspecified vocabulary"
        return .unsupported(
            reason:
                "Athena's Whisper decoder is pinned to the large-v3 vocabulary "
                + "(51866); this checkpoint declares \(have)",
            guidance:
                "use a large-v3-family Whisper checkpoint (the large-v3 or "
                + "large-v3-turbo vocabulary), whose special-token layout the "
                + "decoder is built for")
    }

    /// Parakeet loadability: the loader reads `joint.vocabulary` from a
    /// NeMo-format export. A transformers-format `parakeet_tdt` checkpoint omits
    /// it (the M76 field incident), so the route would give a false green and
    /// the loader fail deep — refuse it here by the missing structural field.
    static func parakeetLoadability(hasJointVocabulary: Bool) -> Loadability {
        if hasJointVocabulary { return .loadable }
        return .unsupported(
            reason:
                "the checkpoint's config.json has no joint.vocabulary array, "
                + "which the Parakeet loader reads to build the token table",
            guidance:
                "use a NeMo-format Parakeet-TDT export — an RNN-T or TDT "
                + "checkpoint whose config carries the joint vocabulary; a "
                + "transformers-format Parakeet checkpoint is not loadable")
    }

    /// The modalities Athena serves, named without any slash so the
    /// guidance-rule pin (no `org/name` repo id) stays trivially satisfiable.
    static let serveListGuidance =
        "Athena serves chat models (LLM and vision), text-embedding models, "
        + "transcription models (Whisper or Parakeet), diarization models "
        + "(Sortformer or pyannote), and speaker-embedding models"

    /// Read the routing-relevant fields from `<dir>/config.json` (+ ST marker
    /// files) and classify. Store-entry symlinks are resolved first, like the
    /// detectors it composes. MLX-free `JSONSerialization`; missing fields ⇒
    /// the conservative default.
    public static func detect(in dir: URL) -> ModelSupport {
        let resolved = dir.resolvingSymlinksInPath()
        let cfg = resolved.appendingPathComponent("config.json")
        let dict: [String: Any] =
            (try? Data(contentsOf: cfg)).flatMap {
                try? JSONSerialization.jsonObject(with: $0)
            } as? [String: Any] ?? [:]

        let info =
            ModelConfigInfo.read(modelDirectory: resolved)
            ?? ModelConfigInfo()
        let ta = TranscriptionArch.readConfig(in: resolved)
        let hasST = ModelClass.hasSentenceTransformerMarkers(in: resolved)

        // Whisper `n_vocab` (top-level int).
        let nVocab = (dict["n_vocab"] as? NSNumber)?.intValue
        // NeMo `joint.vocabulary` — present and non-empty.
        let joint = dict["joint"] as? [String: Any]
        let hasJoint = (joint?["vocabulary"] as? [Any])?.isEmpty == false

        return classify(
            Probe(
                info: info, transcription: ta,
                hasSentenceTransformerMarkers: hasST,
                whisperNVocab: nVocab, hasJointVocabulary: hasJoint))
    }
}
