import AthenaCore
import Foundation

/// One speaker turn (seconds from audio start; `speaker` is a stable
/// integer id within this result).
public struct DiarizationTurn: Sendable, Codable {
    public let start: Double
    public let end: Double
    public let speaker: Int
    public init(start: Double, end: Double, speaker: Int) {
        self.start = start
        self.end = end
        self.speaker = speaker
    }
}

/// Clean, `Sendable` diarization result for the module boundary —
/// decoupled from the vendored `DiarizationOutput` (which carries a
/// non-Sendable MLXArray).
public struct DiarizationResult: Sendable {
    public let turns: [DiarizationTurn]
    public let numSpeakers: Int
    public init(turns: [DiarizationTurn], numSpeakers: Int) {
        self.turns = turns
        self.numSpeakers = numSpeakers
    }
}

/// One locally-active speaker region from the segmentation front-end of the
/// pyannote pipeline (ADR 018): a time span during which one *local* speaker
/// track is active. `localSpeaker` is meaningful only relative to `window` —
///
/// - two regions in the SAME `window` with different `localSpeaker` ARE known
///   to be different people (the same-window cannot-link clustering
///   constraint), and
/// - two regions in DIFFERENT windows are not known to be the same or
///   different person — that is what cross-window clustering resolves.
///
/// Overlapping speech is represented by two regions whose spans intersect.
public struct SpeakerActivityRegion: Sendable, Equatable {
    public let start: Double
    public let end: Double
    /// Index of the segmentation window this region came from (for the
    /// cannot-link constraint).
    public let window: Int
    /// Local speaker track within `window` (0-based, ≤ the model's
    /// max-simultaneous-speakers).
    public let localSpeaker: Int
    public init(start: Double, end: Double, window: Int, localSpeaker: Int) {
        self.start = start
        self.end = end
        self.window = window
        self.localSpeaker = localSpeaker
    }
}

/// Typed inference contract for the speaker-diarization module. The resident
/// model's class (`DiarizationBackend`) decides which method is valid:
/// end-to-end Sortformer answers `diarize`; a pyannote segmentation model
/// answers `segment` (its output feeds the route's embed→cluster pipeline,
/// ADR 018). Calling the wrong one for the resident backend throws
/// `AthenaError.diarizationMethodInvalid`.
public protocol DiarizationModule: InferenceModule {
    /// "Who spoke when" over `audio` (raw uploaded file bytes;
    /// `filename` hints the container). End-to-end backends (Sortformer) only.
    func diarize(
        audio: Data, filename: String?
    ) async throws -> DiarizationResult

    /// Per-window locally-active speaker regions from a learned segmentation
    /// model (pyannote pipeline). Segmentation backends only.
    func segment(
        audio: Data, filename: String?
    ) async throws -> [SpeakerActivityRegion]

    /// Backend family of the currently-resident model (ADR 018), so the serve
    /// path can match the request `method` and emit a clear cause-naming error
    /// on a mismatch before doing any work.
    func residentBackend() async -> DiarizationBackend
}

/// `--engine stub` placeholder: a single canned turn so the endpoint
/// and governor wiring/accounting work without a model.
public actor StubDiarizationModule: DiarizationModule, ModelSelectable {
    public nonisolated let id: ModuleID = .diarization
    public nonisolated var moduleID: ModuleID { .diarization }

    private let reserveBytes: Int
    private let modelIds: [String]
    private let configuredDefault: String?
    private var residentId: String?

    public init(
        reserveBytes: Int = 512 * 1024 * 1024,
        modelIds: [String] = ["athena-stub-diarization"],
        configuredDefault: String? = nil
    ) {
        precondition(
            !modelIds.isEmpty,
            "StubDiarizationModule needs at least one model id")
        self.reserveBytes = reserveBytes
        self.modelIds = modelIds
        self.configuredDefault =
            (configuredDefault?.isEmpty == true) ? nil : configuredDefault
    }

    public var residentBytes: Int { residentId == nil ? 0 : reserveBytes }
    public func memoryEstimate() -> Int { reserveBytes }
    public func load(reservation: MemoryReservation) async throws {
        if residentId == nil { residentId = try resolve(nil) }
    }
    public func unload() async { residentId = nil }

    public func allowedModelIds() -> [String] { modelIds }
    public func defaultModelId() -> String {
        ModelSelection.displayDefault(
            available: modelIds, configuredDefault: configuredDefault)
    }
    public func residentModelId() -> String? { residentId }
    public func rebind(to id: String?) async throws {
        residentId = try resolve(id)
    }

    /// ADR 026 resolution against the injected stub set.
    private func resolve(_ id: String?) throws -> String {
        switch ModelSelection.resolve(
            available: modelIds, configuredDefault: configuredDefault,
            requested: id)
        {
        case .resolved(let t): return t
        case .notAvailable:
            throw AthenaError.modelNotAvailable(
                requested: id ?? (configuredDefault ?? ""),
                available: modelIds)
        case .ambiguous:
            throw AthenaError.ambiguousModel(
                module: .diarization, available: modelIds)
        }
    }

    public func diarize(
        audio: Data, filename: String?
    ) async throws -> DiarizationResult {
        DiarizationResult(
            turns: [DiarizationTurn(start: 0, end: 0, speaker: 0)],
            numSpeakers: 1)
    }

    /// The stub emulates an end-to-end Sortformer-class engine.
    public func residentBackend() async -> DiarizationBackend { .sortformer }

    /// The stub is an end-to-end placeholder, not a segmentation model, so it
    /// has no `segment` path — `method=pyannote` against `--engine stub` is a
    /// method/model mismatch.
    public func segment(
        audio: Data, filename: String?
    ) async throws -> [SpeakerActivityRegion] {
        throw AthenaError.diarizationMethodInvalid(
            method: "pyannote",
            reason: "the stub diarization engine has no segmentation backend")
    }
}
