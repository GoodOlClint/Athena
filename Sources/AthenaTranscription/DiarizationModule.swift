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

/// Typed inference contract for the speaker-diarization module
/// (vendored Sortformer). The serve path holds this.
public protocol DiarizationModule: InferenceModule {
    /// "Who spoke when" over `audio` (raw uploaded file bytes;
    /// `filename` hints the container).
    func diarize(
        audio: Data, filename: String?
    ) async throws -> DiarizationResult
}

/// `--engine stub` placeholder: a single canned turn so the endpoint
/// and governor wiring/accounting work without a model.
public actor StubDiarizationModule: DiarizationModule, ModelSelectable {
    public nonisolated let id: ModuleID = .diarization
    public nonisolated var moduleID: ModuleID { .diarization }

    private let reserveBytes: Int
    private var modelIds: [String]
    private var defaultId: String
    private var residentId: String?

    public init(
        reserveBytes: Int = 512 * 1024 * 1024,
        modelIds: [String] = ["athena-stub-diarization"]
    ) {
        precondition(
            !modelIds.isEmpty,
            "StubDiarizationModule needs at least one model id")
        self.reserveBytes = reserveBytes
        self.modelIds = modelIds
        self.defaultId = modelIds[0]
    }

    public var residentBytes: Int { residentId == nil ? 0 : reserveBytes }
    public func memoryEstimate() -> Int { reserveBytes }
    public func load(reservation: MemoryReservation) async throws {
        if residentId == nil { residentId = defaultId }
    }
    public func unload() async { residentId = nil }

    public func allowedModelIds() -> [String] { modelIds }
    public func defaultModelId() -> String { defaultId }
    public func residentModelId() -> String? { residentId }
    public func rebind(to id: String?) async throws {
        let target = id ?? defaultId
        guard modelIds.contains(target) else {
            throw AthenaError.modelNotAvailable(
                requested: target, available: modelIds)
        }
        residentId = target
    }

    public func setAllowedModelIds(_ ids: [String]) {
        modelIds = ids
        defaultId = ids.first ?? defaultId
        if let r = residentId, !ids.contains(r) { residentId = nil }
    }

    public func diarize(
        audio: Data, filename: String?
    ) async throws -> DiarizationResult {
        DiarizationResult(
            turns: [DiarizationTurn(start: 0, end: 0, speaker: 0)],
            numSpeakers: 1)
    }
}
