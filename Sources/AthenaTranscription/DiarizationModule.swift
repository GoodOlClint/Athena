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
public actor StubDiarizationModule: DiarizationModule {
    public nonisolated let id: ModuleID = .diarization

    private let reserveBytes: Int
    private var loaded = false

    public init(reserveBytes: Int = 512 * 1024 * 1024) {
        self.reserveBytes = reserveBytes
    }

    public var residentBytes: Int { loaded ? reserveBytes : 0 }
    public func memoryEstimate() -> Int { reserveBytes }
    public func load(reservation: MemoryReservation) async throws {
        loaded = true
    }
    public func unload() async { loaded = false }

    public func diarize(
        audio: Data, filename: String?
    ) async throws -> DiarizationResult {
        DiarizationResult(
            turns: [DiarizationTurn(start: 0, end: 0, speaker: 0)],
            numSpeakers: 1)
    }
}
