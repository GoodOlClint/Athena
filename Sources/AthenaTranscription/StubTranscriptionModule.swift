import AthenaCore
import Foundation

/// Typed inference contract for the speech-to-text module. The serve
/// path holds this; the MLX-backed impl is `MLXTranscriptionModule`.
public protocol TranscriptionModule: InferenceModule {
    /// Transcribe `audio` (raw uploaded file bytes; `filename` hints the
    /// container) → text. `language` is an ISO code (nil ⇒ default en).
    func transcribe(
        audio: Data, filename: String?, language: String?
    ) async throws -> String
}

/// M0 placeholder, still used by `--engine stub`. Returns a fixed
/// string so `/v1/audio/transcriptions` is demoable without a model and
/// the governor wiring/budget accounting are exercised.
public actor StubTranscriptionModule: TranscriptionModule {
    public nonisolated let id: ModuleID = .transcription

    private let reserveBytes: Int
    private var loaded = false

    public init(reserveBytes: Int = 2 * 1024 * 1024 * 1024) {
        self.reserveBytes = reserveBytes
    }

    public var residentBytes: Int { loaded ? reserveBytes : 0 }
    public func memoryEstimate() -> Int { reserveBytes }
    public func load(reservation: MemoryReservation) async throws { loaded = true }
    public func unload() async { loaded = false }

    public func transcribe(
        audio: Data, filename: String?, language: String?
    ) async throws -> String {
        "[athena stub transcription]"
    }
}
