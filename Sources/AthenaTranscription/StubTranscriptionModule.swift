import AthenaCore
import Foundation

/// M0 placeholder. The WhisperKit-backed implementation lands in M4; this
/// stub exists so the module target and governor wiring are in place and the
/// global budget already accounts for transcription.
public actor StubTranscriptionModule: InferenceModule {
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
}
