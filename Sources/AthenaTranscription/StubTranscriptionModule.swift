import AthenaCore
import Foundation

/// Typed inference contract for the speech-to-text module. The serve
/// path holds this; the MLX-backed impl is `MLXTranscriptionModule`.
public protocol TranscriptionModule: InferenceModule {
    /// Transcribe `audio` (raw uploaded file bytes; `filename` hints the
    /// container). `language` is an ISO code (nil/"auto" ⇒ detect).
    /// `wordTimestamps` adds cross-attention DTW word alignment (M26.2)
    /// — only requested for the `verbose_json` response.
    func transcribe(
        audio: Data, filename: String?, language: String?,
        wordTimestamps: Bool
    ) async throws -> TranscriptionResult

    /// Transcribe already-decoded 16 kHz mono PCM — the shared engine entry
    /// (ADR 022 M78.1 S2). The audio route reaches it after `AudioDecode`, the
    /// video route after `VideoAudioTrack.extractPCM`; both share one dispatch
    /// with no re-encode.
    func transcribePCM(
        _ pcm: [Float], language: String?, wordTimestamps: Bool
    ) async throws -> TranscriptionResult
}

/// M0 placeholder, still used by `--engine stub`. Returns a fixed
/// string so `/v1/audio/transcriptions` is demoable without a model and
/// the governor wiring/budget accounting are exercised.
public actor StubTranscriptionModule: TranscriptionModule, ModelSelectable {
    public nonisolated let id: ModuleID = .transcription
    public nonisolated var moduleID: ModuleID { .transcription }

    private let reserveBytes: Int
    private var modelIds: [String]
    private var defaultId: String
    private var residentId: String?

    public init(
        reserveBytes: Int = 2 * 1024 * 1024 * 1024,
        modelIds: [String] = ["athena-stub-whisper"]
    ) {
        precondition(
            !modelIds.isEmpty,
            "StubTranscriptionModule needs at least one model id")
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
        let requested = id ?? defaultId
        // M46.4 — case-insensitive lookup; canonical id from storage.
        guard let target =
            modelIds.canonicalCaseInsensitive(requested)
        else {
            throw AthenaError.modelNotAvailable(
                requested: requested, available: modelIds)
        }
        residentId = target
    }

    public func setAllowedModelIds(_ ids: [String]) {
        modelIds = ids
        defaultId = ids.first ?? defaultId
        if let r = residentId, !ids.contains(r) { residentId = nil }
    }

    public func transcribe(
        audio: Data, filename: String?, language: String?,
        wordTimestamps: Bool
    ) async throws -> TranscriptionResult {
        try await transcribePCM(
            [], language: language, wordTimestamps: wordTimestamps)
    }

    public func transcribePCM(
        _ pcm: [Float], language: String?, wordTimestamps: Bool
    ) async throws -> TranscriptionResult {
        let words =
            wordTimestamps
            ? [
                WordTiming(
                    word: "[stub]", start: 0, end: 0, probability: 1)
            ] : []
        return TranscriptionResult(
            text: "[athena stub transcription]",
            language: language ?? "en", duration: 0,
            segments: [
                TranscriptionSegment(
                    start: 0, end: 0,
                    text: "[athena stub transcription]",
                    words: wordTimestamps ? words : nil)
            ],
            words: words)
    }
}
