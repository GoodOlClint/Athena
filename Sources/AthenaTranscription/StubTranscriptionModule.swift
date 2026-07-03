import AthenaCore
import Foundation

/// Typed inference contract for the speech-to-text module. The serve
/// path holds this; the MLX-backed impl is `MLXTranscriptionModule`.
public protocol TranscriptionModule: InferenceModule {
    /// Transcribe `audio` (raw uploaded file bytes; `filename` hints the
    /// container). `language` is an ISO code (nil/"auto" ⇒ detect).
    /// `wordTimestamps` adds cross-attention DTW word alignment (M26.2)
    /// — only requested for the `verbose_json` response.
    /// `requestedModel` (ADR 029 residual) is re-bound INSIDE the gated
    /// execution span so a concurrent different-model rebind can't swap the
    /// shared slot between the server's preflight rebind and this forward —
    /// the same H3 wrong-model window WP6 closed for the LLM path.
    func transcribe(
        audio: Data, filename: String?, language: String?,
        wordTimestamps: Bool, requestedModel: String?
    ) async throws -> TranscriptionResult

    /// Transcribe already-decoded 16 kHz mono PCM — the shared engine entry
    /// (ADR 022 M78.1 S2). The audio route reaches it after `AudioDecode`, the
    /// video route after `VideoAudioTrack.extractPCM`; both share one dispatch
    /// with no re-encode.
    func transcribePCM(
        _ pcm: [Float], language: String?, wordTimestamps: Bool,
        requestedModel: String?
    ) async throws -> TranscriptionResult
}

/// M0 placeholder, still used by `--engine stub`. Returns a fixed
/// string so `/v1/audio/transcriptions` is demoable without a model and
/// the governor wiring/budget accounting are exercised.
public actor StubTranscriptionModule: TranscriptionModule, ModelSelectable {
    public nonisolated let id: ModuleID = .transcription
    public nonisolated var moduleID: ModuleID { .transcription }

    private let reserveBytes: Int
    private let modelIds: [String]
    private let configuredDefault: String?
    private var residentId: String?

    public init(
        reserveBytes: Int = 2 * 1024 * 1024 * 1024,
        modelIds: [String] = ["athena-stub-whisper"],
        configuredDefault: String? = nil
    ) {
        precondition(
            !modelIds.isEmpty,
            "StubTranscriptionModule needs at least one model id")
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
                module: .transcription, available: modelIds)
        }
    }

    public func transcribe(
        audio: Data, filename: String?, language: String?,
        wordTimestamps: Bool, requestedModel: String? = nil
    ) async throws -> TranscriptionResult {
        try await transcribePCM(
            [], language: language, wordTimestamps: wordTimestamps,
            requestedModel: requestedModel)
    }

    public func transcribePCM(
        _ pcm: [Float], language: String?, wordTimestamps: Bool,
        requestedModel: String? = nil
    ) async throws -> TranscriptionResult {
        // Same atomic rebind-then-serve as the MLX module, so the stub tier
        // exercises the decision behavior (ADR 009).
        if let m = requestedModel, !m.isEmpty { try await rebind(to: m) }
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
