import AthenaCore
import Foundation
import MLXLMCommon

/// The real MLX-backed speech-to-text module (M4.2d). Loads the ported
/// Whisper model + its tokenizer and runs the M4.2c greedy decode over
/// a single 30 s log-mel window.
///
/// Memory accounting is a fixed pre-load estimate (weights aren't on
/// disk until first download); live reconciliation is the shared M5
/// follow-up. The HF cache root follows `HF_HOME` (SSD-or-local — set
/// by the serve entrypoint), so this module stays location-agnostic.
public actor MLXTranscriptionModule: TranscriptionModule, ModelSelectable {
    public nonisolated let id: ModuleID = .transcription
    public nonisolated var moduleID: ModuleID { .transcription }

    private let modelId: String
    private let estimatedBytes: Int
    private var model: WhisperModel?
    private var tokenizer: (any MLXLMCommon.Tokenizer)?
    /// nil ⇒ unloaded; otherwise the id resident in `model`. M41.1
    /// single-id allowlist; M41.3 generalizes to a repeatable
    /// `--whisper-model` set with per-request rebind.
    private var residentId: String?

    /// - Parameters:
    ///   - modelId: HF id (default `mlx-community/whisper-large-v3-turbo`).
    ///   - estimatedBytes: governor admission estimate. Default 3 GiB:
    ///     headroom over the F16 weights (~1.6 GB) + tokenizer + the
    ///     encoder/decoder activation working set.
    public init(
        modelId: String = "mlx-community/whisper-large-v3-turbo",
        estimatedBytes: Int = 3 * 1024 * 1024 * 1024
    ) {
        self.modelId = modelId
        self.estimatedBytes = estimatedBytes
    }

    public var residentBytes: Int { model == nil ? 0 : estimatedBytes }

    public func memoryEstimate() -> Int { estimatedBytes }

    public func load(reservation: MemoryReservation) async throws {
        if model != nil { return }
        do {
            model = try await WhisperLoader.load(modelId: modelId)
            tokenizer = try await WhisperLoader.loadTokenizer()
            residentId = modelId
        } catch {
            throw AthenaError.moduleLoadFailed(
                .transcription,
                reason: "whisper \(modelId): \(error)")
        }
    }

    public func unload() async {
        model = nil
        tokenizer = nil
        residentId = nil
    }

    // M41 — ModelSelectable. M41.1 single-id allowlist; M41.3
    // generalizes to a repeatable `--whisper-model` set.
    public func allowedModelIds() -> [String] { [modelId] }
    public func defaultModelId() -> String { modelId }
    public func residentModelId() -> String? { residentId }
    public func rebind(to id: String?) async throws {
        let target = id ?? modelId
        guard target == modelId else {
            throw AthenaError.modelNotAvailable(
                requested: target, available: [modelId])
        }
    }

    public func transcribe(
        audio: Data, filename: String?, language: String?,
        wordTimestamps: Bool
    ) async throws -> TranscriptionResult {
        guard let model, let tokenizer else {
            throw AthenaError.moduleLoadFailed(
                .transcription, reason: "transcribe called before load")
        }
        // AVFoundation reads from a URL — stage the upload to a temp
        // file (keep the client's extension so the container is
        // detected) and always clean it up.
        let ext = (filename as NSString?)?.pathExtension ?? ""
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent(
                "athena-stt-\(UUID().uuidString)"
                    + (ext.isEmpty ? "" : ".\(ext)"))
        try audio.write(to: tmp)
        defer { try? FileManager.default.removeItem(at: tmp) }

        let pcm = try AudioDecode.pcm16kMono(from: tmp)
        // PCM-level entry: >30 s is chunked into 30 s windows. nil
        // language ⇒ Whisper auto-detects (once, on window 0). Result
        // carries timed segments for verbose_json/srt/vtt.
        return WhisperDecode.transcribeResult(
            model: model, pcm: pcm, tokenizer: tokenizer,
            language: language, wordTimestamps: wordTimestamps)
    }
}
