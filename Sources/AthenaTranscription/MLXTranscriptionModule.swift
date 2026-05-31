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

    /// M41.3 operator-declared allowlist (HF ids; the substrate's HF
    /// cache is the load source). First-declared = the default.
    /// M42.2: mutable so the persistent DB allowlist can be pushed in
    /// at runtime without a daemon restart.
    private var allowedIds: [String]
    private var defaultId: String
    private let estimatedBytes: Int
    private var model: WhisperModel?
    private var tokenizer: (any MLXLMCommon.Tokenizer)?
    /// nil ⇒ unloaded; otherwise the id resident in `model`.
    private var residentId: String?
    /// Model-store root (M54) — load from the local store dir when the
    /// model is materialized there, so a bare store-dir name or full HF
    /// id both work, like the LLM loader. nil ⇒ Hub-by-id fallback.
    private let modelStoreRoot: URL?

    /// - Parameters:
    ///   - modelIds: HF id allowlist (first = default). Default
    ///     `[mlx-community/whisper-large-v3-turbo]`. M41.3.
    ///   - estimatedBytes: governor admission estimate. Default 3 GiB:
    ///     headroom over the F16 weights (~1.6 GB) + tokenizer + the
    ///     encoder/decoder activation working set. One model is
    ///     resident at a time, so the fixed estimate bounds the slot.
    public init(
        modelIds: [String] = ["mlx-community/whisper-large-v3-turbo"],
        modelStoreRoot: URL? = nil,
        estimatedBytes: Int = 3 * 1024 * 1024 * 1024
    ) {
        precondition(
            !modelIds.isEmpty,
            "MLXTranscriptionModule needs at least one model id")
        self.allowedIds = modelIds
        self.defaultId = modelIds[0]
        self.modelStoreRoot = modelStoreRoot
        self.estimatedBytes = estimatedBytes
    }

    /// Source-compat init for callers that still pass a single id.
    public init(
        modelId: String,
        estimatedBytes: Int = 3 * 1024 * 1024 * 1024
    ) {
        self.init(modelIds: [modelId], estimatedBytes: estimatedBytes)
    }

    public var residentBytes: Int { model == nil ? 0 : estimatedBytes }

    public func memoryEstimate() -> Int { estimatedBytes }

    public func load(reservation: MemoryReservation) async throws {
        if model != nil { return }
        try await loadModel(id: residentId ?? defaultId)
    }

    private func loadModel(id: String) async throws {
        // M54 — match by store-dir identity (bare name or full HF id).
        guard let canonical =
            allowedIds.canonicalByStoreIdentity(id)
        else {
            throw AthenaError.modelNotAvailable(
                requested: id, available: allowedIds)
        }
        // M54.3 — load from the local store dir; inference never
        // auto-downloads (operator pulls a missing model at startup /
        // allowlist-add). WhisperLoader also seeds the alignment_heads
        // for M26 word timestamps; an unload+load on rebind picks up the
        // new model's heads.
        guard let dir = ModelStoreLayout.localDirectory(
            for: canonical, storeRoot: modelStoreRoot)
        else {
            throw AthenaError.moduleLoadFailed(
                .transcription,
                reason: "model '\(canonical)' is not in the model store "
                    + "— pull it first (operator action); inference does "
                    + "not auto-download")
        }
        do {
            model = try WhisperLoader.load(
                directory: dir.resolvingSymlinksInPath())
            tokenizer = try await WhisperLoader.loadTokenizer()
            residentId = canonical
        } catch {
            model = nil
            tokenizer = nil
            residentId = nil
            throw AthenaError.moduleLoadFailed(
                .transcription,
                reason: "whisper \(id): \(error)")
        }
    }

    public func unload() async {
        model = nil
        tokenizer = nil
        residentId = nil
    }

    // M41 — ModelSelectable. M41.3 generalizes to a repeatable
    // `--whisper-model` allowlist; rebind unloads + reloads in place
    // under the same governor reservation.
    public func allowedModelIds() -> [String] { allowedIds }
    public func defaultModelId() -> String { defaultId }
    public func residentModelId() -> String? { residentId }
    public func rebind(to id: String?) async throws {
        let requested = id ?? defaultId
        // M54 — match by store-dir identity (bare name or full HF id).
        guard let target =
            allowedIds.canonicalByStoreIdentity(requested)
        else {
            throw AthenaError.modelNotAvailable(
                requested: requested, available: allowedIds)
        }
        if residentId == target, model != nil { return }
        model = nil
        tokenizer = nil
        residentId = nil
        try await loadModel(id: target)
    }

    public func setAllowedModelIds(_ ids: [String]) {
        allowedIds = ids
        defaultId = ids.first ?? defaultId
        if let r = residentId, !ids.contains(r) {
            model = nil
            tokenizer = nil
            residentId = nil
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
