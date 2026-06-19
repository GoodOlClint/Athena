import AthenaCore
import Foundation
import MLXLMCommon

/// The real MLX-backed speech-to-text module. A single governed slot (ADR 011)
/// whose allowlist spans two engine families (ADR 020): OpenAI **Whisper**
/// (the default) and NVIDIA **Parakeet-TDT**. `TranscriptionArch` classifies
/// the resident checkpoint from its `config.json` and `loadModel` instantiates
/// the matching engine; `transcribe` dispatches on the loaded engine. Whisper
/// runs the M4.2c greedy decode over 30 s log-mel windows; Parakeet runs the
/// FastConformer encoder + greedy TDT decode (the hardened ADR-019 port).
///
/// Memory accounting is a fixed pre-load estimate sized to cover either engine
/// (Whisper F16 ≈ 1.6 GB, Parakeet fp ≈ 2.4 GB); one model is resident at a
/// time, so the fixed estimate bounds the slot, with live reconciliation the
/// shared M5 follow-up. The HF cache root follows `HF_HOME` (SSD-or-local — set
/// by the serve entrypoint), so this module stays location-agnostic.
public actor MLXTranscriptionModule: TranscriptionModule, ModelSelectable {
    public nonisolated let id: ModuleID = .transcription
    public nonisolated var moduleID: ModuleID { .transcription }

    /// The loaded inference engine for the resident checkpoint (ADR 020). The
    /// resident model's class (`TranscriptionArch`) decides which case is built
    /// at load; `transcribe` dispatches on it.
    private enum Engine {
        case whisper(WhisperModel, any MLXLMCommon.Tokenizer)
        case parakeet(ParakeetTDTModel)
    }

    /// M41.3 operator-declared allowlist (HF ids; the substrate's HF
    /// cache is the load source). First-declared = the default.
    /// M42.2: mutable so the persistent DB allowlist can be pushed in
    /// at runtime without a daemon restart.
    private var allowedIds: [String]
    private var defaultId: String
    private let estimatedBytes: Int
    private var engine: Engine?
    /// nil ⇒ unloaded; otherwise the id resident in `engine`.
    private var residentId: String?
    /// Model-store root (M54) — load from the local store dir when the
    /// model is materialized there, so a bare store-dir name or full HF
    /// id both work, like the LLM loader. nil ⇒ Hub-by-id fallback.
    private let modelStoreRoot: URL?

    /// - Parameters:
    ///   - modelIds: HF id allowlist (first = default). Default
    ///     `[mlx-community/whisper-large-v3-turbo]`. M41.3.
    ///   - estimatedBytes: governor admission estimate. Default 3 GiB:
    ///     headroom over the larger engine's weights + tokenizer + the
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

    public var residentBytes: Int { engine == nil ? 0 : estimatedBytes }

    public func memoryEstimate() -> Int { estimatedBytes }

    public func load(reservation: MemoryReservation) async throws {
        if engine != nil { return }
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
        // ADR 021 — one predicate decides BOTH modality and loadability. The
        // single governed slot spans Whisper and Parakeet; a checkpoint that
        // classifies as one of them by family but whose PACKAGING the loader
        // cannot read — a non-large-v3 Whisper vocab, or a transformers-format
        // Parakeet with no `joint.vocabulary` (the M76 field incident) — is
        // refused here with a cause-naming 4xx naming the structural
        // requirement, instead of dispatching into a loader that fails deep
        // with an opaque 500. This subsumes the old `TranscriptionArch.detect`
        // switch and the post-load whisper-vocab guard (kept below as a
        // belt-and-braces backstop). The guidance is repo-id-free (ADR 021 D5).
        let resolved = dir.resolvingSymlinksInPath()
        let support = ModelSupport.detect(in: resolved)
        if case let .unsupported(reason, guidance) = support.loadability {
            throw AthenaError.unsupportedTranscriptionArch(
                model: canonical, detail: "\(reason); \(guidance)")
        }
        switch support.modality {
        case .transcription(.whisper):
            try await loadWhisper(canonical: canonical, dir: resolved)
        case .transcription(.parakeet):
            try loadParakeet(canonical: canonical, dir: resolved)
        default:
            throw AthenaError.unsupportedTranscriptionArch(
                model: canonical,
                detail: "the checkpoint is neither a Whisper nor a Parakeet "
                    + "transcription model; the transcription slot serves only "
                    + "those two engine families")
        }
    }

    private func loadWhisper(canonical: String, dir: URL) async throws {
        let model: WhisperModel
        let tokenizer: any MLXLMCommon.Tokenizer
        do {
            model = try WhisperLoader.load(directory: dir)
            tokenizer = try await WhisperLoader.loadTokenizer()
        } catch {
            engine = nil
            residentId = nil
            throw AthenaError.moduleLoadFailed(
                .transcription,
                reason: "whisper \(canonical): \(error)")
        }
        // ND2: the decoder's special-token ids (eot/sot/langBase/transcribe/
        // timestampBegin) are pinned to the whisper-large-v3 family
        // (vocab 51866). A non-v3 checkpoint (e.g. v1/v2/medium, vocab
        // 51865/51864) shifts every special id from `transcribe` onward by
        // one (v3 inserted Cantonese), so the forced prefix, the eot stop
        // test, the special mask and timestamp parsing would all target the
        // wrong tokens — silently mis-decoding. Fail loud at load instead.
        if model.config.n_vocab != 51866 {
            let bad = model.config.n_vocab
            engine = nil
            residentId = nil
            throw AthenaError.moduleLoadFailed(
                .transcription,
                reason: "whisper model '\(canonical)' has vocab \(bad); "
                    + "Athena's decoder is pinned to the large-v3 family "
                    + "(vocab 51866) — use a large-v3 / large-v3-turbo "
                    + "checkpoint")
        }
        engine = .whisper(model, tokenizer)
        residentId = canonical
    }

    private func loadParakeet(canonical: String, dir: URL) throws {
        do {
            let model = try ParakeetLoader.fromModelDirectory(dir)
            engine = .parakeet(model)
            residentId = canonical
        } catch {
            engine = nil
            residentId = nil
            throw AthenaError.moduleLoadFailed(
                .transcription,
                reason: "parakeet \(canonical): \(error)")
        }
    }

    public func unload() async {
        engine = nil
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
        if residentId == target, engine != nil { return }
        engine = nil
        residentId = nil
        try await loadModel(id: target)
    }

    public func setAllowedModelIds(_ ids: [String]) {
        allowedIds = ids
        defaultId = ids.first ?? defaultId
        // ND5: evict only when the resident id is no longer allowed under
        // the SAME store-identity resolver that load/rebind use. A plain
        // `!ids.contains(r)` is a stricter case-sensitive full-string match,
        // so an M42 live refresh re-supplying the resident model under an
        // equivalent spelling (bare vs full HF id, case diff) would needlessly
        // drop the loaded model+tokenizer and force a multi-GB reload.
        if let r = residentId, ids.canonicalByStoreIdentity(r) == nil {
            engine = nil
            residentId = nil
        }
    }

    public func transcribe(
        audio: Data, filename: String?, language: String?,
        wordTimestamps: Bool
    ) async throws -> TranscriptionResult {
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

        let pcm = try AudioDecode.pcm16kMono(from: tmp, module: .transcription)
        return try transcribePCM(
            pcm, language: language, wordTimestamps: wordTimestamps)
    }

    /// Transcribe already-decoded 16 kHz mono PCM — the shared engine entry
    /// (ADR 022 M78.1 S2). The audio route reaches it after `AudioDecode`; the
    /// video route reaches it after `VideoAudioTrack.extractPCM`, so both share
    /// the identical Whisper/Parakeet dispatch, chunking, and timestamping with
    /// no re-encode. Actor-isolated (reads `engine`); callers `await` it.
    public func transcribePCM(
        _ pcm: [Float], language: String?, wordTimestamps: Bool
    ) throws -> TranscriptionResult {
        guard let engine else {
            throw AthenaError.moduleLoadFailed(
                .transcription, reason: "transcribe called before load")
        }
        switch engine {
        case .whisper(let model, let tokenizer):
            // PCM-level entry: >30 s is chunked into 30 s windows. nil
            // language ⇒ Whisper auto-detects (once, on window 0). Result
            // carries timed segments for verbose_json/srt/vtt.
            return WhisperDecode.transcribeResult(
                model: model, pcm: pcm, tokenizer: tokenizer,
                language: language, wordTimestamps: wordTimestamps)
        case .parakeet(let model):
            // Whole-file greedy TDT decode. Segment/word timestamps come from
            // the TDT durations (0.08 s/encoder frame) via the MLX-free
            // `ParakeetAlignment` grouping (S3); long-audio chunking lands in
            // S4. Parakeet greedy decode does not surface a detected language
            // (no forced language prompt, unlike Whisper), so the response
            // echoes the requested `language` or "auto".
            let r = model.transcribe(pcm)
            let text = r.transcript.trimmingCharacters(
                in: .whitespacesAndNewlines)
            let duration = Double(pcm.count) / 16000.0
            var segments = ParakeetAlignment.segments(
                from: r.tokens, attachWords: wordTimestamps)
            if segments.isEmpty {
                // Edge: speech with no sentence-final punctuation → one span.
                segments = [
                    TranscriptionSegment(start: 0, end: duration, text: text)
                ]
            }
            let words =
                wordTimestamps ? ParakeetAlignment.words(from: r.tokens) : []
            return TranscriptionResult(
                text: text, language: language ?? "auto", duration: duration,
                segments: segments, words: words)
        }
    }
}
