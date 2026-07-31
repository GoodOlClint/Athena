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

    /// ADR 026 — the selectable set is the store's transcription models
    /// (Whisper/Parakeet), scanned live via `ModelSupport`; `configuredDefault`
    /// is the per-module TOML default (nil ⇒ ambiguity rule). The substrate's
    /// HF cache is the load source via the store dir.
    private let configuredDefault: String?
    private let estimatedBytes: Int
    private var engine: Engine?
    /// nil ⇒ unloaded; otherwise the id resident in `engine`.
    private var residentId: String?
    /// Model-store root (M54) — load from the local store dir when the
    /// model is materialized there, so a bare store-dir name or full HF
    /// id both work, like the LLM loader. nil ⇒ Hub-by-id fallback.
    private let modelStoreRoot: URL?

    /// - Parameters:
    ///   - configuredDefault: the per-module TOML default transcription id
    ///     (ADR 026), used when a request omits `model` (nil ⇒ ambiguity rule).
    ///   - modelStoreRoot: the store root scanned for transcription models.
    ///   - estimatedBytes: governor admission estimate. Default 3 GiB:
    ///     headroom over the larger engine's weights + tokenizer + the
    ///     encoder/decoder activation working set. One model is
    ///     resident at a time, so the fixed estimate bounds the slot.
    public init(
        configuredDefault: String? = nil,
        modelStoreRoot: URL? = nil,
        estimatedBytes: Int = 3 * 1024 * 1024 * 1024
    ) {
        self.configuredDefault =
            (configuredDefault?.isEmpty == true) ? nil : configuredDefault
        self.modelStoreRoot = modelStoreRoot
        self.estimatedBytes = estimatedBytes
    }

    public var residentBytes: Int { engine == nil ? 0 : estimatedBytes }

    public func memoryEstimate() -> Int { estimatedBytes }

    /// The store's transcription models (ADR 026 live scan).
    private func storeModelIds() -> [String] {
        StoreModelClass.ids(
            storeRoot: modelStoreRoot, accept: { $0.isTranscriptionSlot })
    }

    /// ADR 026 resolution against the live store scan (rebind/load).
    private func resolve(_ id: String?) throws -> String {
        let available = storeModelIds()
        switch ModelSelection.resolve(
            available: available, configuredDefault: configuredDefault,
            requested: id)
        {
        case .resolved(let t): return t
        case .notAvailable:
            throw AthenaError.modelNotAvailable(
                requested: id ?? (configuredDefault ?? ""),
                available: available)
        case .ambiguous:
            throw AthenaError.ambiguousModel(
                module: .transcription, available: available)
        }
    }

    public func load(reservation: MemoryReservation) async throws {
        if engine != nil { return }
        try await loadModel(id: residentId ?? resolve(nil))
    }

    private func loadModel(id canonical: String) async throws {
        // M54.3 — load from the local store dir; inference never
        // auto-downloads (operator pulls a missing model at startup).
        // WhisperLoader also seeds the alignment_heads
        // for M26 word timestamps; an unload+load on rebind picks up the
        // new model's heads.
        guard
            let dir = ModelStoreLayout.localDirectory(
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

    // M41 / ADR 026 — ModelSelectable over the store-classified set; rebind
    // unloads + reloads in place under the same governor reservation.
    public func allowedModelIds() -> [String] { storeModelIds() }
    public func defaultModelId() -> String {
        ModelSelection.displayDefault(
            available: storeModelIds(), configuredDefault: configuredDefault)
    }
    public func residentModelId() -> String? { residentId }
    public func rebind(to id: String?) async throws {
        let target = try resolve(id)
        if residentId == target, engine != nil { return }
        engine = nil
        residentId = nil
        try await loadModel(id: target)
    }

    public func transcribe(
        audio: Data, filename: String?, language: String?,
        wordTimestamps: Bool, requestedModel: String? = nil
    ) async throws -> TranscriptionResult {
        // Option D (ADR 025 S5): decode from the in-memory upload bytes — no
        // temp file. `filename` carries the container hint for the decoder.
        var pcm = try await AudioDecode.pcm16kMono(
            from: audio, filename: filename, module: .transcription)
        // ADR 024 T2: best-effort zeroize the decoded speech PCM on the way out.
        defer { ProcessHardening.secureZero(&pcm) }
        return try await transcribePCM(
            pcm, language: language, wordTimestamps: wordTimestamps,
            requestedModel: requestedModel)
    }

    /// Transcribe already-decoded 16 kHz mono PCM — the shared engine entry
    /// (ADR 022 M78.1 S2). The audio route reaches it after `AudioDecode`; the
    /// video route reaches it after `VideoAudioTrack.extractPCM`, so both share
    /// the identical Whisper/Parakeet dispatch, chunking, and timestamping with
    /// no re-encode. Actor-isolated (reads `engine`); callers `await` it.
    public func transcribePCM(
        _ pcm: [Float], language: String?, wordTimestamps: Bool,
        requestedModel: String? = nil
    ) async throws -> TranscriptionResult {
        // ADR 029 — gate the Metal forward against other tenants on the one
        // Metal pool. The actor-isolated worker runs inside the gate via a
        // self-hop so the @Sendable closure never captures actor state.
        try await InferenceGate.shared.withExclusiveExecution {
            // ADR 029 residual (audio) — bind the requested model INSIDE the
            // gate so a concurrent different-model rebind can't swap the slot
            // between the server's preflight rebind and this forward. No-op
            // when already bound; `decodePCM` reads `engine` after this.
            if let m = requestedModel, !m.isEmpty {
                try await self.rebind(to: m)
            }
            return try await self.decodePCM(
                pcm, language: language, wordTimestamps: wordTimestamps)
        }
    }

    private func decodePCM(
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
