import AthenaCore
import Foundation

/// The real MLX-backed speaker-embedding module (M25.1). Wraps the
/// vendored WeSpeaker ResNet34-LM. Decodes the upload once, slices each
/// requested segment, and embeds it to a 256-d L2-normalized voice
/// vector. Memory accounting is a fixed pre-load estimate (weights are
/// not on disk until first download); the M5 governor probe reconciles
/// to the real footprint. HF cache root follows `HF_HOME`
/// (SSD/local fallback), so this stays location-agnostic.
public actor MLXSpeakerEmbeddingModule: SpeakerEmbeddingModule,
    ModelSelectable
{
    public nonisolated let id: ModuleID = .speakerEmbedding
    public nonisolated var moduleID: ModuleID { .speakerEmbedding }

    /// ADR 026 — the selectable set is the store's speaker-embedding models
    /// (WeSpeaker), scanned live via `ModelSupport`; `configuredDefault` is the
    /// per-module TOML default (nil ⇒ ambiguity rule).
    private let configuredDefault: String?
    private let estimatedBytes: Int
    private var model: WeSpeakerModel?
    private var residentId: String?
    /// Model-store root (M54) — local-store-dir load when materialized,
    /// so bare name or full HF id both work. nil ⇒ Hub-by-id fallback.
    private let modelStoreRoot: URL?

    /// - Parameters:
    ///   - configuredDefault: the per-module TOML default speaker-embedding id
    ///     (ADR 026), used when a request omits `model` (nil ⇒ ambiguity rule).
    ///   - modelStoreRoot: the store root scanned for speaker-embedding models.
    ///   - estimatedBytes: governor admission estimate. 512 MiB: the
    ///     weights are tiny (~25 MB) but ResNet activations over a
    ///     long segment's mel grid dominate; conservative headroom that
    ///     M5 reconciles to the real footprint post-load.
    public init(
        configuredDefault: String? = nil,
        modelStoreRoot: URL? = nil,
        estimatedBytes: Int = 512 * 1024 * 1024
    ) {
        self.configuredDefault =
            (configuredDefault?.isEmpty == true) ? nil : configuredDefault
        self.modelStoreRoot = modelStoreRoot
        self.estimatedBytes = estimatedBytes
    }

    public var residentBytes: Int { model == nil ? 0 : estimatedBytes }

    public func memoryEstimate() -> Int { estimatedBytes }

    /// The store's speaker-embedding models (ADR 026 live scan).
    private func storeModelIds() -> [String] {
        StoreModelClass.ids(
            storeRoot: modelStoreRoot, accept: { $0.isSpeakerEmbeddingSlot })
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
                module: .speakerEmbedding, available: available)
        }
    }

    public func load(reservation: MemoryReservation) async throws {
        if model != nil { return }
        try await loadModel(id: residentId ?? resolve(nil))
    }

    private func loadModel(id canonical: String) async throws {
        // M54.3 — local-store-dir load only; inference never auto-downloads
        // (operator pulls a missing model at startup).
        guard let dir = ModelStoreLayout.localDirectory(
            for: canonical, storeRoot: modelStoreRoot)
        else {
            throw AthenaError.moduleLoadFailed(
                .speakerEmbedding,
                reason: "model '\(canonical)' is not in the model store "
                    + "— pull it first (operator action); inference does "
                    + "not auto-download")
        }
        do {
            model = try WeSpeakerModel.fromModelDirectory(
                dir.resolvingSymlinksInPath())
            residentId = canonical
        } catch {
            model = nil
            residentId = nil
            throw AthenaError.moduleLoadFailed(
                .speakerEmbedding,
                reason: "wespeaker \(canonical): \(error)")
        }
    }

    public func unload() async {
        model = nil
        residentId = nil
    }

    // M41 / ADR 026 — ModelSelectable over the store-classified set with
    // in-place rebind.
    public func allowedModelIds() -> [String] { storeModelIds() }
    public func defaultModelId() -> String {
        ModelSelection.displayDefault(
            available: storeModelIds(), configuredDefault: configuredDefault)
    }
    public func residentModelId() -> String? { residentId }
    public func rebind(to id: String?) async throws {
        let target = try resolve(id)
        if residentId == target, model != nil { return }
        model = nil
        residentId = nil
        try await loadModel(id: target)
    }

    public func embed(
        audio: Data, filename: String?, segments: [SpeakerSegmentRequest],
        requestedModel: String? = nil
    ) async throws -> SpeakerEmbeddingResult {
        // ADR 029 — gate the per-segment model.embed Metal forwards against
        // other tenants via an actor self-hop (the @Sendable closure never
        // captures the non-Sendable model).
        try await InferenceGate.shared.withExclusiveExecution {
            // ADR 029 residual (audio) — in-gate rebind; `embedWorker` reads
            // `model` after it.
            if let m = requestedModel, !m.isEmpty {
                try await self.rebind(to: m)
            }
            return try await self.embedWorker(
                audio: audio, filename: filename, segments: segments)
        }
    }

    private func embedWorker(
        audio: Data, filename: String?, segments: [SpeakerSegmentRequest]
    ) async throws -> SpeakerEmbeddingResult {
        guard let model else {
            throw AthenaError.moduleLoadFailed(
                .speakerEmbedding, reason: "embed called before load")
        }

        // Option D (ADR 025 S5): decode from the in-memory upload bytes.
        var pcm = try await AudioDecode.pcm16kMono(
            from: audio, filename: filename, module: .speakerEmbedding)
        defer { ProcessHardening.secureZero(&pcm) }  // ADR 024 T2
        let sr = Double(AudioDecode.sampleRate)
        let totalSeconds = Double(pcm.count) / sr

        // No segments ⇒ embed the whole clip as one segment.
        let reqs =
            segments.isEmpty
            ? [SpeakerSegmentRequest(start: 0, end: totalSeconds)]
            : segments

        var out: [SpeakerSegmentEmbedding] = []
        out.reserveCapacity(reqs.count)
        // Convert seconds → sample index with the clamp done in the
        // Double domain BEFORE the Int cast: `Int(Double)` traps on
        // non-finite/out-of-range input, so a tiny caller-supplied scalar
        // like end=1e30 would abort the whole daemon (ND1). After this,
        // the cast only ever runs on a value already in [0, pcm.count].
        func sampleIndex(_ seconds: Double) -> Int {
            let n = (seconds * sr).rounded()
            if n <= 0 { return 0 }
            if n >= Double(pcm.count) { return pcm.count }
            return Int(n)
        }
        for r in reqs {
            guard r.start.isFinite, r.end.isFinite else {
                throw AthenaError.audioSegmentInvalid(
                    module: .speakerEmbedding,
                    detail: "segment bounds must be finite seconds")
            }
            let s = sampleIndex(r.start)
            let e = sampleIndex(r.end)
            guard e > s else {
                // Classified 400 — never a silent zero embedding
                // (M24.4a precedent).
                throw AthenaError.audioSegmentInvalid(
                    module: .speakerEmbedding,
                    detail: String(
                        format:
                            "segment [%.3f, %.3f]s is empty or out of "
                            + "range for a %.3fs clip",
                        r.start, r.end, totalSeconds))
            }
            let slice = Array(pcm[s..<e])
            let vec = model.embed(slice)
            out.append(
                SpeakerSegmentEmbedding(
                    start: r.start, end: r.end, embedding: vec,
                    durationSeconds: Double(e - s) / sr))
        }
        return SpeakerEmbeddingResult(
            segments: out, dimension: model.embeddingDimension)
    }

    public func windowEmbeddings(
        audio: Data, filename: String?,
        windowSeconds: Double, hopSeconds: Double
    ) async throws -> SpeakerEmbeddingResult {
        // ADR 029 — gate the Metal forwards via an actor self-hop.
        try await InferenceGate.shared.withExclusiveExecution {
            try await self.windowEmbeddingsWorker(
                audio: audio, filename: filename,
                windowSeconds: windowSeconds, hopSeconds: hopSeconds)
        }
    }

    private func windowEmbeddingsWorker(
        audio: Data, filename: String?,
        windowSeconds: Double, hopSeconds: Double
    ) async throws -> SpeakerEmbeddingResult {
        guard let model else {
            throw AthenaError.moduleLoadFailed(
                .speakerEmbedding, reason: "embed called before load")
        }
        // Option D (ADR 025 S5): decode from the in-memory upload bytes.
        var pcm = try await AudioDecode.pcm16kMono(
            from: audio, filename: filename, module: .speakerEmbedding)
        defer { ProcessHardening.secureZero(&pcm) }  // ADR 024 T2
        let sr = Double(AudioDecode.sampleRate)
        let win = max(1, Int(windowSeconds * sr))
        let hop = max(1, Int(hopSeconds * sr))
        guard pcm.count >= win else {
            // Clip shorter than one window — embed the whole thing.
            let vec = model.embed(pcm)
            return SpeakerEmbeddingResult(
                segments: [
                    SpeakerSegmentEmbedding(
                        start: 0, end: Double(pcm.count) / sr,
                        embedding: vec,
                        durationSeconds: Double(pcm.count) / sr)
                ],
                dimension: model.embeddingDimension)
        }

        // Candidate windows + per-window RMS for a relative silence gate.
        var starts: [Int] = []
        var rms: [Float] = []
        var i = 0
        while i + win <= pcm.count {
            var acc: Float = 0
            for k in i..<(i + win) { acc += pcm[k] * pcm[k] }
            starts.append(i)
            rms.append((acc / Float(win)).squareRoot())
            i += hop
        }
        // Relative-silence gate decision (pure, unit-pinned — ND15):
        // keep windows ≥ 20% of the loudest, falling back to all windows if
        // the gate would empty everything (uniformly quiet audio).
        let kept = SpeakerWindowGate.keptIndices(rms: rms)

        var out: [SpeakerSegmentEmbedding] = []
        out.reserveCapacity(kept.count)
        for idx in kept {
            let s = starts[idx]
            let vec = model.embed(Array(pcm[s..<(s + win)]))
            out.append(
                SpeakerSegmentEmbedding(
                    start: Double(s) / sr,
                    end: Double(s + win) / sr,
                    embedding: vec,
                    durationSeconds: Double(win) / sr))
        }
        return SpeakerEmbeddingResult(
            segments: out, dimension: model.embeddingDimension)
    }
}
