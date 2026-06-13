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

    private var allowedIds: [String]
    private var defaultId: String
    private let estimatedBytes: Int
    private var model: WeSpeakerModel?
    private var residentId: String?
    /// Model-store root (M54) — local-store-dir load when materialized,
    /// so bare name or full HF id both work. nil ⇒ Hub-by-id fallback.
    private let modelStoreRoot: URL?

    /// - Parameters:
    ///   - modelIds: HF id allowlist (first = default). Default
    ///     `[aufklarer/WeSpeaker-ResNet34-LM-MLX]` — the ungated
    ///     safetensors mirror (substrate cannot read the mlx-community
    ///     `.npz`).
    ///   - estimatedBytes: governor admission estimate. 512 MiB: the
    ///     weights are tiny (~25 MB) but ResNet activations over a
    ///     long segment's mel grid dominate; conservative headroom that
    ///     M5 reconciles to the real footprint post-load.
    public init(
        modelIds: [String] = ["aufklarer/WeSpeaker-ResNet34-LM-MLX"],
        modelStoreRoot: URL? = nil,
        estimatedBytes: Int = 512 * 1024 * 1024
    ) {
        precondition(
            !modelIds.isEmpty,
            "MLXSpeakerEmbeddingModule needs at least one model id")
        self.allowedIds = modelIds
        self.defaultId = modelIds[0]
        self.modelStoreRoot = modelStoreRoot
        self.estimatedBytes = estimatedBytes
    }

    public init(
        modelId: String,
        estimatedBytes: Int = 512 * 1024 * 1024
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
        // M54.3 — local-store-dir load only; inference never auto-downloads
        // (operator pulls a missing model at startup / allowlist-add).
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

    // M41 — ModelSelectable. M41.3 repeatable
    // `--speaker-embedding-model` allowlist with in-place rebind.
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
        residentId = nil
        try await loadModel(id: target)
    }

    public func setAllowedModelIds(_ ids: [String]) {
        allowedIds = ids
        defaultId = ids.first ?? defaultId
        // ND5: evict only when the resident id is no longer allowed under the
        // SAME store-identity resolver as load/rebind (an equivalent spelling
        // must NOT force a needless reload).
        if let r = residentId, ids.canonicalByStoreIdentity(r) == nil {
            model = nil
            residentId = nil
        }
    }

    public func embed(
        audio: Data, filename: String?, segments: [SpeakerSegmentRequest]
    ) async throws -> SpeakerEmbeddingResult {
        guard let model else {
            throw AthenaError.moduleLoadFailed(
                .speakerEmbedding, reason: "embed called before load")
        }

        let ext = (filename as NSString?)?.pathExtension ?? ""
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent(
                "athena-spkemb-\(UUID().uuidString)"
                    + (ext.isEmpty ? "" : ".\(ext)"))
        try audio.write(to: tmp)
        defer { try? FileManager.default.removeItem(at: tmp) }

        let pcm = try AudioDecode.pcm16kMono(from: tmp)
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
        guard let model else {
            throw AthenaError.moduleLoadFailed(
                .speakerEmbedding, reason: "embed called before load")
        }
        let ext = (filename as NSString?)?.pathExtension ?? ""
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent(
                "athena-spkwin-\(UUID().uuidString)"
                    + (ext.isEmpty ? "" : ".\(ext)"))
        try audio.write(to: tmp)
        defer { try? FileManager.default.removeItem(at: tmp) }

        let pcm = try AudioDecode.pcm16kMono(from: tmp)
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
        let maxRMS = rms.max() ?? 0
        let gate = maxRMS * 0.20  // keep windows ≥ 20% of the loudest

        var out: [SpeakerSegmentEmbedding] = []
        out.reserveCapacity(starts.count)
        for (idx, s) in starts.enumerated() {
            // If the gate would drop everything (uniformly quiet), keep all.
            if maxRMS > 0, rms[idx] < gate { continue }
            let vec = model.embed(Array(pcm[s..<(s + win)]))
            out.append(
                SpeakerSegmentEmbedding(
                    start: Double(s) / sr,
                    end: Double(s + win) / sr,
                    embedding: vec,
                    durationSeconds: Double(win) / sr))
        }
        // Fallback: gate removed everything → embed every window ungated.
        if out.isEmpty {
            for s in starts {
                let vec = model.embed(Array(pcm[s..<(s + win)]))
                out.append(
                    SpeakerSegmentEmbedding(
                        start: Double(s) / sr,
                        end: Double(s + win) / sr,
                        embedding: vec,
                        durationSeconds: Double(win) / sr))
            }
        }
        return SpeakerEmbeddingResult(
            segments: out, dimension: model.embeddingDimension)
    }
}
