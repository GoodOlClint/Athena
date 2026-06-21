import AthenaCore
import Foundation
import MLX

/// Carries the non-Sendable model + audio across the actor→generate
/// await as one Sendable unit (the actor already serialises access;
/// the vendored Sortformer boxes its own internals likewise).
private struct Send<T>: @unchecked Sendable {
    let v: T
    init(_ v: T) { self.v = v }
}

/// The real MLX-backed speaker-diarization module (M4.3b). Wraps the
/// vendored Sortformer model. Memory accounting is a fixed pre-load
/// estimate (weights aren't on disk until first download); live
/// reconciliation is handled by the M5 governor probe. HF cache root
/// follows `HF_HOME` (SSD/local fallback), so this stays
/// location-agnostic.
public actor MLXDiarizationModule: DiarizationModule, ModelSelectable {
    public nonisolated let id: ModuleID = .diarization
    public nonisolated var moduleID: ModuleID { .diarization }

    private var allowedIds: [String]
    private var defaultId: String
    private let estimatedBytes: Int
    private var model: SortformerModel?
    /// The pyannote PyanNet segmentation engine, resident iff
    /// `backend == .pyannoteSegmentation` (ADR 018 / S2).
    private var segmentationModel: PyanNetSegmentationModel?
    private var residentId: String?
    /// Backend of the resident model (ADR 018). Set on load from the
    /// checkpoint's `model_type`; gates which method (`diarize` vs `segment`)
    /// the resident model can serve. Defaults to `.sortformer` so the dense
    /// Sortformer path is byte-unchanged.
    private var backend: DiarizationBackend = .sortformer
    /// D1/D2 (M68.3) — the in-flight generate, if any. `diarize` chains new
    /// generates after it (FIFO) so two never run concurrent forwards on the
    /// shared `SortformerModel` — `generate`/`generateStream` each run on a
    /// `Task.detached` that races the global `MLX.Memory.clearCache`, so two
    /// concurrent diarize calls would corrupt each other's forward. And
    /// `rebind`/`unload` await it before dropping the model, so a model swap
    /// never lands mid-generate (no drain pre-fix). Token-keyed cleanup so a
    /// newer generate isn't clobbered.
    private var generateInFlight: Task<DiarizationResult, Error>?
    private var generateSeq: UInt64 = 0
    private var generateGateSeq: UInt64 = 0
    /// Model-store root (M54) — local-store-dir load when materialized,
    /// so bare name or full HF id both work. nil ⇒ Hub-by-id fallback.
    private let modelStoreRoot: URL?

    /// - Parameters:
    ///   - modelIds: HF id allowlist (first = default). Default
    ///     `[mlx-community/diar_streaming_sortformer_4spk-v2.1-fp16]`.
    ///   - estimatedBytes: governor admission estimate. 1 GiB:
    ///     conservative headroom over the small fp16 weights + the
    ///     FastConformer/transformer activation working set on long
    ///     audio (M5 reconciles to the real footprint post-load).
    public init(
        modelIds: [String] = [
            "mlx-community/diar_streaming_sortformer_4spk-v2.1-fp16"
        ],
        modelStoreRoot: URL? = nil,
        estimatedBytes: Int = 1 * 1024 * 1024 * 1024
    ) {
        precondition(
            !modelIds.isEmpty,
            "MLXDiarizationModule needs at least one model id")
        self.allowedIds = modelIds
        self.defaultId = modelIds[0]
        self.modelStoreRoot = modelStoreRoot
        self.estimatedBytes = estimatedBytes
    }

    public init(
        modelId: String,
        estimatedBytes: Int = 1 * 1024 * 1024 * 1024
    ) {
        self.init(modelIds: [modelId], estimatedBytes: estimatedBytes)
    }

    public var residentBytes: Int {
        model == nil && segmentationModel == nil ? 0 : estimatedBytes
    }

    public func memoryEstimate() -> Int { estimatedBytes }

    public func load(reservation: MemoryReservation) async throws {
        if model != nil || segmentationModel != nil { return }
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
                .diarization,
                reason: "model '\(canonical)' is not in the model store "
                    + "— pull it first (operator action); inference does "
                    + "not auto-download")
        }
        let resolved = dir.resolvingSymlinksInPath()
        // ADR 018 — classify the resident model's backend from its config.json
        // so the route can match the request `method` against it.
        let detected = DiarizationBackend.detect(in: resolved)
        switch detected {
        case .sortformer, .unknown:
            // `.unknown` falls back to Sortformer for back-compat (older
            // Sortformer snapshots that predate an explicit `model_type`).
            do {
                model = try SortformerModel.fromModelDirectory(resolved)
                segmentationModel = nil
                residentId = canonical
                backend = .sortformer
            } catch {
                model = nil
                residentId = nil
                throw AthenaError.moduleLoadFailed(
                    .diarization, reason: "sortformer \(canonical): \(error)")
            }
        case .pyannoteSegmentation:
            do {
                segmentationModel =
                    try PyanNetSegmentationModel.fromModelDirectory(resolved)
                model = nil
                residentId = canonical
                backend = .pyannoteSegmentation
            } catch {
                segmentationModel = nil
                residentId = nil
                throw AthenaError.moduleLoadFailed(
                    .diarization,
                    reason: "pyannote-segmentation \(canonical): \(error)")
            }
        }
    }

    public func unload() async {
        // D2 — drain an in-flight generate before dropping the model so the
        // detached forward never runs against a torn-down model / freed slot.
        if let t = generateInFlight { _ = try? await t.value }
        model = nil
        segmentationModel = nil
        residentId = nil
        backend = .sortformer
    }

    // M41 — ModelSelectable. M41.3 repeatable `--diarization-model`
    // allowlist with in-place rebind.
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
        if residentId == target, model != nil || segmentationModel != nil {
            return
        }
        // D2 — drain an in-flight generate before swapping the model.
        if let t = generateInFlight { _ = try? await t.value }
        model = nil
        segmentationModel = nil
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
            segmentationModel = nil
            residentId = nil
            backend = .sortformer
        }
    }

    /// Backend of the resident model (ADR 018) — the route uses this to match
    /// the request `method` against the loaded model.
    public func residentBackend() -> DiarizationBackend { backend }

    public func diarize(
        audio: Data, filename: String?
    ) async throws -> DiarizationResult {
        // ADR 018 — end-to-end `diarize` is Sortformer-only. A segmentation
        // model answers `segment`, not `diarize`.
        guard backend == .sortformer else {
            throw AthenaError.diarizationMethodInvalid(
                method: "sortformer",
                reason: "the loaded model is a \(backend.rawValue) "
                    + "segmentation model; use method=pyannote")
        }
        guard let model else {
            throw AthenaError.moduleLoadFailed(
                .diarization, reason: "diarize called before load")
        }
        // Option D (ADR 025 S5): decode from the in-memory upload bytes.
        var pcm = try await AudioDecode.pcm16kMono(
            from: audio, filename: filename, module: .diarization)
        defer { ProcessHardening.secureZero(&pcm) }  // ADR 024 T2
        let audio = MLXArray(pcm).asType(.float32)

        // M24.4b: the offline single-pass path is capped by the
        // transformer encoder's positional table (~120s); longer audio is
        // routed to the model's native streaming path (chunked, with a
        // persistent speaker cache so ids stay stable across chunks). A
        // small margin keeps clips near the boundary on streaming so they
        // never overrun the offline table.
        let cfg = model.config
        let frameDuration =
            Double(
                cfg.processorConfig.hopLength
                    * cfg.fcEncoderConfig.subsamplingFactor)
            / Double(cfg.processorConfig.samplingRate)
        let offlineMaxSeconds =
            Double(cfg.tfEncoderConfig.maxSourcePositions) * frameDuration
        let durationSeconds =
            Double(pcm.count) / Double(AudioDecode.sampleRate)

        let streaming = durationSeconds > offlineMaxSeconds * 0.95
        // Box the non-Sendable model+audio once (the actor already serialises
        // access) so the serialization Task below can capture it.
        let send = Send((model, audio))

        // D1 — serialize: chain this generate after any in-flight one so two
        // never run concurrent forwards on the shared model (racing the global
        // clearCache). Record it so rebind/unload can drain it (D2).
        let prior = generateInFlight
        generateSeq &+= 1
        let mySeq = generateSeq
        let task = Task { () async throws -> DiarizationResult in
            _ = try? await prior?.value
            return streaming
                ? try await Self.runStreaming(send)
                : try await Self.runOffline(send)
        }
        generateInFlight = task
        generateGateSeq = mySeq
        defer { if generateGateSeq == mySeq { generateInFlight = nil } }
        return try await task.value
    }

    public func segment(
        audio: Data, filename: String?
    ) async throws -> [SpeakerActivityRegion] {
        // ADR 018 — segmentation is the pyannote backend only.
        guard backend == .pyannoteSegmentation, let segmentationModel else {
            throw AthenaError.diarizationMethodInvalid(
                method: "pyannote",
                reason: "the loaded model is a \(backend.rawValue) model, "
                    + "which has no segmentation backend; use method=sortformer")
        }
        // Option D (ADR 025 S5): decode from the in-memory upload bytes.
        var pcm = try await AudioDecode.pcm16kMono(
            from: audio, filename: filename, module: .diarization)
        defer { ProcessHardening.secureZero(&pcm) }  // ADR 024 T2
        // PyanNet is length-agnostic and stream-windowed (no positional cap),
        // so the whole file is segmented in one pass; box the non-Sendable
        // model+pcm so the detached run can capture it (the actor serialises).
        let send = Send((segmentationModel, pcm))
        let task = Task { () -> [SpeakerActivityRegion] in
            let (m, samples) = send.v
            return m.segment(samples)
        }
        return try await task.value
    }

    /// nonisolated: the model/audio arrive boxed-Sendable and the
    /// non-Sendable `DiarizationOutput` is reduced to the Sendable
    /// `DiarizationResult` here, so nothing non-Sendable crosses back
    /// to the actor.
    private static func runOffline(
        _ b: Send<(SortformerModel, MLXArray)>
    ) async throws -> DiarizationResult {
        let (model, audio) = b.v
        let out = try await model.generate(
            audio: audio, sampleRate: 16_000)
        return DiarizationResult(
            turns: out.segments.map {
                DiarizationTurn(
                    start: Double($0.start), end: Double($0.end),
                    speaker: $0.speaker)
            },
            numSpeakers: out.numSpeakers)
    }

    /// Long-audio path (M24.4b): drain the model's streaming generator
    /// and aggregate every chunk's turns. The streaming state carries a
    /// persistent speaker cache, so speaker ids are consistent across
    /// chunks; the union is the speaker count. Same Sendable-reduction
    /// contract as `runOffline`.
    private static func runStreaming(
        _ b: Send<(SortformerModel, MLXArray)>
    ) async throws -> DiarizationResult {
        let (model, audio) = b.v
        var turns: [DiarizationTurn] = []
        for try await out in model.generateStream(
            audio: audio, sampleRate: 16_000, chunkDuration: 10.0)
        {
            for s in out.segments {
                turns.append(
                    DiarizationTurn(
                        start: Double(s.start), end: Double(s.end),
                        speaker: s.speaker))
            }
        }
        turns.sort { $0.start < $1.start }
        let speakers = Set(turns.map { $0.speaker }).count
        return DiarizationResult(turns: turns, numSpeakers: speakers)
    }
}
