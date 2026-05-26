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

    private let allowedIds: [String]
    private let defaultId: String
    private let estimatedBytes: Int
    private var model: SortformerModel?
    private var residentId: String?

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
        estimatedBytes: Int = 1 * 1024 * 1024 * 1024
    ) {
        precondition(
            !modelIds.isEmpty,
            "MLXDiarizationModule needs at least one model id")
        self.allowedIds = modelIds
        self.defaultId = modelIds[0]
        self.estimatedBytes = estimatedBytes
    }

    public init(
        modelId: String,
        estimatedBytes: Int = 1 * 1024 * 1024 * 1024
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
        guard allowedIds.contains(id) else {
            throw AthenaError.modelNotAvailable(
                requested: id, available: allowedIds)
        }
        do {
            model = try await SortformerModel.fromPretrained(id)
            residentId = id
        } catch {
            model = nil
            residentId = nil
            throw AthenaError.moduleLoadFailed(
                .diarization, reason: "sortformer \(id): \(error)")
        }
    }

    public func unload() async {
        model = nil
        residentId = nil
    }

    // M41 — ModelSelectable. M41.3 repeatable `--diarization-model`
    // allowlist with in-place rebind.
    public func allowedModelIds() -> [String] { allowedIds }
    public func defaultModelId() -> String { defaultId }
    public func residentModelId() -> String? { residentId }
    public func rebind(to id: String?) async throws {
        let target = id ?? defaultId
        guard allowedIds.contains(target) else {
            throw AthenaError.modelNotAvailable(
                requested: target, available: allowedIds)
        }
        if residentId == target, model != nil { return }
        model = nil
        residentId = nil
        try await loadModel(id: target)
    }

    public func diarize(
        audio: Data, filename: String?
    ) async throws -> DiarizationResult {
        guard let model else {
            throw AthenaError.moduleLoadFailed(
                .diarization, reason: "diarize called before load")
        }
        let ext = (filename as NSString?)?.pathExtension ?? ""
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent(
                "athena-diar-\(UUID().uuidString)"
                    + (ext.isEmpty ? "" : ".\(ext)"))
        try audio.write(to: tmp)
        defer { try? FileManager.default.removeItem(at: tmp) }

        let pcm = try AudioDecode.pcm16kMono(from: tmp)
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

        if durationSeconds <= offlineMaxSeconds * 0.95 {
            return try await Self.runOffline(Send((model, audio)))
        }
        return try await Self.runStreaming(Send((model, audio)))
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
