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
public actor MLXDiarizationModule: DiarizationModule {
    public nonisolated let id: ModuleID = .diarization

    private let modelId: String
    private let estimatedBytes: Int
    private var model: SortformerModel?

    /// - Parameters:
    ///   - modelId: HF id (default the ungated mlx-community 4-speaker
    ///     streaming Sortformer).
    ///   - estimatedBytes: governor admission estimate. 1 GiB:
    ///     conservative headroom over the small fp16 weights + the
    ///     FastConformer/transformer activation working set on long
    ///     audio (M5 reconciles to the real footprint post-load).
    public init(
        modelId: String =
            "mlx-community/diar_streaming_sortformer_4spk-v2.1-fp16",
        estimatedBytes: Int = 1 * 1024 * 1024 * 1024
    ) {
        self.modelId = modelId
        self.estimatedBytes = estimatedBytes
    }

    public var residentBytes: Int { model == nil ? 0 : estimatedBytes }

    public func memoryEstimate() -> Int { estimatedBytes }

    public func load(reservation: MemoryReservation) async throws {
        if model != nil { return }
        do {
            model = try await SortformerModel.fromPretrained(modelId)
        } catch {
            throw AthenaError.moduleLoadFailed(
                .diarization, reason: "sortformer \(modelId): \(error)")
        }
    }

    public func unload() async { model = nil }

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
        return try await Self.run(Send((model, audio)))
    }

    /// nonisolated: the model/audio arrive boxed-Sendable and the
    /// non-Sendable `DiarizationOutput` is reduced to the Sendable
    /// `DiarizationResult` here, so nothing non-Sendable crosses back
    /// to the actor.
    private static func run(
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
}
