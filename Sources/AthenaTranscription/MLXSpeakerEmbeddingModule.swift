import AthenaCore
import Foundation

/// The real MLX-backed speaker-embedding module (M25.1). Wraps the
/// vendored WeSpeaker ResNet34-LM. Decodes the upload once, slices each
/// requested segment, and embeds it to a 256-d L2-normalized voice
/// vector. Memory accounting is a fixed pre-load estimate (weights are
/// not on disk until first download); the M5 governor probe reconciles
/// to the real footprint. HF cache root follows `HF_HOME`
/// (SSD/local fallback), so this stays location-agnostic.
public actor MLXSpeakerEmbeddingModule: SpeakerEmbeddingModule {
    public nonisolated let id: ModuleID = .speakerEmbedding

    private let modelId: String
    private let estimatedBytes: Int
    private var model: WeSpeakerModel?

    /// - Parameters:
    ///   - modelId: HF id (default the ungated WeSpeaker ResNet34-LM
    ///     safetensors mirror — the substrate can't read the
    ///     mlx-community `.npz`).
    ///   - estimatedBytes: governor admission estimate. 512 MiB: the
    ///     weights are tiny (~25 MB) but ResNet activations over a
    ///     long segment's mel grid dominate; conservative headroom that
    ///     M5 reconciles to the real footprint post-load.
    public init(
        modelId: String =
            "aufklarer/WeSpeaker-ResNet34-LM-MLX",
        estimatedBytes: Int = 512 * 1024 * 1024
    ) {
        self.modelId = modelId
        self.estimatedBytes = estimatedBytes
    }

    public var residentBytes: Int { model == nil ? 0 : estimatedBytes }

    public func memoryEstimate() -> Int { estimatedBytes }

    public func load(reservation: MemoryReservation) async throws {
        if model != nil { return }
        do {
            model = try await WeSpeakerModel.fromPretrained(modelId)
        } catch {
            throw AthenaError.moduleLoadFailed(
                .speakerEmbedding, reason: "wespeaker \(modelId): \(error)")
        }
    }

    public func unload() async { model = nil }

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
        for r in reqs {
            let s = max(0, Int((r.start * sr).rounded()))
            let e = min(pcm.count, Int((r.end * sr).rounded()))
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
}
