import AthenaCore
import Foundation

/// A segment of the uploaded audio to embed (seconds from clip start).
public struct SpeakerSegmentRequest: Sendable, Equatable {
    public let start: Double
    public let end: Double
    public init(start: Double, end: Double) {
        self.start = start
        self.end = end
    }
}

/// One speaker-embedding vector for a segment.
public struct SpeakerSegmentEmbedding: Sendable {
    public let start: Double
    public let end: Double
    public let embedding: [Float]
    public let durationSeconds: Double
    public init(
        start: Double, end: Double, embedding: [Float],
        durationSeconds: Double
    ) {
        self.start = start
        self.end = end
        self.embedding = embedding
        self.durationSeconds = durationSeconds
    }
}

/// `Sendable` result for the module boundary.
public struct SpeakerEmbeddingResult: Sendable {
    public let segments: [SpeakerSegmentEmbedding]
    public let dimension: Int
    public init(segments: [SpeakerSegmentEmbedding], dimension: Int) {
        self.segments = segments
        self.dimension = dimension
    }
}

/// Typed inference contract for the speaker-embedding (speaker-
/// verification) module (vendored WeSpeaker ResNet34-LM). Produces a
/// per-segment voice vector for cross-recording speaker identification —
/// distinct from Sortformer's internal per-frame features (which are not
/// verification embeddings).
public protocol SpeakerEmbeddingModule: InferenceModule {
    /// Embed each `segment` of `audio` (raw uploaded file bytes;
    /// `filename` hints the container). An empty `segments` list embeds
    /// the whole clip as one segment.
    func embed(
        audio: Data, filename: String?, segments: [SpeakerSegmentRequest]
    ) async throws -> SpeakerEmbeddingResult

    /// Slide a `windowSeconds` window with `hopSeconds` hop across the
    /// clip, gate out near-silent windows, and embed each retained one.
    /// Feeds the embedding+clustering diarizer (M25.3).
    func windowEmbeddings(
        audio: Data, filename: String?,
        windowSeconds: Double, hopSeconds: Double
    ) async throws -> SpeakerEmbeddingResult
}

/// `--engine stub` placeholder: a deterministic pseudo-vector per
/// segment so the endpoint + governor wiring/accounting work without a
/// model. Matches the real 256-d shape and is L2-normalized.
public actor StubSpeakerEmbeddingModule: SpeakerEmbeddingModule,
    ModelSelectable
{
    public nonisolated let id: ModuleID = .speakerEmbedding
    public nonisolated var moduleID: ModuleID { .speakerEmbedding }

    private let reserveBytes: Int
    private let dimension = 256
    private let modelIds: [String]
    private let defaultId: String
    private var residentId: String?

    public init(
        reserveBytes: Int = 256 * 1024 * 1024,
        modelIds: [String] = ["athena-stub-wespeaker"]
    ) {
        precondition(
            !modelIds.isEmpty,
            "StubSpeakerEmbeddingModule needs at least one model id")
        self.reserveBytes = reserveBytes
        self.modelIds = modelIds
        self.defaultId = modelIds[0]
    }

    public var residentBytes: Int { residentId == nil ? 0 : reserveBytes }
    public func memoryEstimate() -> Int { reserveBytes }
    public func load(reservation: MemoryReservation) async throws {
        if residentId == nil { residentId = defaultId }
    }
    public func unload() async { residentId = nil }

    public func allowedModelIds() -> [String] { modelIds }
    public func defaultModelId() -> String { defaultId }
    public func residentModelId() -> String? { residentId }
    public func rebind(to id: String?) async throws {
        let target = id ?? defaultId
        guard modelIds.contains(target) else {
            throw AthenaError.modelNotAvailable(
                requested: target, available: modelIds)
        }
        residentId = target
    }

    public func embed(
        audio: Data, filename: String?, segments: [SpeakerSegmentRequest]
    ) async throws -> SpeakerEmbeddingResult {
        let reqs =
            segments.isEmpty
            ? [SpeakerSegmentRequest(start: 0, end: 0)] : segments
        let out = reqs.map { seg -> SpeakerSegmentEmbedding in
            var v = [Float](repeating: 0, count: dimension)
            var h: UInt64 = 1_469_598_103_934_665_603
            for b in "\(seg.start):\(seg.end)".utf8 {
                h = (h ^ UInt64(b)) &* 1_099_511_628_211
            }
            for i in 0..<dimension {
                h = (h ^ UInt64(i)) &* 1_099_511_628_211
                v[i] = Float(Int64(bitPattern: h) % 1000) / 1000.0
            }
            let n = max(sqrt(v.reduce(0) { $0 + $1 * $1 }), 1e-9)
            return SpeakerSegmentEmbedding(
                start: seg.start, end: seg.end,
                embedding: v.map { $0 / n },
                durationSeconds: max(0, seg.end - seg.start))
        }
        return SpeakerEmbeddingResult(segments: out, dimension: dimension)
    }

    public func windowEmbeddings(
        audio: Data, filename: String?,
        windowSeconds: Double, hopSeconds: Double
    ) async throws -> SpeakerEmbeddingResult {
        // Stub: three canned windows so the clustering path is wired.
        try await embed(
            audio: audio, filename: filename,
            segments: [
                SpeakerSegmentRequest(start: 0, end: windowSeconds),
                SpeakerSegmentRequest(
                    start: hopSeconds, end: hopSeconds + windowSeconds),
                SpeakerSegmentRequest(
                    start: 2 * hopSeconds, end: 2 * hopSeconds + windowSeconds),
            ])
    }
}
