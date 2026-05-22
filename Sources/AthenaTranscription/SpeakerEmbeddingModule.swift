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
}

/// `--engine stub` placeholder: a deterministic pseudo-vector per
/// segment so the endpoint + governor wiring/accounting work without a
/// model. Matches the real 256-d shape and is L2-normalized.
public actor StubSpeakerEmbeddingModule: SpeakerEmbeddingModule {
    public nonisolated let id: ModuleID = .speakerEmbedding

    private let reserveBytes: Int
    private let dimension = 256
    private var loaded = false

    public init(reserveBytes: Int = 256 * 1024 * 1024) {
        self.reserveBytes = reserveBytes
    }

    public var residentBytes: Int { loaded ? reserveBytes : 0 }
    public func memoryEstimate() -> Int { reserveBytes }
    public func load(reservation: MemoryReservation) async throws {
        loaded = true
    }
    public func unload() async { loaded = false }

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
}
