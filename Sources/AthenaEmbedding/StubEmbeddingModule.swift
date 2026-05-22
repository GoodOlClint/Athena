import AthenaCore
import Foundation

/// Result of an embedding batch (M27.1): the vectors (order preserved)
/// plus the total number of input tokens consumed, for the OpenAI
/// `usage` object. Embeddings have no completion, so `promptTokens`
/// equals `total_tokens` and `completion_tokens` is 0.
public struct EmbeddingBatch: Sendable {
    public let vectors: [[Float]]
    public let promptTokens: Int
    public init(vectors: [[Float]], promptTokens: Int) {
        self.vectors = vectors
        self.promptTokens = promptTokens
    }
}

/// Typed inference contract for the text-embedding module. The serve path
/// holds this; the MLX-backed implementation is `MLXEmbeddingModule`.
public protocol EmbeddingModule: InferenceModule {
    /// Embed each input string into its vector, order preserved, with the
    /// total input token count for usage accounting (M27.1).
    func embed(_ texts: [String]) async throws -> EmbeddingBatch
}

/// M0 placeholder, still used by `--engine stub`. Returns a deterministic
/// low-dim vector per input so `/v1/embeddings` is demoable without a
/// model and the governor wiring/budget accounting are exercised.
public actor StubEmbeddingModule: EmbeddingModule {
    public nonisolated let id: ModuleID = .textEmbedding

    private let reserveBytes: Int
    private var loaded = false

    public init(reserveBytes: Int = 1 * 1024 * 1024 * 1024) {
        self.reserveBytes = reserveBytes
    }

    public var residentBytes: Int { loaded ? reserveBytes : 0 }
    public func memoryEstimate() -> Int { reserveBytes }
    public func load(reservation: MemoryReservation) async throws { loaded = true }
    public func unload() async { loaded = false }

    /// Deterministic 8-dim pseudo-embedding (FNV-1a byte folds). Not
    /// semantically meaningful — only stable per input so the endpoint
    /// and clients work end-to-end under `--engine stub`. Token count is
    /// a whitespace-delimited estimate (≥1 per non-empty input) so
    /// `usage` is non-zero without a real tokenizer.
    public func embed(_ texts: [String]) async throws -> EmbeddingBatch {
        let vectors = texts.map { text in
            var h: UInt64 = 1_469_598_103_934_665_603
            var v = [Float](repeating: 0, count: 8)
            for (i, b) in Array(text.utf8).enumerated() {
                h = (h ^ UInt64(b)) &* 1_099_511_628_211
                v[i & 7] += Float(h % 1000) / 1000.0
            }
            let n = max(1e-12, sqrt(v.reduce(0) { $0 + $1 * $1 }))
            return v.map { $0 / n }
        }
        let tokens = texts.reduce(0) { acc, t in
            let n = t.split(whereSeparator: { $0.isWhitespace }).count
            return acc + (t.isEmpty ? 0 : max(1, n))
        }
        return EmbeddingBatch(vectors: vectors, promptTokens: tokens)
    }
}
