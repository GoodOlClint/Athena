import AthenaCore
import Foundation

/// Typed inference contract for the text-embedding module. The serve path
/// holds this; the MLX-backed implementation is `MLXEmbeddingModule`.
public protocol EmbeddingModule: InferenceModule {
    /// Embed each input string into its vector, order preserved.
    func embed(_ texts: [String]) async throws -> [[Float]]
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
    /// and clients work end-to-end under `--engine stub`.
    public func embed(_ texts: [String]) async throws -> [[Float]] {
        texts.map { text in
            var h: UInt64 = 1_469_598_103_934_665_603
            var v = [Float](repeating: 0, count: 8)
            for (i, b) in Array(text.utf8).enumerated() {
                h = (h ^ UInt64(b)) &* 1_099_511_628_211
                v[i & 7] += Float(h % 1000) / 1000.0
            }
            let n = max(1e-12, sqrt(v.reduce(0) { $0 + $1 * $1 }))
            return v.map { $0 / n }
        }
    }
}
