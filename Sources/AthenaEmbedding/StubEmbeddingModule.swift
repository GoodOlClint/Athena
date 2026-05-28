import AthenaCore
import Foundation

/// Result of an embedding batch (M27.1): the vectors (order preserved)
/// plus the total number of input tokens consumed, for the OpenAI
/// `usage` object. Embeddings have no completion, so `promptTokens`
/// equals `total_tokens` and `completion_tokens` is 0.
///
/// `model` is the id the module ACTUALLY served (M39): with per-request
/// model selection the served model may differ from the default, so the
/// caller echoes this back rather than the requested string — no more
/// "false friend" where the response model was the unused request echo.
public struct EmbeddingBatch: Sendable {
    public let vectors: [[Float]]
    public let promptTokens: Int
    public let model: String
    public init(vectors: [[Float]], promptTokens: Int, model: String) {
        self.vectors = vectors
        self.promptTokens = promptTokens
        self.model = model
    }
}

/// Typed inference contract for the text-embedding module. The serve path
/// holds this; the MLX-backed implementation is `MLXEmbeddingModule`.
public protocol EmbeddingModule: InferenceModule {
    /// Embed each input string into its vector, order preserved, with the
    /// total input token count for usage accounting (M27.1).
    ///
    /// `model` (M39) selects among the module's configured set: `nil` ⇒
    /// the default (first declared) model; a non-`nil` id that is NOT in
    /// the set ⇒ `AthenaError.modelNotAvailable` (400), never a silent
    /// fallback. The returned batch reports the id actually served.
    func embed(_ texts: [String], model: String?) async throws
        -> EmbeddingBatch
}

/// M0 placeholder, still used by `--engine stub`. Returns a deterministic
/// low-dim vector per input so `/v1/embeddings` is demoable without a
/// model and the governor wiring/budget accounting are exercised.
public actor StubEmbeddingModule: EmbeddingModule, ModelSelectable {
    public nonisolated let id: ModuleID = .textEmbedding
    public nonisolated var moduleID: ModuleID { .textEmbedding }

    private var allowedIds: [String]
    private var defaultId: String
    private let reserveBytes: Int
    /// nil ⇒ unloaded. The stub has no real container; the value tracks
    /// which id is "resident" for M39/M41 selection + rebind semantics.
    private var residentId: String?

    /// - Parameters:
    ///   - modelIds: the selectable set (first = default). The stub
    ///     "serves" any of these truthfully (M39 parity with the real
    ///     module); an id outside the set is a `modelNotAvailable` 400.
    public init(
        modelIds: [String] = ["athena-embedding"],
        reserveBytes: Int = 1 * 1024 * 1024 * 1024
    ) {
        precondition(
            !modelIds.isEmpty,
            "StubEmbeddingModule needs at least one model id")
        self.allowedIds = modelIds
        self.defaultId = modelIds[0]
        self.reserveBytes = reserveBytes
    }

    public var residentBytes: Int { residentId == nil ? 0 : reserveBytes }
    public func memoryEstimate() -> Int { reserveBytes }
    public func load(reservation: MemoryReservation) async throws {
        if residentId == nil { residentId = defaultId }
    }
    public func unload() async { residentId = nil }

    public func allowedModelIds() -> [String] { allowedIds }
    public func defaultModelId() -> String { defaultId }
    public func residentModelId() -> String? { residentId }
    public func rebind(to id: String?) async throws {
        let requested = id ?? defaultId
        // M46.4 — case-insensitive lookup; canonical id from storage.
        guard let target =
            allowedIds.canonicalCaseInsensitive(requested)
        else {
            throw AthenaError.modelNotAvailable(
                requested: requested, available: allowedIds)
        }
        residentId = target
    }

    public func setAllowedModelIds(_ ids: [String]) {
        allowedIds = ids
        defaultId = ids.first ?? defaultId
        if let r = residentId, !ids.contains(r) { residentId = nil }
    }

    /// Deterministic 8-dim pseudo-embedding (FNV-1a byte folds). Not
    /// semantically meaningful — only stable per input so the endpoint
    /// and clients work end-to-end under `--engine stub`. Token count is
    /// a whitespace-delimited estimate (≥1 per non-empty input) so
    /// `usage` is non-zero without a real tokenizer.
    ///
    /// M39: `model` selects from the configured set (the stub ignores the
    /// id for the actual math — every model yields the same fixed 8-dim
    /// fold — but it validates membership and echoes the served id
    /// truthfully, so selection/400 behavior matches the real module).
    public func embed(_ texts: [String], model: String? = nil) async throws
        -> EmbeddingBatch
    {
        let requested = model ?? defaultId
        // M46.4 — case-insensitive lookup; canonical id from storage.
        guard let served =
            allowedIds.canonicalCaseInsensitive(requested)
        else {
            throw AthenaError.modelNotAvailable(
                requested: requested, available: allowedIds)
        }
        // M41: per-request selection rebinds the slot's "resident" id in the
        // stub too, so /api/models/resident reflects what an embed call
        // would actually serve.
        residentId = served
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
        return EmbeddingBatch(
            vectors: vectors, promptTokens: tokens, model: served)
    }
}
