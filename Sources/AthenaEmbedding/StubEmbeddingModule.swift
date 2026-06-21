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

    private let modelIds: [String]
    private let configuredDefault: String?
    private let reserveBytes: Int
    /// nil ⇒ unloaded. The stub has no real container; the value tracks
    /// which id is "resident" for M39/M41 selection + rebind semantics.
    private var residentId: String?
    /// NI3 parity — staged selection honored on (re)load, mirroring the MLX
    /// module so the stub reflects production evict→reload behavior.
    private var desiredName: String?

    /// - Parameters:
    ///   - modelIds: the stand-in store set (the stub has no disk). The stub
    ///     "serves" any of these truthfully (M39 parity); an id outside the set
    ///     is a `modelNotAvailable` 400.
    ///   - configuredDefault: per-module TOML default (ADR 026; nil ⇒ ambiguity).
    public init(
        modelIds: [String] = ["athena-embedding"],
        configuredDefault: String? = nil,
        reserveBytes: Int = 1 * 1024 * 1024 * 1024
    ) {
        precondition(
            !modelIds.isEmpty,
            "StubEmbeddingModule needs at least one model id")
        self.modelIds = modelIds
        self.configuredDefault =
            (configuredDefault?.isEmpty == true) ? nil : configuredDefault
        self.reserveBytes = reserveBytes
    }

    public var residentBytes: Int { residentId == nil ? 0 : reserveBytes }
    public func memoryEstimate() -> Int { reserveBytes }
    public func load(reservation: MemoryReservation) async throws {
        if residentId == nil { residentId = try desiredName ?? resolve(nil) }
    }
    public func unload() async { residentId = nil }

    public func allowedModelIds() -> [String] { modelIds }
    public func defaultModelId() -> String {
        ModelSelection.displayDefault(
            available: modelIds, configuredDefault: configuredDefault)
    }
    public func residentModelId() -> String? { residentId }
    public func rebind(to id: String?) async throws {
        let target = try resolve(id)
        desiredName = target
        residentId = target
    }

    /// ADR 026 resolution against the injected stub set.
    private func resolve(_ id: String?) throws -> String {
        switch ModelSelection.resolve(
            available: modelIds, configuredDefault: configuredDefault,
            requested: id)
        {
        case .resolved(let t): return t
        case .notAvailable:
            throw AthenaError.modelNotAvailable(
                requested: id ?? (configuredDefault ?? ""),
                available: modelIds)
        case .ambiguous:
            throw AthenaError.ambiguousModel(
                module: .textEmbedding, available: modelIds)
        }
    }

    /// Deterministic 8-dim pseudo-embedding (FNV-1a byte folds), MLX-free.
    /// Not semantically meaningful — only stable per (model, text) so the
    /// endpoint and clients work end-to-end under `--engine stub`.
    ///
    /// L8 (M70.3): the served model id is folded into the hash BEFORE the
    /// text, so distinct models yield distinct vectors — mirroring the real
    /// module (where different models genuinely produce different vectors) and
    /// making "the vector actually depends on the model, not just the echoed
    /// label" CI-testable. A separator byte after the id keeps the
    /// model/text boundary unambiguous. (A `nil`/empty text still folds to a
    /// zero vector regardless of model — the I8 stub-empty-string item,
    /// deferred; non-empty inputs are model-distinguishable.)
    static func stubVector(text: String, model: String) -> [Float] {
        var h: UInt64 = 1_469_598_103_934_665_603
        for b in model.utf8 { h = (h ^ UInt64(b)) &* 1_099_511_628_211 }
        h = (h ^ 0x1) &* 1_099_511_628_211  // model/text separator
        var v = [Float](repeating: 0, count: 8)
        for (i, b) in Array(text.utf8).enumerated() {
            h = (h ^ UInt64(b)) &* 1_099_511_628_211
            v[i & 7] += Float(h % 1000) / 1000.0
        }
        let n = max(1e-12, sqrt(v.reduce(0) { $0 + $1 * $1 }))
        return v.map { $0 / n }
    }

    /// Token count is a whitespace-delimited estimate (≥1 per non-empty
    /// input) so `usage` is non-zero without a real tokenizer.
    ///
    /// M39: `model` selects from the configured set; it validates membership
    /// and echoes the served id truthfully (so selection/400 behavior matches
    /// the real module) AND, per L8, drives the vector math.
    public func embed(_ texts: [String], model: String? = nil) async throws
        -> EmbeddingBatch
    {
        // ADR 026 — resolve `model` against the injected stub set (bare name
        // OR full HF id; omit ⇒ configured default / sole model / ambiguity).
        let served = try resolve(model)
        // M41: per-request selection rebinds the slot's "resident" id in the
        // stub too, so /api/models/resident reflects what an embed call
        // would actually serve. NI3 parity: also stage it for reload.
        desiredName = served
        residentId = served
        let vectors = texts.map { Self.stubVector(text: $0, model: served) }
        let tokens = texts.reduce(0) { acc, t in
            let n = t.split(whereSeparator: { $0.isWhitespace }).count
            return acc + (t.isEmpty ? 0 : max(1, n))
        }
        return EmbeddingBatch(
            vectors: vectors, promptTokens: tokens, model: served)
    }
}
