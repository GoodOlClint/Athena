import AthenaCore
import Foundation
import HuggingFace
import MLX
import MLXEmbedders
import MLXHuggingFace
import MLXLMCommon
import Tokenizers

/// The real MLX-backed text-embedding module (M4.1). Loads a
/// sentence-embedding model through the substrate's `MLXEmbedders`
/// (`EmbedderModelContainer`) and produces L2-normalized vectors.
///
/// Sourcing mirrors the LLM module: a Hugging Face id resolved through
/// the Hub downloader into the HF cache (the cache root — SSD when
/// mounted, else the local drive — is selected by the serve entrypoint
/// via `HF_HOME`, so this module stays location-agnostic).
///
/// Memory accounting is a fixed pre-load estimate (the weights aren't on
/// disk until the first download, so there is nothing to size); live
/// Metal-footprint reconciliation is the shared M5 follow-up.
public actor MLXEmbeddingModule: EmbeddingModule, ModelSelectable {
    public nonisolated let id: ModuleID = .textEmbedding
    public nonisolated var moduleID: ModuleID { .textEmbedding }

    /// The selectable set (M39). The single resident slot is rebound to
    /// whichever member a request asks for; an id outside this set is a
    /// `modelNotAvailable` 400 — a request can never trigger an arbitrary
    /// HF download. (Future governor evolution: a multi-resident pool so
    /// hot models stay loaded; today it is one-at-a-time.)
    private let allowedIds: [String]
    private let defaultId: String
    private let estimatedBytes: Int
    private var container: EmbedderModelContainer?
    /// The id currently resident in `container` (nil ⇒ nothing loaded).
    private var residentId: String?

    /// - Parameters:
    ///   - modelIds: the selectable set of embedding model HF ids; the
    ///     first is the default (used when a request omits `model`).
    ///     Default `[BAAI/bge-small-en-v1.5]` — 384-dim, ~130 MB.
    ///   - estimatedBytes: governor admission estimate. Default 512 MiB:
    ///     safe headroom over bge-small's weights + tokenizer + the
    ///     transient activation working set. One model is resident at a
    ///     time, so the fixed estimate still bounds the slot even though
    ///     a larger member (e.g. bge-large) exceeds bge-small's weights.
    public init(
        modelIds: [String] = ["BAAI/bge-small-en-v1.5"],
        estimatedBytes: Int = 512 * 1024 * 1024
    ) {
        precondition(
            !modelIds.isEmpty,
            "MLXEmbeddingModule needs at least one model id")
        self.allowedIds = modelIds
        self.defaultId = modelIds[0]
        self.estimatedBytes = estimatedBytes
    }

    public var residentBytes: Int { container == nil ? 0 : estimatedBytes }

    public func memoryEstimate() -> Int { estimatedBytes }

    public func load(reservation: MemoryReservation) async throws {
        if container != nil { return }
        try await loadContainer(defaultId)
    }

    public func unload() async {
        container = nil
        residentId = nil
    }

    public func allowedModelIds() -> [String] { allowedIds }
    public func defaultModelId() -> String { defaultId }
    public func residentModelId() -> String? { residentId }
    /// M41 explicit rebind: validate id ∈ allowlist and (when the slot
    /// is currently loaded) unload+reload to switch the resident model
    /// in place. If the slot is unloaded the call only stages the target
    /// id by triggering a fresh load — same fixed governor reservation.
    public func rebind(to id: String?) async throws {
        let target = id ?? defaultId
        guard allowedIds.contains(target) else {
            throw AthenaError.modelNotAvailable(
                requested: target, available: allowedIds)
        }
        if residentId == target, container != nil { return }
        container = nil
        residentId = nil
        try await loadContainer(target)
    }

    /// Load `id` into the single resident slot, replacing whatever was
    /// there. On failure the slot is left empty (container + residentId
    /// nil) so the next request re-attempts rather than wedging.
    private func loadContainer(_ id: String) async throws {
        do {
            container = try await EmbedderModelFactory.shared.loadContainer(
                from: #hubDownloader(
                    HuggingFace.HubClient(
                        session: AthenaProxy.proxiedURLSession())),
                using: #huggingFaceTokenizerLoader(),
                configuration: ModelConfiguration(id: id))
            residentId = id
        } catch {
            container = nil
            residentId = nil
            throw AthenaError.moduleLoadFailed(
                .textEmbedding,
                reason: "embedding model \(id): \(error)")
        }
    }

    /// Embed each input → an L2-normalized vector (order preserved),
    /// plus the total tokenized input length for usage accounting
    /// (M27.1). Mirrors the substrate's canonical
    /// tokenize→pad→mask→pool flow.
    ///
    /// M39: `model` selects from `allowedModelIds` (nil ⇒ default). If the
    /// requested model isn't the resident one, the slot is rebound
    /// (unload current → load requested) before embedding. An id outside
    /// the set throws `modelNotAvailable` (400) — never a silent
    /// wrong-dimension fallback, never an on-request download. The
    /// returned batch reports the id actually served.
    public func embed(_ texts: [String], model: String? = nil) async throws
        -> EmbeddingBatch
    {
        let target = model ?? defaultId
        guard allowedIds.contains(target) else {
            throw AthenaError.modelNotAvailable(
                requested: target, available: allowedIds)
        }
        if residentId != target || container == nil {
            container = nil
            residentId = nil
            try await loadContainer(target)
        }
        guard let container else {
            throw AthenaError.moduleLoadFailed(
                .textEmbedding, reason: "embed called before load")
        }
        if texts.isEmpty {
            return EmbeddingBatch(
                vectors: [], promptTokens: 0, model: target)
        }
        return await container.perform { ctx in
            let tokenizer = ctx.tokenizer
            let encoded = texts.map {
                tokenizer.encode(text: $0, addSpecialTokens: true)
            }
            let promptTokens = encoded.reduce(0) { $0 + $1.count }
            let maxLength = encoded.reduce(into: 1) {
                $0 = max($0, $1.count)
            }
            let pad = tokenizer.eosTokenId ?? 0
            let padded = stacked(
                encoded.map {
                    MLXArray(
                        $0
                            + Array(
                                repeating: pad,
                                count: maxLength - $0.count))
                })
            let mask = padded .!= pad
            let tokenTypes = MLXArray.zeros(like: padded)
            let out = ctx.model(
                padded, positionIds: nil, tokenTypeIds: tokenTypes,
                attentionMask: mask)
            let result = ctx.pooling(
                out, normalize: true, applyLayerNorm: true)
            result.eval()
            return EmbeddingBatch(
                vectors: result.map { $0.asArray(Float.self) },
                promptTokens: promptTokens, model: target)
        }
    }
}
