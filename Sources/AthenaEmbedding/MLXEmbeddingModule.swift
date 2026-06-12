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
    private var allowedIds: [String]
    private var defaultId: String
    private let estimatedBytes: Int
    /// Model-store root, so a configured id resolves to its local store
    /// directory and loads from there (like the LLM loader) — which lets
    /// a request name the model by either its full HF id or its bare
    /// store-dir name. nil ⇒ pre-store-root callers; falls back to
    /// loading by HF id via the Hub.
    private let modelStoreRoot: URL?
    private var container: EmbedderModelContainer?
    /// The id currently resident in `container` (nil ⇒ nothing loaded).
    private var residentId: String?
    /// Real weight footprint, summed at load by walking
    /// `model.parameters().flattened()`. Surfaced via `residentBytes` so
    /// `/healthz` reports what the loaded model actually holds instead
    /// of the static `estimatedBytes` (a 4B embedder's true footprint
    /// is several GB; the static estimate is sized for the bge-small
    /// default and lies by an order of magnitude for larger members).
    private var weightBytes: Int?

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
        modelStoreRoot: URL? = nil,
        estimatedBytes: Int = 512 * 1024 * 1024
    ) {
        precondition(
            !modelIds.isEmpty,
            "MLXEmbeddingModule needs at least one model id")
        self.allowedIds = modelIds
        self.defaultId = modelIds[0]
        self.modelStoreRoot = modelStoreRoot
        self.estimatedBytes = estimatedBytes
    }

    /// Local store directory for `id` (bare name or full HF id) when
    /// materialized — shared resolution (`ModelStoreLayout`) so every
    /// module class loads the same way.
    private func localDirectory(for id: String) -> URL? {
        ModelStoreLayout.localDirectory(for: id, storeRoot: modelStoreRoot)
    }

    public var residentBytes: Int {
        container == nil ? 0 : (weightBytes ?? estimatedBytes)
    }

    public func memoryEstimate() -> Int { estimatedBytes }

    public func load(reservation: MemoryReservation) async throws {
        if container != nil { return }
        try await loadContainer(defaultId)
    }

    public func unload() async {
        container = nil
        residentId = nil
        weightBytes = nil
    }

    public func allowedModelIds() -> [String] { allowedIds }
    public func defaultModelId() -> String { defaultId }
    public func residentModelId() -> String? { residentId }
    /// M41 explicit rebind: validate id ∈ allowlist and (when the slot
    /// is currently loaded) unload+reload to switch the resident model
    /// in place. If the slot is unloaded the call only stages the target
    /// id by triggering a fresh load — same fixed governor reservation.
    public func rebind(to id: String?) async throws {
        let requested = id ?? defaultId
        // M54 — match by store-dir identity so a request naming the model
        // by either its full HF id or its bare store-dir name resolves the
        // same allowlist row (like the LLM loader). Canonical id from
        // storage drives the load.
        guard let target =
            allowedIds.canonicalByStoreIdentity(requested)
        else {
            throw AthenaError.modelNotAvailable(
                requested: requested, available: allowedIds)
        }
        if residentId == target, container != nil { return }
        container = nil
        residentId = nil
        weightBytes = nil
        try await loadContainer(target)
    }

    public func setAllowedModelIds(_ ids: [String]) {
        allowedIds = ids
        defaultId = ids.first ?? defaultId
        if let r = residentId, !ids.contains(r) {
            container = nil
            residentId = nil
            weightBytes = nil
        }
    }

    /// Load `id` into the single resident slot, replacing whatever was
    /// there. On failure the slot is left empty (container + residentId
    /// nil) so the next request re-attempts rather than wedging.
    private func loadContainer(_ id: String) async throws {
        // M54 — load from the LOCAL store directory (like the LLM loader):
        // works for a bare store-dir name AND a full HF id, resolving the
        // `pull`-created symlink to the real HF-cache snapshot. M54.3 —
        // inference NEVER auto-downloads: a model absent from the store is
        // a hard error (operator pulls it at startup / allowlist-add).
        guard let dir = localDirectory(for: id) else {
            throw AthenaError.moduleLoadFailed(
                .textEmbedding,
                reason: "model '\(id)' is not in the model store — pull "
                    + "it first (operator action); inference does not "
                    + "auto-download")
        }
        do {
            let loaded =
                try await EmbedderModelFactory.shared.loadContainer(
                    from: #hubDownloader(
                        HuggingFace.HubClient(
                            session: AthenaProxy.proxiedURLSession())),
                    using: #huggingFaceTokenizerLoader(),
                    configuration: ModelConfiguration(
                        directory: dir.resolvingSymlinksInPath()))
            container = loaded
            residentId = id
            // Sum `nbytes` over every parameter array — the substrate's
            // wired-memory.md calls this "the most accurate approach"
            // and the measurements there put it within ~MB of MLX's own
            // active-memory delta. `nbytes` is metadata-only (no
            // realization cost), so this is essentially free.
            weightBytes = await loaded.perform { ctx in
                ctx.model.parameters().flattened()
                    .reduce(0) { $0 + $1.1.nbytes }
            }
        } catch {
            container = nil
            residentId = nil
            weightBytes = nil
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
    /// Per-input token ceiling (NI4). Matches the per-batch `tokenBudget`
    /// below: a single input may be as large as one full bucket of work,
    /// but no larger — above this it's rejected with a 400 rather than
    /// driving an unbounded O(L²) forward pass via a singleton bucket.
    static let maxInputTokens = 32_768

    public func embed(_ texts: [String], model: String? = nil) async throws
        -> EmbeddingBatch
    {
        let requested = model ?? defaultId
        // M54 — match by store-dir identity (bare name or full HF id).
        guard let target =
            allowedIds.canonicalByStoreIdentity(requested)
        else {
            throw AthenaError.modelNotAvailable(
                requested: requested, available: allowedIds)
        }
        if residentId != target || container == nil {
            container = nil
            residentId = nil
            weightBytes = nil
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
        return try await container.perform { ctx in
            let tokenizer = ctx.tokenizer
            let encoded = texts.map {
                tokenizer.encode(text: $0, addSpecialTokens: true)
            }
            // NI4: reject any single input above the per-input token
            // ceiling with a 400, before padding/forward. A lone oversized
            // input otherwise gets its own singleton bucket — bypassing the
            // per-batch token budget below — and drives an unbounded
            // O(L²) forward pass on a RoPE embedder (remote OOM/compute
            // DoS); OpenAI likewise 400s over-length input rather than
            // returning a silently-truncated vector.
            if let longest = encoded.map(\.count).max(),
                longest > Self.maxInputTokens
            {
                throw AthenaError.inputTooLong(
                    module: .textEmbedding, tokens: longest,
                    maxTokens: Self.maxInputTokens)
            }
            let promptTokens = encoded.reduce(0) { $0 + $1.count }
            let pad = tokenizer.eosTokenId ?? 0

            // Length-bucketing. The substrate's pad-to-max in a single
            // batch makes one 2500-token outlier in a batch of 64 cost
            // as if all 64 were 2500 tokens — attention is O(B·L²) and
            // activations are O(B·L·H). On a corpus that mixes ~400-
            // token text with multi-thousand-token sections, that
            // produced 100× per-batch latency outliers and enough
            // allocator/kernel pressure to evict cold weight pages,
            // forcing the *next* batch to fault them back in from
            // SSD. Sort by token length, then greedy-pack mini-
            // batches under a fixed `count × maxLen` token budget:
            // short texts pack many-per-batch, long texts pack few-
            // per-batch, no item is ever padded by more than the
            // span of its own bucket. Results are reassembled in
            // input order so the EmbeddingBatch contract is unchanged.
            let order = (0..<encoded.count).sorted {
                encoded[$0].count < encoded[$1].count
            }
            // 32 768 ≈ batch=64 × L=512 (typical embedding workload).
            // A bucket with L=2500 packs ~13 items; one with L=200
            // packs 64 (hard cap below).
            let tokenBudget = 32_768
            let maxItemsPerBucket = 64
            var buckets: [[Int]] = []
            var current: [Int] = []
            var currentMaxLen = 0
            for idx in order {
                let L = max(1, encoded[idx].count)
                let nextMaxLen = max(currentMaxLen, L)
                let nextWork = (current.count + 1) * nextMaxLen
                if !current.isEmpty
                    && (nextWork > tokenBudget
                        || current.count >= maxItemsPerBucket)
                {
                    buckets.append(current)
                    current = [idx]
                    currentMaxLen = L
                } else {
                    current.append(idx)
                    currentMaxLen = nextMaxLen
                }
            }
            if !current.isEmpty { buckets.append(current) }

            var vectors = [[Float]](
                repeating: [], count: texts.count)
            for bucket in buckets {
                let bucketEncoded = bucket.map { encoded[$0] }
                let maxLength = bucketEncoded.reduce(into: 1) {
                    $0 = max($0, $1.count)
                }
                let padded = stacked(
                    bucketEncoded.map {
                        MLXArray(
                            $0
                                + Array(
                                    repeating: pad,
                                    count: maxLength - $0.count))
                    })
                let mask = padded .!= pad
                let tokenTypes = MLXArray.zeros(like: padded)
                let out = ctx.model(
                    padded, positionIds: nil,
                    tokenTypeIds: tokenTypes,
                    attentionMask: mask)
                let result = ctx.pooling(
                    out, normalize: true, applyLayerNorm: true)
                result.eval()
                let bucketVectors = result.map {
                    $0.asArray(Float.self)
                }
                for (k, idx) in bucket.enumerated() {
                    vectors[idx] = bucketVectors[k]
                }
                // M46.6 — release the per-bucket transient buffers
                // (padded/mask/tokenTypes/out/result) back to the system.
                // Without this the MLX allocator pool keeps the
                // activation buffers in cache across calls; a 4B
                // embedder's residentBytes drifts from its ~8 GiB
                // weight footprint up to tens of GiB after a handful
                // of mixed-length batches, eating budget the LLM needs
                // and pushing total RSS into unified-memory-thrash
                // territory. Mirrors the LLM decode loops' periodic
                // clearCache (e.g. SpeculativeGeneration.swift:184) —
                // the activations are bucket-scoped, so a per-bucket
                // clear is the embedding analog of "every 256 tokens."
                // bucketVectors above is already a Swift [Float] copy
                // before this line, so the clear cannot reclaim
                // anything the caller still needs.
                MLX.Memory.clearCache()
            }
            return EmbeddingBatch(
                vectors: vectors, promptTokens: promptTokens,
                model: target)
        }
    }
}
