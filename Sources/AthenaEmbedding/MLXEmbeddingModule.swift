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

    /// ADR 026 — the selectable set is the store's embedding models, scanned
    /// live via `ModelSupport`; `configuredDefault` is the per-module TOML
    /// default used when a request omits `model` (nil ⇒ ambiguity rule). The
    /// single resident slot is rebound to whichever model a request asks for;
    /// a model absent from the store is a `modelNotAvailable` 400 — a request
    /// can never trigger an arbitrary HF download.
    private let configuredDefault: String?
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
    /// NI3 — the operator's last selection (rebind/embed), staged so a
    /// governor evict→reload restores it instead of silently reverting the
    /// slot to the default. Survives `unload()`; `load(reservation:)`
    /// honors `desiredName ?? residentId ?? <resolved default>`, mirroring the
    /// LLM module's M62 cold-load-binds-requested-model seam.
    private var desiredName: String?
    /// Real weight footprint, summed at load by walking
    /// `model.parameters().flattened()`. Surfaced via `residentBytes` so
    /// `/healthz` reports what the loaded model actually holds instead
    /// of the static `estimatedBytes` (a 4B embedder's true footprint
    /// is several GB; the static estimate is sized for the bge-small
    /// default and lies by an order of magnitude for larger members).
    private var weightBytes: Int?
    /// I2 (M68.3) — serializes `embed` so two concurrent calls can't
    /// interleave their rebind + forward. Pre-fix, call A's `await
    /// loadContainer` / `await container.perform` released the actor, letting
    /// call B rebind the single slot to a DIFFERENT model; A then resumed,
    /// re-read `self.container` (now B's model), and embedded A's texts
    /// against the wrong model while reporting A's requested id (wrong-model
    /// vectors). New embeds chain after the in-flight one (FIFO) so each
    /// load→capture→forward runs atomically; the worker also captures its
    /// container handle locally instead of re-reading `self.container` after
    /// the await. Token-keyed cleanup so a newer embed isn't clobbered.
    private var embedInFlight: Task<EmbeddingBatch, Error>?
    private var embedSeq: UInt64 = 0
    private var embedGateSeq: UInt64 = 0
    /// I3 — only trim the MLX allocator pool after a LARGE bucket. A
    /// `clearCache` after every short-query bucket drops warm buffers the next
    /// call would reuse (and races concurrent global-pool users), turning the
    /// common length-1 query into pool thrash; the big-batch case (the one
    /// that actually drifts `residentBytes` into tens of GiB) is what needs
    /// the reclaim. 16 384 ≈ half the per-bucket token budget: a length-1
    /// query (work ≈ 1) skips; a real 32×512 / 13×2500 batch clears.
    private static let clearCacheWorkThreshold = 16_384

    /// - Parameters:
    ///   - configuredDefault: the per-module TOML default embedding id (ADR
    ///     026), used when a request omits `model` (nil ⇒ ambiguity rule).
    ///   - modelStoreRoot: the store root scanned for embedding models.
    ///   - estimatedBytes: governor admission estimate. Default 512 MiB:
    ///     safe headroom over bge-small's weights + tokenizer + the
    ///     transient activation working set. One model is resident at a
    ///     time, so the fixed estimate still bounds the slot even though
    ///     a larger member (e.g. bge-large) exceeds bge-small's weights.
    public init(
        configuredDefault: String? = nil,
        modelStoreRoot: URL? = nil,
        estimatedBytes: Int = 512 * 1024 * 1024
    ) {
        self.configuredDefault =
            (configuredDefault?.isEmpty == true) ? nil : configuredDefault
        self.modelStoreRoot = modelStoreRoot
        self.estimatedBytes = estimatedBytes
    }

    /// Local store directory for `id` (bare name or full HF id) when
    /// materialized — shared resolution (`ModelStoreLayout`) so every
    /// module class loads the same way.
    private func localDirectory(for id: String) -> URL? {
        ModelStoreLayout.localDirectory(for: id, storeRoot: modelStoreRoot)
    }

    /// The store's embedding models (ADR 026 live scan).
    private func storeModelIds() -> [String] {
        StoreModelClass.ids(
            storeRoot: modelStoreRoot, accept: { $0.isEmbeddingSlot })
    }

    /// ADR 026 resolution against the live store scan (used by rebind/embed).
    private func resolve(_ id: String?) throws -> String {
        let available = storeModelIds()
        switch ModelSelection.resolve(
            available: available, configuredDefault: configuredDefault,
            requested: id)
        {
        case .resolved(let t): return t
        case .notAvailable:
            throw AthenaError.modelNotAvailable(
                requested: id ?? (configuredDefault ?? ""),
                available: available)
        case .ambiguous:
            throw AthenaError.ambiguousModel(
                module: .textEmbedding, available: available)
        }
    }

    public var residentBytes: Int {
        container == nil ? 0 : (weightBytes ?? estimatedBytes)
    }

    public func memoryEstimate() -> Int { estimatedBytes }

    public func load(reservation: MemoryReservation) async throws {
        if container != nil { return }
        // NI3: honor the staged selection on a cold/reload, else the resolved
        // configured default (ADR 026 — throws if store empty/ambiguous).
        let name = try desiredName ?? residentId ?? resolve(nil)
        try await loadContainer(name)
    }

    public func unload() async {
        container = nil
        residentId = nil
        weightBytes = nil
    }

    public func allowedModelIds() -> [String] { storeModelIds() }
    public func defaultModelId() -> String {
        ModelSelection.displayDefault(
            available: storeModelIds(), configuredDefault: configuredDefault)
    }
    public func residentModelId() -> String? { residentId }
    /// M41 / ADR 026 explicit rebind: resolve `id` against the store (when the
    /// slot is loaded, unload+reload to switch in place; when unloaded, stage
    /// the target by triggering a fresh load — same fixed governor reservation).
    public func rebind(to id: String?) async throws {
        let target = try resolve(id)
        // NI3: remember the selection so a later evict→reload restores it.
        desiredName = target
        if residentId == target, container != nil { return }
        container = nil
        residentId = nil
        weightBytes = nil
        try await loadContainer(target)
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
        // ADR 026 — resolve `model` against the store (bare name or full HF
        // id; omit ⇒ configured default / sole model / ambiguity 400).
        let target = try resolve(model)
        // I2 — serialize: chain after any in-flight embed so the rebind +
        // forward for THIS request run atomically. Without it a concurrent
        // embed for a different model swaps the single slot mid-request and
        // the wrong model serves these texts.
        let prior = embedInFlight
        embedSeq &+= 1
        let mySeq = embedSeq
        let task = Task { () async throws -> EmbeddingBatch in
            _ = try? await prior?.value
            return try await self.embedSerialized(texts, target: target)
        }
        embedInFlight = task
        embedGateSeq = mySeq
        defer { if embedGateSeq == mySeq { embedInFlight = nil } }
        return try await task.value
    }

    /// NI6 — greedy length-bucketing packer (pure index/length bookkeeping, no
    /// MLX). Given each input's token length, return buckets of ORIGINAL
    /// indices: sort by length, then greedy-pack under `tokenBudget`
    /// (`count × bucket-maxLen`) and `maxItemsPerBucket`. A single item always
    /// fits its own bucket even if it alone exceeds the budget (it still has to
    /// be embedded; the NI4 ceiling rejects truly-oversized inputs upstream).
    /// Extracted from `embedSerialized` so the pack invariant (every index
    /// exactly once; per-bucket caps honored) is unit-testable on CI without a
    /// Metal device; the forward then reassembles `vectors[idx]` in input
    /// order, so this is the order-preservation seam too.
    static func lengthBuckets(
        tokenLengths: [Int], tokenBudget: Int, maxItemsPerBucket: Int
    ) -> [[Int]] {
        let order = (0..<tokenLengths.count).sorted {
            tokenLengths[$0] < tokenLengths[$1]
        }
        var buckets: [[Int]] = []
        var current: [Int] = []
        var currentMaxLen = 0
        for idx in order {
            let L = max(1, tokenLengths[idx])
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
        return buckets
    }

    /// I2 — the serialized embed worker. Runs under the FIFO chain above, so
    /// the rebind below is never concurrent with another embed; it also
    /// captures the container handle locally (`liveContainer`) rather than
    /// re-reading `self.container` after the load `await`.
    private func embedSerialized(
        _ texts: [String], target: String
    ) async throws -> EmbeddingBatch {
        // NI3: a per-request selection is also an operator choice — stage it
        // so a governor evict→reload restores the last-served model.
        desiredName = target
        if residentId != target || container == nil {
            container = nil
            residentId = nil
            weightBytes = nil
            try await loadContainer(target)
        }
        guard let liveContainer = container else {
            throw AthenaError.moduleLoadFailed(
                .textEmbedding, reason: "embed called before load")
        }
        if texts.isEmpty {
            return EmbeddingBatch(
                vectors: [], promptTokens: 0, model: target)
        }
        return try await liveContainer.perform { ctx in
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
            // NI6: greedy length-bucketing — pure index/length bookkeeping,
            // extracted to `lengthBuckets` so the pack/reassembly invariant is
            // CI-testable. 32 768 ≈ batch=64 × L=512 (typical embedding
            // workload): a bucket with L=2500 packs ~13 items; one with L=200
            // packs 64 (the hard per-bucket cap).
            let buckets = Self.lengthBuckets(
                tokenLengths: encoded.map(\.count),
                tokenBudget: 32_768, maxItemsPerBucket: 64)

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
                // I1/NI1/I4: derive the attention mask from each row's REAL
                // token length (floored to ≥1 so an empty input can't make
                // mean-pooling divide by zero → NaN), NOT pad-equality. A
                // real token whose id == pad (id 0 when eos is nil) is no
                // longer masked, and — critically — the SAME mask is now
                // handed to pooling, so padded positions can't pollute a
                // mean vector or be selected as the "last" token. Without it
                // the substrate defaults the pooling mask to all-ones and
                // pools over pad (corrupt vectors on any mixed-length batch).
                let lengths = MLXArray(
                    bucketEncoded.map { Int32(max(1, $0.count)) })
                let positions = MLXArray(0 ..< maxLength)
                let mask =
                    positions.expandedDimensions(axis: 0)
                    .< lengths.expandedDimensions(axis: 1)
                let tokenTypes = MLXArray.zeros(like: padded)
                let out = ctx.model(
                    padded, positionIds: nil,
                    tokenTypeIds: tokenTypes,
                    attentionMask: mask)
                // I5: do NOT apply the substrate's parameterless layerNorm.
                // Canonical sentence-transformers pooling for the configured
                // models (bge-small mean→normalize, Qwen3-Embedding
                // lasttoken→normalize) has no such step; forcing it on
                // produced non-canonical vectors (a spurious standardization
                // that changes direction, not just scale). PoolingConfiguration
                // carries no layernorm flag, so `false` is the model-faithful
                // value. `normalize: true` stays — Athena's documented
                // L2-normalized-vector contract, and both configured models
                // ship a Normalize module. NOTE: this changes every produced
                // vector vs ≤v0.10.129; any externally persisted embedding
                // index must be re-embedded to stay comparable with new queries.
                let result = ctx.pooling(
                    out, mask: mask, normalize: true, applyLayerNorm: false)
                result.eval()
                let bucketVectors = result.map {
                    $0.asArray(Float.self)
                }
                // I6: the produced vector width must match the model's
                // configured output dimension; a mismatch (or a zero-length
                // vector) means a wrong/mis-sanitized model and would
                // silently store wrong-width vectors. Fail loudly instead.
                if let bad = bucketVectors.first(where: { $0.isEmpty }) {
                    throw AthenaError.moduleLoadFailed(
                        .textEmbedding,
                        reason:
                            "model produced a zero-length embedding vector "
                            + "(\(bad.count))")
                }
                if let expected = ctx.pooling.dimension,
                    let bad = bucketVectors.first(where: {
                        $0.count != expected
                    })
                {
                    throw AthenaError.moduleLoadFailed(
                        .textEmbedding,
                        reason: "produced embedding dim \(bad.count) != "
                            + "model output dim \(expected)")
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
                // I3 — only for a LARGE bucket: a per-bucket clear on every
                // short query thrashes the warm pool the next call reuses.
                if maxLength * bucket.count >= Self.clearCacheWorkThreshold {
                    MLX.Memory.clearCache()
                }
            }
            return EmbeddingBatch(
                vectors: vectors, promptTokens: promptTokens,
                model: target)
        }
    }
}
