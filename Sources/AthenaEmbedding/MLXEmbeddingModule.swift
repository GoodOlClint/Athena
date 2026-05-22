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
public actor MLXEmbeddingModule: EmbeddingModule {
    public nonisolated let id: ModuleID = .textEmbedding

    private let modelId: String
    private let estimatedBytes: Int
    private var container: EmbedderModelContainer?

    /// - Parameters:
    ///   - modelId: HF id of the embedding model (default
    ///     `BAAI/bge-small-en-v1.5` — 384-dim, ~130 MB).
    ///   - estimatedBytes: governor admission estimate. Default 512 MiB:
    ///     safe headroom over bge-small's weights + tokenizer + the
    ///     transient activation working set.
    public init(
        modelId: String = "BAAI/bge-small-en-v1.5",
        estimatedBytes: Int = 512 * 1024 * 1024
    ) {
        self.modelId = modelId
        self.estimatedBytes = estimatedBytes
    }

    public var residentBytes: Int { container == nil ? 0 : estimatedBytes }

    public func memoryEstimate() -> Int { estimatedBytes }

    public func load(reservation: MemoryReservation) async throws {
        if container != nil { return }
        do {
            container = try await EmbedderModelFactory.shared.loadContainer(
                from: #hubDownloader(
                    HuggingFace.HubClient(
                        session: AthenaProxy.proxiedURLSession())),
                using: #huggingFaceTokenizerLoader(),
                configuration: ModelConfiguration(id: modelId))
        } catch {
            throw AthenaError.moduleLoadFailed(
                .textEmbedding,
                reason: "embedding model \(modelId): \(error)")
        }
    }

    public func unload() async {
        container = nil
    }

    /// Embed each input → an L2-normalized vector (order preserved),
    /// plus the total tokenized input length for usage accounting
    /// (M27.1). Mirrors the substrate's canonical
    /// tokenize→pad→mask→pool flow.
    public func embed(_ texts: [String]) async throws -> EmbeddingBatch {
        guard let container else {
            throw AthenaError.moduleLoadFailed(
                .textEmbedding, reason: "embed called before load")
        }
        if texts.isEmpty {
            return EmbeddingBatch(vectors: [], promptTokens: 0)
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
                promptTokens: promptTokens)
        }
    }
}
