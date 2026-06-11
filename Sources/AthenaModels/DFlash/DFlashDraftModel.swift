import Foundation
import MLX
import MLXLMCommon
import MLXNN

// DFlash speculative-draft model (Gemma4-first), vendored from
// bstnxbt/dflash-mlx `dflash_mlx/model.py` (Apache-2.0; paper
// arXiv:2602.06036) — see Sources/AthenaModels/DFlash/NOTICE. Technique +
// architecture port; weights are the operator-pulled z-lab drafter
// checkpoints (one per target).
//
// M63.1 scope: the draft module + its NO-CACHE block forward + the
// target-hidden projection, loaded and strict-bound from the drafter
// checkpoint. This is the surface the M63.1 parity test exercises. The
// incremental context caches (ContextOnlyDraftKVCache /
// FullContextDraftKVCache in the reference) and the verify/accept loop are
// M63.3 (the DFlash decode engine). Pure-MLX path only; the reference's
// optional `mx.fast.dflash_cross_attention` kernel (which has a pure-mx
// fallback in the reference too) is a deferred perf follow-up, so there is
// no Metal kernel here.
//
// The drafter shares the TARGET's token embedding and tied lm_head: the
// checkpoint carries NO embed/head tensors. The caller builds the noise-
// block embeddings from the target embedding and projects the draft's
// returned `norm(hidden)` through the target's tied head to get draft
// logits (M63.3). `embedScale` is likewise the target's embed scale,
// supplied by the caller via `bindTargetModel`.

/// DFlash drafter configuration (decoded from the drafter `config.json`).
public struct DFlashDraftConfiguration: Codable, Sendable {
    public var hiddenSize: Int
    public var numHiddenLayers: Int
    public var intermediateSize: Int
    public var numAttentionHeads: Int
    public var numKeyValueHeads: Int
    public var headDim: Int
    public var rmsNormEps: Float
    public var vocabSize: Int
    public var maxPositionEmbeddings: Int
    public var ropeTheta: Float
    public var ropeScaling: [String: StringOrNumber]?
    public var layerTypes: [String]
    public var slidingWindow: Int?
    public var numTargetLayers: Int
    public var blockSize: Int
    public var attentionBias: Bool
    public var finalLogitSoftcapping: Float?
    public var tieWordEmbeddings: Bool
    public var dflashConfig: DFlashSubConfig?

    public struct DFlashSubConfig: Codable, Sendable {
        public var maskTokenId: Int?
        public var targetLayerIds: [Int]?
        enum CodingKeys: String, CodingKey {
            case maskTokenId = "mask_token_id"
            case targetLayerIds = "target_layer_ids"
        }
    }

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case rmsNormEps = "rms_norm_eps"
        case vocabSize = "vocab_size"
        case maxPositionEmbeddings = "max_position_embeddings"
        case ropeTheta = "rope_theta"
        case ropeScaling = "rope_scaling"
        case layerTypes = "layer_types"
        case slidingWindow = "sliding_window"
        case numTargetLayers = "num_target_layers"
        case blockSize = "block_size"
        case attentionBias = "attention_bias"
        case finalLogitSoftcapping = "final_logit_softcapping"
        case tieWordEmbeddings = "tie_word_embeddings"
        case dflashConfig = "dflash_config"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        numHiddenLayers = try c.decode(Int.self, forKey: .numHiddenLayers)
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        numAttentionHeads = try c.decode(Int.self, forKey: .numAttentionHeads)
        numKeyValueHeads = try c.decode(Int.self, forKey: .numKeyValueHeads)
        headDim = try c.decode(Int.self, forKey: .headDim)
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        vocabSize = try c.decode(Int.self, forKey: .vocabSize)
        maxPositionEmbeddings =
            try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 262144
        ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 1_000_000
        ropeScaling = try c.decodeIfPresent(
            [String: StringOrNumber].self, forKey: .ropeScaling)
        layerTypes = try c.decodeIfPresent([String].self, forKey: .layerTypes) ?? []
        slidingWindow = try c.decodeIfPresent(Int.self, forKey: .slidingWindow)
        numTargetLayers = try c.decode(Int.self, forKey: .numTargetLayers)
        blockSize = try c.decode(Int.self, forKey: .blockSize)
        attentionBias = try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        finalLogitSoftcapping = try c.decodeIfPresent(
            Float.self, forKey: .finalLogitSoftcapping)
        tieWordEmbeddings =
            try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? true
        dflashConfig = try c.decodeIfPresent(DFlashSubConfig.self, forKey: .dflashConfig)
    }

    /// Target hidden-layer indices the draft conditions on (the
    /// `extract_context_feature` selection). The `fc` projection input
    /// width is `targetLayerIds.count * hiddenSize`.
    public var targetLayerIds: [Int] {
        if let ids = dflashConfig?.targetLayerIds, !ids.isEmpty { return ids }
        // Reference default `build_target_layer_ids` (model.py).
        let n = numTargetLayers, k = numHiddenLayers
        if k <= 1 { return [n / 2] }
        let start = 1, end = n - 3, span = end - start
        return (0 ..< k).map { i in
            Int((Double(start) + Double(i * span) / Double(k - 1)).rounded())
        }
    }
}

/// One DFlash draft attention block — plain Q/K/V (no Qwen3.5 gating),
/// per-head q/k RMSNorm, RoPE, cross-attention to the projected target
/// context concatenated with the in-block "noise" keys/values. NO-CACHE
/// forward (the M63.1 surface): context is RoPE'd at absolute positions
/// `0..<ctxLen`, the noise block at `ctxLen..<ctxLen+blockLen`. Faithful to
/// `DFlashAttention.__call__` (model.py) cache==None branch with the
/// pure-mx SDPA fallback.
final class DFlashAttention: Module {
    let nHeads: Int
    let nKVHeads: Int
    let headDim: Int
    let scale: Float
    let slidingWindow: Int?

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    let rope: RoPELayer

    init(_ args: DFlashDraftConfiguration, layerIndex: Int) {
        nHeads = args.numAttentionHeads
        nKVHeads = args.numKeyValueHeads
        headDim = args.headDim
        scale = pow(Float(args.headDim), -0.5)
        let layerType =
            layerIndex < args.layerTypes.count ? args.layerTypes[layerIndex] : ""
        slidingWindow =
            (layerType == "sliding_attention") ? args.slidingWindow : nil

        _qProj.wrappedValue = Linear(
            args.hiddenSize, nHeads * headDim, bias: args.attentionBias)
        _kProj.wrappedValue = Linear(
            args.hiddenSize, nKVHeads * headDim, bias: args.attentionBias)
        _vProj.wrappedValue = Linear(
            args.hiddenSize, nKVHeads * headDim, bias: args.attentionBias)
        _oProj.wrappedValue = Linear(
            nHeads * headDim, args.hiddenSize, bias: args.attentionBias)
        _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)

        rope = initializeRope(
            dims: headDim,
            base: args.ropeTheta,
            traditional: false,
            scalingConfig: args.ropeScaling,
            maxPositionEmbeddings: args.maxPositionEmbeddings)
        super.init()
    }

    /// Additive (0 / -inf) attention mask matching the reference
    /// `_attention_mask` for the no-cache case: query position
    /// `queryOffset+i` attends key position `j` iff `0 <= (qpos-kpos) <
    /// window`. Full-attention layers (`slidingWindow == nil`) return
    /// `.none` (unmasked over context+block).
    private func maskMode(
        blockLen: Int, queryOffset: Int, keyLen: Int
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        guard let window = slidingWindow else { return .none }
        let qpos = MLXArray(
            (0 ..< blockLen).map { Int32(queryOffset + $0) }
        ).reshaped(blockLen, 1)
        let kpos = MLXArray((0 ..< keyLen).map { Int32($0) }).reshaped(1, keyLen)
        let d = qpos - kpos
        // Boolean mask (True = attend), matching the reference
        // `_attention_mask` / `create_causal_mask`. A boolean SDPA mask is
        // handled directly by MLX; a float additive mask would have to
        // promote to the bf16 score dtype, which float32 cannot.
        let allow = logicalAnd(d .>= MLXArray(Int32(0)), d .< MLXArray(Int32(window)))
        return .array(allow)
    }

    /// `hidden`: noise-block embeddings (B, blockLen, H). `context`: the
    /// already-projected draft context (B, ctxLen, H) — i.e.
    /// `hidden_norm(fc(target_hidden))` from the model. Returns (B,
    /// blockLen, H).
    func callAsFunction(_ hidden: MLXArray, context: MLXArray) -> MLXArray {
        let B = hidden.dim(0)
        let blockLen = hidden.dim(1)
        let ctxLen = context.dim(1)

        var queries = qNorm(
            qProj(hidden).reshaped(B, blockLen, nHeads, headDim)
        ).transposed(0, 2, 1, 3)

        var contextKeys = kNorm(
            kProj(context).reshaped(B, ctxLen, nKVHeads, headDim)
        ).transposed(0, 2, 1, 3)
        let contextValues = vProj(context)
            .reshaped(B, ctxLen, nKVHeads, headDim).transposed(0, 2, 1, 3)

        var noiseKeys = kNorm(
            kProj(hidden).reshaped(B, blockLen, nKVHeads, headDim)
        ).transposed(0, 2, 1, 3)
        let noiseValues = vProj(hidden)
            .reshaped(B, blockLen, nKVHeads, headDim).transposed(0, 2, 1, 3)

        // No-cache RoPE offsets: context at 0, queries + noise keys at ctxLen.
        queries = applyRotaryPosition(rope, to: queries, offset: .scalar(ctxLen))
        contextKeys = applyRotaryPosition(rope, to: contextKeys, offset: .scalar(0))
        noiseKeys = applyRotaryPosition(rope, to: noiseKeys, offset: .scalar(ctxLen))

        let keys = concatenated([contextKeys, noiseKeys], axis: -2)
        let values = concatenated([contextValues, noiseValues], axis: -2)
        let mask = maskMode(
            blockLen: blockLen, queryOffset: ctxLen, keyLen: keys.dim(2))

        let output = attentionWithCacheUpdate(
            queries: queries, keys: keys, values: values,
            cache: nil, scale: scale, mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, blockLen, -1)
        return oProj(output)
    }
}

/// One DFlash decoder layer — standard pre-norm residual block (attention
/// then gated MLP). Reuses `Qwen3NextMLP` (gate_proj/up_proj/down_proj
/// SiLU), whose keys match the drafter checkpoint.
final class DFlashDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: DFlashAttention
    @ModuleInfo(key: "mlp") var mlp: Qwen3NextMLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ args: DFlashDraftConfiguration, layerIndex: Int) {
        _selfAttn.wrappedValue = DFlashAttention(args, layerIndex: layerIndex)
        _mlp.wrappedValue = Qwen3NextMLP(
            dimensions: args.hiddenSize, hiddenDimensions: args.intermediateSize)
        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, context: MLXArray) -> MLXArray {
        let h = x + selfAttn(inputLayerNorm(x), context: context)
        return h + mlp(postAttentionLayerNorm(h))
    }
}

/// The DFlash draft model. `fc` fuses the concatenated per-target-layer
/// hidden states (`targetLayerIds.count * H → H`), `hiddenNorm` normalizes
/// the projected context, and the decoder layers cross-attend to it from
/// the noise block. Returns `norm(hidden)`; the caller applies the
/// target's tied head.
public final class DFlashDraftModel: Module {
    public let config: DFlashDraftConfiguration
    public let targetLayerIds: [Int]
    public let blockSize: Int
    public let maskTokenId: Int

    /// Target embed scale (set via `bindTargetModel`). Default 1.0.
    public var embedScale: Float = 1.0

    @ModuleInfo(key: "layers") var layers: [DFlashDecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm
    @ModuleInfo(key: "fc") var fc: Linear
    @ModuleInfo(key: "hidden_norm") var hiddenNorm: RMSNorm

    public init(_ args: DFlashDraftConfiguration) {
        config = args
        targetLayerIds = args.targetLayerIds
        blockSize = args.blockSize
        maskTokenId = args.dflashConfig?.maskTokenId ?? 0
        _layers.wrappedValue = (0 ..< args.numHiddenLayers).map {
            DFlashDecoderLayer(args, layerIndex: $0)
        }
        _norm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _fc.wrappedValue = Linear(
            targetLayerIds.count * args.hiddenSize, args.hiddenSize, bias: false)
        _hiddenNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        super.init()
    }

    /// Adopt the target's token-embedding scale (Gemma4 scales embeddings
    /// by sqrt(hidden_size)). Mirrors `DFlashDraftModel.bind_target_model`.
    public func bindTargetModel(embedScale: Float) {
        self.embedScale = embedScale
    }

    /// `hidden_norm(fc(targetHidden))`. `targetHidden`: concatenated
    /// captured target hidden states (B, ctx, targetLayerIds.count * H).
    public func projectTargetHidden(_ targetHidden: MLXArray) -> MLXArray {
        hiddenNorm(fc(targetHidden))
    }

    /// No-cache block forward. `noiseEmbedding`: (B, blockSize, H) from the
    /// target embedding. `targetHidden`: (B, ctx, targetLayerIds.count * H).
    /// Returns `norm(hidden)` (B, blockSize, H).
    public func callAsFunction(
        noiseEmbedding: MLXArray, targetHidden: MLXArray
    ) -> MLXArray {
        let context = projectTargetHidden(targetHidden)
        var h = noiseEmbedding * embedScale
        for layer in layers {
            h = layer(h, context: context)
        }
        return norm(h)
    }
}
