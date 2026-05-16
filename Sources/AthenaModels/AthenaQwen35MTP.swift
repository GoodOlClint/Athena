import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

// Multi-Token-Prediction (MTP) head for Qwen3.5 speculative decoding,
// ported from the GoodOlClint/mlx-lm `the consuming application-patches` fork
// (mlx_lm/models/qwen3_5.py MTPModule 292–352) — technique only.
//
// M2.2a scope: structure + weight binding only. The head loads from the
// checkpoint's `mtp.*` tensors and its forward is implemented faithfully,
// but it is NOT yet wired into the generation path — plain generation is
// unchanged. The draft/verify/accept loop is M2.2c.

/// `mtp.fc`: a 2H→H fused projection. Deliberately NOT `nn.Linear`: the
/// checkpoint stores `mtp.fc.weight` in **full precision** (no
/// scales/biases) while config.json declares uniform 4-bit quant with no
/// per-layer exception. The substrate's `quantize(model:)` only touches
/// `Linear`/`Embedding`, so a plain `Module` is skipped and binds the
/// full-precision weight correctly. (Matches the fork's `quant_predicate`
/// `mtp.fc` exclusion without needing to patch the substrate loader.)
final class AthenaMTPFusedProjection: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray

    init(inputDimensions: Int, outputDimensions: Int) {
        // nn.Linear(in, out, bias=false) weight layout: (out, in).
        self._weight.wrappedValue = MLXArray.zeros([
            outputDimensions, inputDimensions,
        ])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        matmul(x, weight.transposed())
    }
}

/// One MTP transformer block — full-attention only (no GatedDeltaNet
/// branch), otherwise the standard residual block. Reuses the backbone's
/// attention and MLP/MoE so weight keys match the checkpoint.
final class AthenaQwen35MTPDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: AthenaQwen35Attention
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm:
        RMSNorm
    @ModuleInfo(key: "mlp") var mlp: Module

    init(_ args: AthenaQwen35TextConfiguration) {
        _selfAttn.wrappedValue = AthenaQwen35Attention(args)
        if args.numExperts > 0 {
            _mlp.wrappedValue = AthenaQwen35SparseMoeBlock(args)
        } else {
            _mlp.wrappedValue = Qwen3NextMLP(
                dimensions: args.hiddenSize,
                hiddenDimensions: args.intermediateSize)
        }
        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let r = selfAttn(inputLayerNorm(x), mask: mask, cache: cache)
        let h = x + r
        return h + (mlp as! UnaryLayer)(postAttentionLayerNorm(h))
    }
}

/// The MTP module: predicts token t+2 from the backbone's PRE-norm hidden
/// at t and the just-sampled token t+1. Embedding + vocab projection are
/// SHARED with the backbone (passed in by the caller).
public final class AthenaQwen35MTPModule: Module {
    @ModuleInfo(key: "pre_fc_norm_hidden") var preFCNormHidden: RMSNorm
    @ModuleInfo(key: "pre_fc_norm_embedding") var preFCNormEmbedding: RMSNorm
    @ModuleInfo(key: "fc") var fc: AthenaMTPFusedProjection
    @ModuleInfo(key: "layers") var layers: [AthenaQwen35MTPDecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    public init(_ args: AthenaQwen35TextConfiguration) {
        _preFCNormHidden.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _preFCNormEmbedding.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _fc.wrappedValue = AthenaMTPFusedProjection(
            inputDimensions: args.hiddenSize * 2,
            outputDimensions: args.hiddenSize)
        _layers.wrappedValue = (0 ..< args.mtpNumHiddenLayers).map { _ in
            AthenaQwen35MTPDecoderLayer(args)
        }
        _norm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        super.init()
    }

    /// `hiddenStates`: backbone PRE-norm hidden (B,N,H). `nextTokenIds`:
    /// the just-sampled token id(s) (B,N). Returns normed (B,N,H); the
    /// caller applies the shared lm_head / embed.asLinear.
    public func callAsFunction(
        _ hiddenStates: MLXArray,
        nextTokenIds: MLXArray,
        embedTokens: Embedding,
        cache: [KVCache?]?
    ) -> MLXArray {
        let e = preFCNormEmbedding(embedTokens(nextTokenIds))
        let h = preFCNormHidden(hiddenStates)
        var fused = fc(concatenated([e, h], axis: -1))

        let mask = createAttentionMask(h: fused, cache: cache?.first ?? nil)
        for (i, layer) in layers.enumerated() {
            fused = layer(fused, mask: mask, cache: cache?[i])
        }
        return norm(fused)
    }
}
