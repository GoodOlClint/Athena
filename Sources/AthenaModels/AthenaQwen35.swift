//
//  Qwen35.swift
//  mlx-swift-lm
//
//  Created by John Mai on 2026/2/9.
//
//  Port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen3_5.py
//

import Dispatch
import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

/// Env-gated (`ATHENA_FWD_PROFILE=1`) per-block forward profiler. Splits the
/// decoder-layer wall time into token-mixing (GDN / attention) vs MLP by
/// forcing `eval()` at the block boundary. Perturbs absolute timing (it
/// serializes the lazy graph), so read the RATIOS. Zero cost when unset.
public enum ForwardProfile {
    public nonisolated(unsafe) static var enabled =
        ProcessInfo.processInfo.environment["ATHENA_FWD_PROFILE"] == "1"
    public nonisolated(unsafe) static var tGDN = 0.0
    public nonisolated(unsafe) static var tAttn = 0.0
    public nonisolated(unsafe) static var tMLP = 0.0
    public nonisolated(unsafe) static var nGDN = 0
    public nonisolated(unsafe) static var nAttn = 0
    public static func reset() { tGDN = 0; tAttn = 0; tMLP = 0; nGDN = 0; nAttn = 0 }
    public static func summary() -> String {
        let tot = tGDN + tAttn + tMLP
        func p(_ x: Double) -> String {
            tot > 0 ? String(format: "%.1f%%", x / tot * 100) : "0%"
        }
        return String(
            format:
                "fwd profile: total=%.3fs gdn=%.3fs(%@) attn=%.3fs(%@) "
                + "mlp=%.3fs(%@) gdnLayers=%d attnLayers=%d",
            tot, tGDN, p(tGDN), tAttn, p(tAttn), tMLP, p(tMLP), nGDN, nAttn)
    }
    @inline(__always) static func clk() -> UInt64 {
        DispatchTime.now().uptimeNanoseconds
    }
}

// MARK: - Configuration

private enum RopeParametersCodingKey: String, CodingKey {
    case ropeParameters = "rope_parameters"
}

public struct AthenaQwen35TextConfiguration: Codable, Sendable {
    var modelType: String = ""
    var hiddenSize: Int = 4096
    var hiddenLayers: Int = 32
    var intermediateSize: Int = 14336
    var attentionHeads: Int = 32
    var kvHeads: Int = 8
    var linearNumValueHeads: Int = 64
    var linearNumKeyHeads: Int = 16
    var linearKeyHeadDim: Int = 192
    var linearValueHeadDim: Int = 128
    var linearConvKernelDim: Int = 4
    var rmsNormEps: Float = 1e-6
    var vocabularySize: Int = 151_936
    var ropeTheta: Float = 100000.0
    var partialRotaryFactor: Float = 0.25
    var maxPositionEmbeddings: Int = 131072
    var tieWordEmbeddings: Bool = false
    var attentionBias: Bool = false
    var headDim: Int?
    var ropeScaling: [String: StringOrNumber]?
    var fullAttentionInterval: Int = 4

    // MoE fields
    var numExperts: Int = 0
    var numExpertsPerTok: Int = 0
    var decoderSparseStep: Int = 1
    var sharedExpertIntermediateSize: Int = 0
    var moeIntermediateSize: Int = 0
    var normTopkProb: Bool = true

    // MTP (multi-token-prediction) speculative-decoding head. 0 ⇒ absent.
    var mtpNumHiddenLayers: Int = 0

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case linearNumValueHeads = "linear_num_value_heads"
        case linearNumKeyHeads = "linear_num_key_heads"
        case linearKeyHeadDim = "linear_key_head_dim"
        case linearValueHeadDim = "linear_value_head_dim"
        case linearConvKernelDim = "linear_conv_kernel_dim"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case ropeTheta = "rope_theta"
        case partialRotaryFactor = "partial_rotary_factor"
        case maxPositionEmbeddings = "max_position_embeddings"
        case tieWordEmbeddings = "tie_word_embeddings"
        case attentionBias = "attention_bias"
        case headDim = "head_dim"
        case ropeScaling = "rope_scaling"
        case fullAttentionInterval = "full_attention_interval"
        case numExperts = "num_experts"
        case numExpertsPerTok = "num_experts_per_tok"
        case decoderSparseStep = "decoder_sparse_step"
        case sharedExpertIntermediateSize = "shared_expert_intermediate_size"
        case moeIntermediateSize = "moe_intermediate_size"
        case normTopkProb = "norm_topk_prob"
        case mtpNumHiddenLayers = "mtp_num_hidden_layers"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let defaultRopeParameters: [String: StringOrNumber] = [
            "type": .string("default"),
            "mrope_section": .ints([11, 11, 10]),
            "rope_theta": .float(100000.0),
            "partial_rotary_factor": .float(0.25),
        ]

        self.modelType = try container.decodeIfPresent(String.self, forKey: .modelType) ?? ""
        self.hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 4096
        self.hiddenLayers = try container.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 32
        self.intermediateSize =
            try container.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 14336
        self.attentionHeads = try container.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 32
        self.kvHeads = try container.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 8
        self.linearNumValueHeads =
            try container.decodeIfPresent(Int.self, forKey: .linearNumValueHeads) ?? 64
        self.linearNumKeyHeads =
            try container.decodeIfPresent(Int.self, forKey: .linearNumKeyHeads) ?? 16
        self.linearKeyHeadDim =
            try container.decodeIfPresent(Int.self, forKey: .linearKeyHeadDim) ?? 192
        self.linearValueHeadDim =
            try container.decodeIfPresent(Int.self, forKey: .linearValueHeadDim) ?? 128
        self.linearConvKernelDim =
            try container.decodeIfPresent(Int.self, forKey: .linearConvKernelDim) ?? 4
        self.rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        self.vocabularySize =
            try container.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 151_936
        self.maxPositionEmbeddings =
            try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 131072
        self.tieWordEmbeddings =
            try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        self.attentionBias =
            try container.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        self.headDim = try container.decodeIfPresent(Int.self, forKey: .headDim)
        self.fullAttentionInterval =
            try container.decodeIfPresent(Int.self, forKey: .fullAttentionInterval) ?? 4

        // MoE fields
        self.numExperts = try container.decodeIfPresent(Int.self, forKey: .numExperts) ?? 0
        self.numExpertsPerTok =
            try container.decodeIfPresent(Int.self, forKey: .numExpertsPerTok) ?? 0
        self.decoderSparseStep =
            try container.decodeIfPresent(Int.self, forKey: .decoderSparseStep) ?? 1
        self.sharedExpertIntermediateSize =
            try container.decodeIfPresent(Int.self, forKey: .sharedExpertIntermediateSize) ?? 0
        self.moeIntermediateSize =
            try container.decodeIfPresent(Int.self, forKey: .moeIntermediateSize) ?? 0
        self.normTopkProb = try container.decodeIfPresent(Bool.self, forKey: .normTopkProb) ?? true
        self.mtpNumHiddenLayers =
            try container.decodeIfPresent(Int.self, forKey: .mtpNumHiddenLayers) ?? 0

        let ropeContainer = try decoder.container(keyedBy: RopeParametersCodingKey.self)
        let ropeParameters = try ropeContainer.decodeIfPresent(
            [String: StringOrNumber].self, forKey: .ropeParameters)

        if var ropeParameters {
            if ropeParameters["type"] == nil, let ropeType = ropeParameters["rope_type"] {
                ropeParameters["type"] = ropeType
            }
            self.ropeTheta = ropeParameters["rope_theta"]?.asFloat() ?? 100000.0
            self.partialRotaryFactor =
                ropeParameters["partial_rotary_factor"]?.asFloat() ?? 0.25
            self.ropeScaling = ropeParameters
        } else {
            self.ropeTheta =
                try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 100000.0
            self.partialRotaryFactor =
                try container.decodeIfPresent(Float.self, forKey: .partialRotaryFactor) ?? 0.25
            self.ropeScaling =
                try container.decodeIfPresent([String: StringOrNumber].self, forKey: .ropeScaling)
                ?? defaultRopeParameters
        }

        if self.headDim == nil {
            self.headDim = self.hiddenSize / self.attentionHeads
        }
    }

    /// A copy with the MTP head disabled. `mtp_num_hidden_layers` is an
    /// architectural constant present even in stock checkpoints that ship
    /// no `mtp.*` weights; the registry uses this to suppress the head
    /// when the checkpoint lacks those tensors.
    public func withMTPDisabled() -> AthenaQwen35TextConfiguration {
        var copy = self
        copy.mtpNumHiddenLayers = 0
        return copy
    }
}

// MARK: - GatedDeltaNet

/// Side channel for MTP single-pass verify (MambaCache is `public` but not
/// `open`, so it can't be subclassed). During an `nConfirmed` split the
/// GatedDeltaNet layers stash their post-confirmed (conv, ssm) state here,
/// keyed by cache-object identity; the speculative loop restores from it on
/// draft rejection. nil ⇒ no MTP path is active (zero overhead/effect).
public final class GDNRollback: @unchecked Sendable {
    public var states: [ObjectIdentifier: [MLXArray]] = [:]
    public init() {}
}

final class AthenaQwen35GatedDeltaNet: Module {
    let hiddenSize: Int
    let numVHeads: Int
    let numKHeads: Int
    let headKDim: Int
    let headVDim: Int
    let keyDim: Int
    let valueDim: Int
    let convKernelSize: Int
    let convDim: Int

    @ModuleInfo(key: "conv1d") var conv1d: Conv1d
    @ModuleInfo(key: "in_proj_qkv") var inProjQKV: Linear
    @ModuleInfo(key: "in_proj_z") var inProjZ: Linear
    @ModuleInfo(key: "in_proj_b") var inProjB: Linear
    @ModuleInfo(key: "in_proj_a") var inProjA: Linear

    @ParameterInfo(key: "dt_bias") var dtBias: MLXArray
    @ParameterInfo(key: "A_log") var aLog: MLXArray

    @ModuleInfo(key: "norm") var norm: Qwen3NextRMSNormGated
    @ModuleInfo(key: "out_proj") var outProj: Linear

    init(_ args: AthenaQwen35TextConfiguration) {
        self.hiddenSize = args.hiddenSize
        self.numVHeads = args.linearNumValueHeads
        self.numKHeads = args.linearNumKeyHeads
        self.headKDim = args.linearKeyHeadDim
        self.headVDim = args.linearValueHeadDim
        self.keyDim = headKDim * numKHeads
        self.valueDim = headVDim * numVHeads
        self.convKernelSize = args.linearConvKernelDim
        self.convDim = keyDim * 2 + valueDim

        precondition(
            numVHeads % numKHeads == 0,
            "num_v_heads (\(numVHeads)) must be divisible by num_k_heads (\(numKHeads))"
        )

        _conv1d.wrappedValue = Conv1d(
            inputChannels: convDim,
            outputChannels: convDim,
            kernelSize: convKernelSize,
            stride: 1,
            padding: 0,
            dilation: 1,
            groups: convDim,
            bias: false
        )

        _inProjQKV.wrappedValue = Linear(hiddenSize, keyDim * 2 + valueDim, bias: false)
        _inProjZ.wrappedValue = Linear(hiddenSize, valueDim, bias: false)
        _inProjB.wrappedValue = Linear(hiddenSize, numVHeads, bias: false)
        _inProjA.wrappedValue = Linear(hiddenSize, numVHeads, bias: false)

        _dtBias.wrappedValue = MLXArray.ones([numVHeads])
        let a = MLXRandom.uniform(low: 0, high: 16, [numVHeads])
        _aLog.wrappedValue = log(a)

        _norm.wrappedValue = Qwen3NextRMSNormGated(dimensions: headVDim, eps: args.rmsNormEps)
        _outProj.wrappedValue = Linear(valueDim, hiddenSize, bias: false)

        super.init()
    }

    func callAsFunction(
        _ inputs: MLXArray,
        mask: MLXArray? = nil,
        cache: MambaCache? = nil,
        nConfirmed: Int = 0,
        rollback: GDNRollback? = nil
    ) -> MLXArray {
        let B = inputs.dim(0)
        let S = inputs.dim(1)

        // MTP single-pass verify: process the confirmed prefix, snapshot
        // the post-confirmed recurrent state, then the draft suffix —
        // carrying conv+ssm state through the existing per-chunk path.
        // Sub-chunking with carried state is mathematically identical to
        // one chunk (the cached-decode invariant), so output stays
        // bit-identical; this just exposes the intermediate state for
        // rollback on draft rejection. Default nConfirmed == 0 ⇒ the
        // original single-chunk path, byte-unchanged for every other
        // caller.
        if let cache, nConfirmed > 0, nConfirmed < S {
            let pre = inputs[0..., 0 ..< nConfirmed, 0...]
            let suf = inputs[0..., nConfirmed ..< S, 0...]
            let mPre = mask.map { $0[0..., 0 ..< nConfirmed] }
            let mSuf = mask.map { $0[0..., nConfirmed ..< S] }
            let outPre = callAsFunction(pre, mask: mPre, cache: cache)
            if let rollback, let c0 = cache[0], let c1 = cache[1] {
                rollback.states[ObjectIdentifier(cache)] = [
                    c0[.ellipsis], c1[.ellipsis],
                ]
            }
            let outSuf = callAsFunction(suf, mask: mSuf, cache: cache)
            return concatenated([outPre, outSuf], axis: 1)
        }

        var qkv = inProjQKV(inputs)
        let z = inProjZ(inputs).reshaped(B, S, numVHeads, headVDim)
        let b = inProjB(inputs)
        let a = inProjA(inputs)

        let convState: MLXArray
        if let cacheState = cache?[0] {
            convState = cacheState
        } else {
            convState = MLXArray.zeros([B, convKernelSize - 1, convDim], dtype: inputs.dtype)
        }

        if let mask {
            qkv = MLX.where(mask[.ellipsis, .newAxis], qkv, 0)
        }

        let convInput = concatenated([convState, qkv], axis: 1)
        if let cache {
            cache[0] = convInput[0..., (-(convKernelSize - 1))...]
        }

        let convOut = silu(conv1d(convInput))

        let convSplit = MLX.split(convOut, indices: [keyDim, 2 * keyDim], axis: -1)
        let q = convSplit[0].reshaped(B, S, numKHeads, headKDim)
        let k = convSplit[1].reshaped(B, S, numKHeads, headKDim)
        let v = convSplit[2].reshaped(B, S, numVHeads, headVDim)

        var state = cache?[1]
        let dtype = q.dtype
        let invScale = pow(Float(headKDim), -0.5)
        let qNormed =
            MLXArray(pow(invScale, 2)).asType(dtype)
            * MLXFast.rmsNorm(q, weight: MLXArray.mlxNone, eps: 1e-6)
        let kNormed =
            MLXArray(invScale).asType(dtype)
            * MLXFast.rmsNorm(k, weight: MLXArray.mlxNone, eps: 1e-6)

        var out: MLXArray

        (out, state) = gatedDeltaUpdate(
            q: qNormed,
            k: kNormed,
            v: v,
            a: a,
            b: b,
            aLog: aLog,
            dtBias: dtBias,
            state: state,
            mask: mask
        )

        if let cache {
            cache[1] = state
        }

        out = norm(out, gate: z)
        return outProj(out.reshaped(B, S, -1))
    }
}

// MARK: - Attention

final class AthenaQwen35Attention: Module {
    let attentionHeads: Int
    let kvHeads: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    let rope: RoPELayer

    init(_ args: AthenaQwen35TextConfiguration) {
        let headDim = args.headDim ?? (args.hiddenSize / args.attentionHeads)
        self.attentionHeads = args.attentionHeads
        self.kvHeads = args.kvHeads
        self.scale = pow(Float(headDim), -0.5)

        _qProj.wrappedValue = Linear(
            args.hiddenSize, args.attentionHeads * headDim * 2, bias: args.attentionBias)
        _kProj.wrappedValue = Linear(
            args.hiddenSize, args.kvHeads * headDim, bias: args.attentionBias)
        _vProj.wrappedValue = Linear(
            args.hiddenSize, args.kvHeads * headDim, bias: args.attentionBias)
        _oProj.wrappedValue = Linear(
            args.attentionHeads * headDim, args.hiddenSize, bias: args.attentionBias)

        _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)

        let ropeDims = Int(Float(headDim) * args.partialRotaryFactor)
        self.rope = initializeRope(
            dims: max(1, ropeDims),
            base: args.ropeTheta,
            traditional: false,
            scalingConfig: args.ropeScaling,
            maxPositionEmbeddings: args.maxPositionEmbeddings
        )

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)

        let qProjOutput = qProj(x)
        let qSplit = qProjOutput.reshaped(B, L, attentionHeads, -1).split(parts: 2, axis: -1)
        var queries = qSplit[0]
        let gate = qSplit[1].reshaped(B, L, -1)

        var keys = kProj(x)
        var values = vProj(x)

        queries = qNorm(queries).transposed(0, 2, 1, 3)
        keys = kNorm(keys.reshaped(B, L, kvHeads, -1)).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, kvHeads, -1).transposed(0, 2, 1, 3)

        let offset = cache?.ropeOffset
        queries = applyRotaryPosition(rope, to: queries, offset: offset)
        keys = applyRotaryPosition(rope, to: keys, offset: offset)

        let output = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: scale,
            mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return oProj(sigmoidMultiply(output, gate))
    }
}

// MARK: - SparseMoeBlock

final class AthenaQwen35SparseMoeBlock: Module, UnaryLayer {
    let normTopkProb: Bool
    let numExperts: Int
    let topK: Int

    @ModuleInfo(key: "gate") var gate: Linear
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU

    @ModuleInfo(key: "shared_expert") var sharedExpert: Qwen3NextMLP
    @ModuleInfo(key: "shared_expert_gate") var sharedExpertGate: Linear

    init(_ args: AthenaQwen35TextConfiguration) {
        self.normTopkProb = args.normTopkProb
        self.numExperts = args.numExperts
        self.topK = args.numExpertsPerTok

        _gate.wrappedValue = Linear(args.hiddenSize, args.numExperts, bias: false)
        _switchMLP.wrappedValue = SwitchGLU(
            inputDims: args.hiddenSize,
            hiddenDims: args.moeIntermediateSize,
            numExperts: args.numExperts
        )

        _sharedExpert.wrappedValue = Qwen3NextMLP(
            dimensions: args.hiddenSize,
            hiddenDimensions: args.sharedExpertIntermediateSize
        )
        _sharedExpertGate.wrappedValue = Linear(args.hiddenSize, 1, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var gates = gate(x)
        gates = MLX.softmax(gates, axis: -1, precise: true)

        let k = topK
        let kth = gates.dim(-1) - k
        let inds = MLX.argPartition(gates, kth: kth, axis: -1)[.ellipsis, (kth)...]
        var scores = MLX.takeAlong(gates, inds, axis: -1)
        if normTopkProb {
            scores = scores / scores.sum(axis: -1, keepDims: true)
        }

        let y = switchMLP(x, inds)
        let combined = (y * scores[.ellipsis, .newAxis]).sum(axis: -2)

        var sharedY = sharedExpert(x)
        sharedY = sigmoid(sharedExpertGate(x)) * sharedY

        return combined + sharedY
    }
}

// MARK: - Decoder Layer

final class AthenaQwen35DecoderLayer: Module {
    let isLinear: Bool

    @ModuleInfo(key: "self_attn") var selfAttn: AthenaQwen35Attention?
    @ModuleInfo(key: "linear_attn") var linearAttn: AthenaQwen35GatedDeltaNet?

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    @ModuleInfo(key: "mlp") var mlp: Module

    init(_ args: AthenaQwen35TextConfiguration, layerIdx: Int) {
        self.isLinear = (layerIdx + 1) % args.fullAttentionInterval != 0

        if isLinear {
            _linearAttn.wrappedValue = AthenaQwen35GatedDeltaNet(args)
        } else {
            _selfAttn.wrappedValue = AthenaQwen35Attention(args)
        }

        if args.numExperts > 0 {
            _mlp.wrappedValue = AthenaQwen35SparseMoeBlock(args)
        } else {
            _mlp.wrappedValue = Qwen3NextMLP(
                dimensions: args.hiddenSize,
                hiddenDimensions: args.intermediateSize
            )
        }

        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize,
            eps: args.rmsNormEps
        )
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize,
            eps: args.rmsNormEps
        )

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        attentionMask: MLXFast.ScaledDotProductAttentionMaskMode,
        ssmMask: MLXArray?,
        cache: KVCache?,
        nConfirmed: Int = 0,
        rollback: GDNRollback? = nil
    ) -> MLXArray {
        let prof = ForwardProfile.enabled
        var t0 = prof ? ForwardProfile.clk() : 0
        let r: MLXArray
        if isLinear {
            r = linearAttn!(
                inputLayerNorm(x), mask: ssmMask,
                cache: cache as? MambaCache, nConfirmed: nConfirmed,
                rollback: rollback)
        } else {
            // Attention runs once over the whole [confirmed|draft] chunk
            // (the speedup); KV reject is a trim(1) in the loop.
            r = selfAttn!(inputLayerNorm(x), mask: attentionMask, cache: cache)
        }
        if prof {
            eval(r)
            let dt = Double(ForwardProfile.clk() - t0) / 1e9
            if isLinear { ForwardProfile.tGDN += dt; ForwardProfile.nGDN += 1 }
            else { ForwardProfile.tAttn += dt; ForwardProfile.nAttn += 1 }
            t0 = ForwardProfile.clk()
        }

        let h = x + r
        let out = h + (mlp as! UnaryLayer)(postAttentionLayerNorm(h))
        if prof { eval(out); ForwardProfile.tMLP += Double(ForwardProfile.clk() - t0) / 1e9 }
        return out
    }
}

// MARK: - Text Model

public class AthenaQwen35TextModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    fileprivate let layers: [AthenaQwen35DecoderLayer]
    let norm: RMSNorm

    let ssmIdx: Int
    let faIdx: Int

    init(_ args: AthenaQwen35TextConfiguration) {
        precondition(args.vocabularySize > 0)

        _embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize,
            dimensions: args.hiddenSize
        )

        self.layers = (0 ..< args.hiddenLayers).map { layerIdx in
            AthenaQwen35DecoderLayer(args, layerIdx: layerIdx)
        }

        self.norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)

        self.ssmIdx = 0
        self.faIdx = args.fullAttentionInterval - 1

        super.init()
    }

    /// The backbone up to (but NOT including) the final norm — the
    /// pre-norm hidden the MTP head consumes. `callAsFunction` applies
    /// `norm` on top, so the non-speculative path is byte-unchanged.
    func backbone(
        _ inputs: MLXArray, cache: [KVCache?]? = nil, nConfirmed: Int = 0,
        rollback: GDNRollback? = nil
    ) -> MLXArray {
        var hiddenStates = embedTokens(inputs)

        var cacheArray = cache
        if cacheArray == nil {
            cacheArray = Array(repeating: nil as KVCache?, count: layers.count)
        }

        let faMask = createAttentionMask(h: hiddenStates, cache: cacheArray?[faIdx])
        let ssmMask = createSSMMask(h: hiddenStates, cache: cacheArray?[ssmIdx] as? MambaCache)

        for (i, layer) in layers.enumerated() {
            let mask = layer.isLinear ? ssmMask : nil
            let attnMask =
                layer.isLinear
                ? MLXFast.ScaledDotProductAttentionMaskMode.none : faMask
            hiddenStates = layer(
                hiddenStates, attentionMask: attnMask, ssmMask: mask,
                cache: cacheArray?[i], nConfirmed: nConfirmed,
                rollback: rollback)
        }

        return hiddenStates
    }

    func normalize(_ x: MLXArray) -> MLXArray { norm(x) }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache?]? = nil) -> MLXArray {
        norm(backbone(inputs, cache: cache))
    }
}

public class AthenaQwen35TextModel: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    public let model: AthenaQwen35TextModelInner
    let configuration: AthenaQwen35TextConfiguration

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    /// Optional MTP speculative-decoding head; present only when the
    /// checkpoint declares `mtp_num_hidden_layers > 0`. Loaded in M2.2a;
    /// wired into generation in M2.2c.
    @ModuleInfo(key: "mtp") var mtp: AthenaQwen35MTPModule?

    /// When non-nil, `newCache` builds self-evicting `TriAttentionKVCache`
    /// for attention layers (the M21 norm-only eviction seam). Set only
    /// by the standard generation path; the MTP/speculative path clears
    /// it so eviction is inert there (it can't un-mix GDN recurrent
    /// state). Not a model weight — plain transient generation state.
    public var triAttentionEviction: TriAttentionConfig?

    public init(_ args: AthenaQwen35TextConfiguration) {
        self.configuration = args
        self.vocabularySize = args.vocabularySize
        self.kvHeads = (0 ..< args.hiddenLayers).map { _ in args.kvHeads }
        self.model = AthenaQwen35TextModelInner(args)

        if !args.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
        }
        if args.mtpNumHiddenLayers > 0 {
            _mtp.wrappedValue = AthenaQwen35MTPModule(args)
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var out = model(inputs, cache: cache)
        if let lmHead {
            out = lmHead(out)
        } else {
            out = model.embedTokens.asLinear(out)
        }
        return out
    }

    // MARK: - MTP (M2.2b) — additive; not wired into generation until M2.2c.

    /// True when this checkpoint carries an MTP head (gate for the
    /// speculative path).
    public var hasMTPHead: Bool { mtp != nil }

    private func project(_ normed: MLXArray) -> MLXArray {
        lmHead.map { $0(normed) } ?? model.embedTokens.asLinear(normed)
    }

    /// One backbone pass returning both the logits AND the pre-norm
    /// hidden the MTP head needs (avoids a second forward).
    public func logitsAndHidden(
        _ inputs: MLXArray, cache: [KVCache]?, nConfirmed: Int = 0,
        rollback: GDNRollback? = nil
    ) -> (logits: MLXArray, hidden: MLXArray) {
        let hidden = model.backbone(
            inputs, cache: cache, nConfirmed: nConfirmed,
            rollback: rollback)
        return (project(model.normalize(hidden)), hidden)
    }

    /// Run the MTP head: predict t+2 from the backbone pre-norm hidden at
    /// t and the just-sampled token t+1. nil when no MTP head.
    public func mtpForward(
        hidden: MLXArray, nextTokenIds: MLXArray, mtpCache: [KVCache?]?
    ) -> MLXArray? {
        guard let mtp else { return nil }
        return project(
            mtp(
                hidden, nextTokenIds: nextTokenIds,
                embedTokens: model.embedTokens, cache: mtpCache))
    }

    /// Dedicated KV cache for the MTP attention sublayers (empty when no
    /// MTP head).
    public func makeMTPCache() -> [KVCache] {
        (mtp?.layers ?? []).map { _ in KVCacheSimple() }
    }

    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        return model.layers.map { layer in
            if layer.isLinear {
                return MambaCache()
            }
            if let evict = triAttentionEviction {
                return TriAttentionKVCache(config: evict)
            }
            return KVCacheSimple()
        }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        // Qwen3.5 uses (1 + weight) RMSNorm. The +1 is applied once, at
        // convert-time, gated ONLY on raw-HF conv1d layout. The
        // GoodOlClint/mlx-lm mtp fork deliberately dropped the
        // `has_mtp_weights` term: an already-converted MLX checkpoint that
        // *contains* mtp.* weights must NOT be shifted again. Stock
        // mlx-swift-lm kept that term, so it double-shifts fork mtp
        // checkpoints to (2 + w) on every layer norm → garbage. We match
        // the fork's convention. Deliberate, non-obvious divergence from
        // the substrate — this is precisely why Athena owns this model.
        let hasUnsanitizedConv1d = weights.contains { key, value in
            key.contains("conv1d.weight") && value.dim(-1) != 1
        }
        let shouldShiftNormWeights = hasUnsanitizedConv1d

        // MTP presence is checkpoint-driven: the registry suppresses the
        // head (config `mtp_num_hidden_layers` is an architectural
        // constant present even in stock checkpoints with no mtp.*
        // weights). So `mtp == nil` reliably means a non-mtp checkpoint —
        // just drop any stray mtp.* in that case (mlx-swift forbids
        // mutating the module tree here, hence the gate lives at
        // construction, not in sanitize).
        var weights = weights
        if mtp == nil {
            weights = weights.filter { !$0.key.contains("mtp.") }
        }

        if configuration.tieWordEmbeddings {
            weights["lm_head.weight"] = nil
        }

        // The fork adds 3 MTP-specific norm suffixes to the +1-shift set
        // (qwen3_5.py:489–492) alongside the generic backbone norms; the
        // MTP attention sublayer norms are covered by the generic
        // .q_norm/.k_norm/.input_layernorm/.post_attention_layernorm.
        let normKeys = [
            ".input_layernorm.weight",
            ".post_attention_layernorm.weight",
            "model.norm.weight",
            ".q_norm.weight",
            ".k_norm.weight",
            ".pre_fc_norm_hidden.weight",
            ".pre_fc_norm_embedding.weight",
            "mtp.norm.weight",
        ]

        for k in Array(weights.keys) {
            guard let v = weights[k] else { continue }
            if k.contains("conv1d.weight") && v.dim(-1) != 1 {
                weights[k] = v.movedAxis(source: 2, destination: 1)
                continue
            }
            if shouldShiftNormWeights
                && normKeys.contains(where: { k.hasSuffix($0) })
                && v.ndim == 1
            {
                weights[k] = v + MLXArray(1, dtype: v.dtype)
            }
        }

        return weights
    }
}

extension AthenaQwen35TextModel: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}

// MARK: - Top-level Model

public class AthenaQwen35Model: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    @ModuleInfo(key: "language_model") var languageModel: AthenaQwen35TextModel

    public init(_ args: AthenaQwen35Configuration) {
        let textModel = AthenaQwen35TextModel(args.textConfig)
        self.vocabularySize = textModel.vocabularySize
        self.kvHeads = textModel.kvHeads
        _languageModel.wrappedValue = textModel
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        languageModel(inputs, cache: cache)
    }

    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        languageModel.newCache(parameters: parameters)
    }

    /// M21 norm-only TriAttention eviction policy (set-through to the
    /// text model). Set by the standard path; cleared by the MTP path.
    public var triAttentionEviction: TriAttentionConfig? {
        get { languageModel.triAttentionEviction }
        set { languageModel.triAttentionEviction = newValue }
    }

    // MARK: - MTP (M2.2b) passthrough — the speculative path uses these.

    public var hasMTPHead: Bool { languageModel.hasMTPHead }

    public func logitsAndHidden(
        _ inputs: MLXArray, cache: [KVCache]?, nConfirmed: Int = 0,
        rollback: GDNRollback? = nil
    ) -> (logits: MLXArray, hidden: MLXArray) {
        languageModel.logitsAndHidden(
            inputs, cache: cache, nConfirmed: nConfirmed,
            rollback: rollback)
    }

    public func mtpForward(
        hidden: MLXArray, nextTokenIds: MLXArray, mtpCache: [KVCache?]?
    ) -> MLXArray? {
        languageModel.mtpForward(
            hidden: hidden, nextTokenIds: nextTokenIds, mtpCache: mtpCache)
    }

    public func makeMTPCache() -> [KVCache] {
        languageModel.makeMTPCache()
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = [String: MLXArray]()
        for (key, value) in weights {
            if key.hasPrefix("vision_tower") || key.hasPrefix("model.visual") {
                continue
            }

            var key = key
            if key.hasPrefix("model.language_model") {
                key = key.replacingOccurrences(
                    of: "model.language_model", with: "language_model.model")
            } else if !key.hasPrefix("language_model.") {
                key = "language_model." + key
            }
            sanitized[key] = value
        }

        return languageModel.sanitize(weights: sanitized)
    }
}

extension AthenaQwen35Model: LoRAModel {
    public var loraLayers: [Module] {
        languageModel.model.layers
    }
}
