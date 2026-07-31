import Foundation
import MLX
import MLXNN

/// Whisper architecture ported to mlx-swift (M4.2b). Faithful to
/// openai-whisper / mlx-examples: pre-norm residual blocks, q/v/out
/// biased + k unbiased attention, runtime sinusoidal encoder positions,
/// learned decoder positions, tied output projection. Module property
/// keys match the `mlx-community/whisper-*` checkpoint exactly so the
/// safetensors load is a straight key map (see `WhisperLoader`).
public struct WhisperConfig: Decodable, Sendable {
    public let n_mels: Int
    public let n_audio_ctx: Int
    public let n_audio_state: Int
    public let n_audio_head: Int
    public let n_audio_layer: Int
    public let n_vocab: Int
    public let n_text_ctx: Int
    public let n_text_state: Int
    public let n_text_head: Int
    public let n_text_layer: Int
}

/// `[length, channels]` sinusoidal positions (openai-whisper `sinusoids`).
func whisperSinusoids(length: Int, channels: Int) -> MLXArray {
    let half = channels / 2
    let logIncr = log(10_000.0) / Double(half - 1)
    let inv = MLX.exp(MLXArray(0 ..< half).asType(.float32) * Float(-logIncr))
    let scaled =
        MLXArray(0 ..< length).asType(.float32).reshaped([length, 1])
        * inv.reshaped([1, half])
    return MLX.concatenated([MLX.sin(scaled), MLX.cos(scaled)], axis: 1)
}

/// Per-decode KV cache (side-channel, like the speculative GDNRollback
/// pattern — keeps the @ModuleInfo modules pure). Self slots grow with
/// generated tokens; cross slots are computed once from the fixed
/// encoder output and reused. M4.2e-3.
public final class WhisperKVCache {
    final class Slot { var k: MLXArray?; var v: MLXArray? }
    let selfSlots: [Slot]
    let crossSlots: [Slot]
    init(layers: Int) {
        selfSlots = (0 ..< layers).map { _ in Slot() }
        crossSlots = (0 ..< layers).map { _ in Slot() }
    }
    /// Force the accumulated K/V to concrete arrays so the lazy graph
    /// stays bounded across the decode loop.
    func evalStep() {
        for s in selfSlots + crossSlots {
            if let k = s.k { k.eval() }
            if let v = s.v { v.eval() }
        }
    }
}

/// Side-channel collector for the decoder's per-layer cross-attention
/// scores (pre-softmax `qk`, shape `[1, H, Tq, Tk]`). Used by the
/// word-timestamp alignment pass (M26.2). Capture is opt-in: when no
/// collector is threaded through `logits`, the forward path is byte-for-
/// byte the existing decode (no extra work, no graph nodes retained).
public final class WhisperCrossQK {
    public var perLayer: [MLXArray?]
    public init(layers: Int) {
        perLayer = .init(repeating: nil, count: layers)
    }
    public func evalAll() { for a in perLayer where a != nil { a!.eval() } }
}

final class WhisperAttention: Module {
    let nHead: Int
    @ModuleInfo(key: "query") var query: Linear
    @ModuleInfo(key: "key") var key: Linear
    @ModuleInfo(key: "value") var value: Linear
    @ModuleInfo(key: "out") var out: Linear

    init(_ nState: Int, _ nHead: Int) {
        self.nHead = nHead
        self._query.wrappedValue = Linear(nState, nState, bias: true)
        self._key.wrappedValue = Linear(nState, nState, bias: false)
        self._value.wrappedValue = Linear(nState, nState, bias: true)
        self._out.wrappedValue = Linear(nState, nState, bias: true)
    }

    /// `x` [B,Tq,D]; `xa` (cross k/v source) [B,Tk,D] or nil (self);
    /// `mask` [Tq,Tk] additive or nil. With `slot`: self-attn appends
    /// the new K/V to the cache and attends the full history; cross-attn
    /// computes K/V once (encoder output is fixed) and reuses them —
    /// mathematically identical to the uncached path.
    func callAsFunction(
        _ x: MLXArray, xa: MLXArray? = nil, mask: MLXArray? = nil,
        slot: WhisperKVCache.Slot? = nil, isCross: Bool = false,
        crossQK: WhisperCrossQK? = nil, layer: Int = 0
    ) -> MLXArray {
        let q = query(x)
        let k: MLXArray
        let v: MLXArray
        if isCross {
            if let slot, let ck = slot.k, let cv = slot.v {
                k = ck
                v = cv
            } else {
                let src = xa ?? x
                k = key(src)
                v = value(src)
                slot?.k = k
                slot?.v = v
            }
        } else {
            let nk = key(x)
            let nv = value(x)
            if let slot, let pk = slot.k, let pv = slot.v {
                k = MLX.concatenated([pk, nk], axis: 1)
                v = MLX.concatenated([pv, nv], axis: 1)
            } else {
                k = nk
                v = nv
            }
            if let slot {
                slot.k = k
                slot.v = v
            }
        }
        let (B, Tq) = (x.dim(0), x.dim(1))
        let D = q.dim(2), Dh = D / nHead
        let Tk = k.dim(1)
        let scale = pow(Double(Dh), -0.25)

        func heads(_ a: MLXArray, _ T: Int) -> MLXArray {
            a.reshaped([B, T, nHead, Dh]).transposed(0, 2, 1, 3)
        }
        let qh = heads(q, Tq) * Float(scale)
        let kh = heads(k, Tk) * Float(scale)
        let vh = heads(v, Tk)
        var qk = MLX.matmul(qh, kh.transposed(0, 1, 3, 2))  // [B,H,Tq,Tk]
        if let mask { qk = qk + mask }
        // Capture the pre-softmax cross-attention scores for word-time
        // alignment (M26.2). Cross only, opt-in via `crossQK`.
        if isCross, let crossQK { crossQK.perLayer[layer] = qk }
        let w = MLX.softmax(qk, axis: -1)
        let o = MLX.matmul(w, vh)  // [B,H,Tq,Dh]
            .transposed(0, 2, 1, 3).reshaped([B, Tq, D])
        return out(o)
    }
}

final class WhisperBlock: Module {
    @ModuleInfo(key: "attn") var attn: WhisperAttention
    @ModuleInfo(key: "attn_ln") var attnLN: LayerNorm
    @ModuleInfo(key: "cross_attn") var crossAttn: WhisperAttention?
    @ModuleInfo(key: "cross_attn_ln") var crossAttnLN: LayerNorm?
    @ModuleInfo(key: "mlp1") var mlp1: Linear
    @ModuleInfo(key: "mlp2") var mlp2: Linear
    @ModuleInfo(key: "mlp_ln") var mlpLN: LayerNorm

    init(_ nState: Int, _ nHead: Int, cross: Bool) {
        self._attn.wrappedValue = WhisperAttention(nState, nHead)
        self._attnLN.wrappedValue = LayerNorm(dimensions: nState)
        if cross {
            self._crossAttn.wrappedValue = WhisperAttention(nState, nHead)
            self._crossAttnLN.wrappedValue = LayerNorm(dimensions: nState)
        }
        self._mlp1.wrappedValue = Linear(nState, 4 * nState, bias: true)
        self._mlp2.wrappedValue = Linear(4 * nState, nState, bias: true)
        self._mlpLN.wrappedValue = LayerNorm(dimensions: nState)
    }

    func callAsFunction(
        _ x: MLXArray, xa: MLXArray? = nil, mask: MLXArray? = nil,
        cache: WhisperKVCache? = nil, layer: Int = 0,
        crossQK: WhisperCrossQK? = nil
    ) -> MLXArray {
        var x =
            x
            + attn(
                attnLN(x), mask: mask,
                slot: cache?.selfSlots[layer], isCross: false)
        if let crossAttn, let crossAttnLN, let xa {
            x =
                x
                + crossAttn(
                    crossAttnLN(x), xa: xa,
                    slot: cache?.crossSlots[layer], isCross: true,
                    crossQK: crossQK, layer: layer)
        }
        x = x + mlp2(gelu(mlp1(mlpLN(x))))
        return x
    }
}

final class WhisperAudioEncoder: Module {
    @ModuleInfo(key: "conv1") var conv1: Conv1d
    @ModuleInfo(key: "conv2") var conv2: Conv1d
    @ModuleInfo(key: "blocks") var blocks: [WhisperBlock]
    @ModuleInfo(key: "ln_post") var lnPost: LayerNorm
    let positions: MLXArray

    init(_ c: WhisperConfig) {
        self._conv1.wrappedValue = Conv1d(
            inputChannels: c.n_mels, outputChannels: c.n_audio_state,
            kernelSize: 3, padding: 1)
        self._conv2.wrappedValue = Conv1d(
            inputChannels: c.n_audio_state,
            outputChannels: c.n_audio_state,
            kernelSize: 3, stride: 2, padding: 1)
        self._blocks.wrappedValue = (0 ..< c.n_audio_layer).map { _ in
            WhisperBlock(c.n_audio_state, c.n_audio_head, cross: false)
        }
        self._lnPost.wrappedValue = LayerNorm(dimensions: c.n_audio_state)
        self.positions = whisperSinusoids(
            length: c.n_audio_ctx, channels: c.n_audio_state)
    }

    /// `mel` [n_mels, n_frames] → audio features [1, n_audio_ctx, D].
    func callAsFunction(_ mel: MLXArray) -> MLXArray {
        // MLX Conv1d is channels-last: [B, L, C_in].
        let m = mel.reshaped([1, mel.dim(0), mel.dim(1)])
            .transposed(0, 2, 1)
        var x = gelu(conv1(m))
        x = gelu(conv2(x))
        x = x + positions
        for b in blocks { x = b(x) }
        return lnPost(x)
    }
}

final class WhisperTextDecoder: Module {
    @ModuleInfo(key: "token_embedding") var tokenEmbedding: Embedding
    @ParameterInfo(key: "positional_embedding") var positionalEmbedding: MLXArray
    @ModuleInfo(key: "blocks") var blocks: [WhisperBlock]
    @ModuleInfo(key: "ln") var ln: LayerNorm

    init(_ c: WhisperConfig) {
        self._tokenEmbedding.wrappedValue = Embedding(
            embeddingCount: c.n_vocab, dimensions: c.n_text_state)
        self._positionalEmbedding.wrappedValue = MLXArray.zeros([
            c.n_text_ctx, c.n_text_state,
        ])
        self._blocks.wrappedValue = (0 ..< c.n_text_layer).map { _ in
            WhisperBlock(c.n_text_state, c.n_text_head, cross: true)
        }
        self._ln.wrappedValue = LayerNorm(dimensions: c.n_text_state)
    }

    /// `tokens` [B,T], `audio` [B,n_audio_ctx,D], `offset` = position of
    /// token 0 (KV-cache decoding lands in M4.2c). → logits [B,T,vocab].
    func callAsFunction(
        _ tokens: MLXArray, audio: MLXArray, offset: Int = 0,
        cache: WhisperKVCache? = nil, crossQK: WhisperCrossQK? = nil
    ) -> MLXArray {
        let T = tokens.dim(1)
        var x =
            tokenEmbedding(tokens)
            + positionalEmbedding[offset ..< (offset + T)]
        // Causal mask only when >1 query token (prefix / uncached
        // full-sequence pass). A single cached step attends the whole
        // cached history, which is exactly the causal set — no mask.
        let mask: MLXArray?
        if T > 1 {
            let r = MLXArray(0 ..< T).reshaped([T, 1])
            let c = MLXArray(0 ..< T).reshaped([1, T])
            mask = (c .> r).asType(.float32) * Float(-1e9)
        } else {
            mask = nil
        }
        for (i, b) in blocks.enumerated() {
            x = b(
                x, xa: audio, mask: mask, cache: cache, layer: i,
                crossQK: crossQK)
        }
        x = ln(x)
        return MLX.matmul(x, tokenEmbedding.weight.transposed(1, 0))
    }
}

public final class WhisperModel: Module {
    @ModuleInfo(key: "encoder") var encoder: WhisperAudioEncoder
    @ModuleInfo(key: "decoder") var decoder: WhisperTextDecoder
    public let config: WhisperConfig
    /// `(decoder layer, head)` pairs the checkpoint flags as good for
    /// word-time alignment (the dropped `alignment_heads` tensor, now
    /// loaded — M26.2). Empty ⇒ word alignment falls back to all heads.
    public var alignmentHeads: [(layer: Int, head: Int)] = []

    public init(_ config: WhisperConfig) {
        self.config = config
        self._encoder.wrappedValue = WhisperAudioEncoder(config)
        self._decoder.wrappedValue = WhisperTextDecoder(config)
    }

    /// `mel` [n_mels, n_frames] → audio features [1, n_audio_ctx, D].
    public func embedAudio(_ mel: MLXArray) -> MLXArray { encoder(mel) }

    /// `tokens` [B,T] + audio features → logits [B,T,vocab]. `cache`
    /// (+`offset`) enables incremental KV-cached decoding; nil keeps the
    /// full-sequence path (unchanged for existing callers/tests).
    /// `crossQK` (opt-in) captures per-layer cross-attention scores for
    /// the word-time alignment pass; nil leaves the decode path intact.
    public func logits(
        _ tokens: MLXArray, audio: MLXArray, offset: Int = 0,
        cache: WhisperKVCache? = nil, crossQK: WhisperCrossQK? = nil
    ) -> MLXArray {
        decoder(
            tokens, audio: audio, offset: offset, cache: cache,
            crossQK: crossQK)
    }
}
