// MLX-Swift port of NVIDIA Parakeet-TDT-0.6B-v3
// (`mlx-community/parakeet-tdt-0.6b-v3`). Hardened from the ADR-019
// feasibility spike into the production Parakeet transcription engine
// (ADR 020): governed via `MLXTranscriptionModule`, model-class routed by
// `TranscriptionArch`, reference-exact L1 mel magnitude (R1), and a proper
// special-token-stripping detokenizer (`ParakeetDetokenizer`, R2). Still
// greedy TDT — beam search / streaming are out of scope (ADR 020).
//
// Architecture follows the live Python MLX reference at
// `/tmp/parakeet-ref/parakeet-mlx/parakeet_mlx/` (audio.py / conformer.py /
// attention.py / rnnt.py / parakeet.py) — the source of truth. The
// FastConformer encoder is the same architecture Athena already ships in
// `Sortformer.swift`; this file re-implements it with @ModuleInfo keys that
// match the Parakeet safetensors (encoder.*, decoder.prediction.*, joint.*)
// and bias=false on the encoder.
//
// Conv weights in the v3 safetensors are ALREADY in MLX layout
// (Conv2d [out,kH,kW,in], Conv1d [out,K,in]) — do NOT transpose.
import AthenaCore
import Foundation
import HuggingFace
import MLX
import MLXFFT
import MLXHuggingFace
import MLXLMCommon
import MLXNN

// MARK: - Config (verified v3 spec)

struct ParakeetConfig {
    // preprocess
    let sampleRate = 16000
    let nMels = 128
    let nFFT = 512
    let winLength = 400
    let hopLength = 160
    let preemph: Float = 0.97
    let fmin: Double = 0
    let fmax: Double = 8000

    // encoder
    let featIn = 128
    let nLayers = 24
    let dModel = 1024
    let nHeads = 8
    let dFF = 4096
    let convKernel = 9
    let subsamplingFactor = 8
    let subsamplingConvChannels = 256

    // decoder / joint
    let predHidden = 640
    let predRNNLayers = 2
    let vocabSize = 8192  // 8192 real tokens; blank = id 8192
    let jointHidden = 640
    let numDurations = 5  // [0,1,2,3,4]

    var blankId: Int { vocabSize }  // 8192
    var durations: [Int] { [0, 1, 2, 3, 4] }
    var maxSymbols = 10  // anti-stall

    var timeRatio: Double {
        Double(subsamplingFactor) / Double(sampleRate) * Double(hopLength)
    }  // 0.08 s/frame
}

// MARK: - Mel preprocessing (audio.py get_logmel, exact)

enum ParakeetMel {
    /// librosa.filters.mel(sr, n_fft, n_mels, fmin, fmax, htk=False,
    /// norm="slaney") → row-major `[nMels, nFFT/2+1]`. Ported deterministically
    /// from librosa (slaney mel scale + triangular filters + slaney area norm).
    static func slaneyMelFilterbank(
        sr: Double, nFFT: Int, nMels: Int, fmin: Double, fmax: Double
    ) -> [Float] {
        let nFreqs = nFFT / 2 + 1

        // slaney hz<->mel
        let fSp = 200.0 / 3.0
        let minLogHz = 1000.0
        let minLogMel = (minLogHz - 0.0) / fSp  // 15
        let logStep = log(6.4) / 27.0
        func hzToMel(_ f: Double) -> Double {
            f >= minLogHz ? minLogMel + log(f / minLogHz) / logStep : f / fSp
        }
        func melToHz(_ m: Double) -> Double {
            m >= minLogMel ? minLogHz * exp(logStep * (m - minLogMel)) : fSp * m
        }

        // fft bin center frequencies (linear 0..sr/2)
        let fftFreqs = (0..<nFreqs).map {
            Double($0) * (sr / 2.0) / Double(nFreqs - 1)
        }
        let minMel = hzToMel(fmin), maxMel = hzToMel(fmax)
        let melPts = (0..<(nMels + 2)).map {
            melToHz(minMel + (maxMel - minMel) * Double($0) / Double(nMels + 1))
        }
        let fdiff = (0..<(nMels + 1)).map { melPts[$0 + 1] - melPts[$0] }

        var w = [Float](repeating: 0, count: nMels * nFreqs)
        for i in 0..<nMels {
            let enorm = 2.0 / (melPts[i + 2] - melPts[i])  // slaney area norm
            for j in 0..<nFreqs {
                let lower = -(melPts[i] - fftFreqs[j]) / fdiff[i]
                let upper = (melPts[i + 2] - fftFreqs[j]) / fdiff[i + 1]
                let v = max(0.0, min(lower, upper))
                w[i * nFreqs + j] = Float(v * enorm)
            }
        }
        return w
    }

    /// Mono 16 kHz PCM → normalized log-mel `[1, time, nMels]`.
    /// Mirrors audio.py get_logmel: preemph → reflect-pad n_fft//2 →
    /// periodic-hann STFT (win 400 zero-padded to 512) → |X|^2 → slaney mel →
    /// log(x+1e-5) → per-feature (time-axis) normalize.
    static func logMel(_ samples: [Float], _ cfg: ParakeetConfig) -> MLXArray {
        let n = samples.count
        // preemphasis: x = concat([x[:1], x[1:] - 0.97*x[:-1]])
        var x = [Float](repeating: 0, count: n)
        if n > 0 { x[0] = samples[0] }
        for i in 1..<max(n, 1) { x[i] = samples[i] - cfg.preemph * samples[i - 1] }

        // reflect pad by n_fft//2 = 256 (matches mlx stft _pad reflect:
        // prefix = x[1:p+1][::-1], suffix = x[-(p+1):-1][::-1])
        let p = cfg.nFFT / 2  // 256
        var padded = [Float](repeating: 0, count: n + 2 * p)
        for k in 0..<p { padded[k] = x[p - k] }  // x[p], x[p-1], ... x[1]
        for i in 0..<n { padded[p + i] = x[i] }
        for k in 0..<p { padded[p + n + k] = x[n - 2 - k] }  // x[n-2]..x[n-1-p]

        // periodic hann over win_length=400, then zero-pad to n_fft=512.
        // np.hanning(N+1)[:-1] == 0.5*(1-cos(2π i/N)) for i in 0..N-1
        let wl = cfg.winLength
        var window = [Float](repeating: 0, count: cfg.nFFT)
        for i in 0..<wl {
            window[i] = Float(0.5 * (1.0 - cos(2.0 * .pi * Double(i) / Double(wl))))
        }

        // frame: t = (len - win + hop) // hop, stride hop, width n_fft.
        // mlx as_strided uses width n_fft with the win-length window padded.
        let totalLen = padded.count
        let nFrames = (totalLen - cfg.nFFT + cfg.hopLength) / cfg.hopLength
        var frames = [Float](repeating: 0, count: nFrames * cfg.nFFT)
        for t in 0..<nFrames {
            let base = t * cfg.hopLength
            for k in 0..<cfg.nFFT {
                frames[t * cfg.nFFT + k] = padded[base + k] * window[k]
            }
        }

        let frameArr = MLXArray(frames, [nFrames, cfg.nFFT])
        let spec = MLXFFT.rfft(frameArr, n: cfg.nFFT, axis: -1)  // [nFrames, 257]
        // Mel-exactness (ADR 020 S2 / R1). audio.py computes the magnitude as
        //   abs = mx.abs(mx.view(x, float32)); x = abs[..., ::2] + abs[..., 1::2]
        // i.e. it views the complex spectrum as interleaved (re, im) float32
        // pairs, takes |·| of each, and sums adjacent — which is the **L1**
        // magnitude |re| + |im| per bin, NOT the Euclidean |X| = sqrt(re²+im²).
        // The spike used sqrt; match the reference exactly so the mel (and thus
        // the encoder input) is numerically faithful. mag_power=2 → power.
        let mag = MLX.abs(spec.realPart()) + MLX.abs(spec.imaginaryPart())
        let power = mag.square()

        let nFreqs = cfg.nFFT / 2 + 1
        let fb = slaneyMelFilterbank(
            sr: Double(cfg.sampleRate), nFFT: cfg.nFFT, nMels: cfg.nMels,
            fmin: cfg.fmin, fmax: cfg.fmax)
        let filters = MLXArray(fb, [cfg.nMels, nFreqs])  // [128, 257]

        // mel = filters @ power.T → [nMels, nFrames]
        var mel = MLX.matmul(filters, power.transposed())  // [128, nFrames]
        mel = MLX.log(mel + 1e-5)

        // per_feature normalize over TIME axis (axis=1): (x-mean)/(std+1e-5)
        let mean = mel.mean(axis: 1, keepDims: true)
        let std = MLX.sqrt(mel.variance(axis: 1, keepDims: true))
        mel = (mel - mean) / (std + 1e-5)

        // → [nFrames, nMels] → [1, nFrames, nMels]
        let out = mel.transposed().expandedDimensions(axis: 0)
        out.eval()
        MLX.Memory.clearCache()
        return out
    }
}

// MARK: - Encoder (FastConformer; keys match Parakeet safetensors)

/// dw_striding subsampling factor 8. Conv weights load in MLX layout directly.
/// Keys: pre_encode.conv.{0,2,3,5,6}.{weight,bias}, pre_encode.out.{weight,bias}
private final class ParakeetSubsampling: Module {
    // The Python reference stores these in a list `conv` at positions
    // 0,2,3,5,6 (1,4,7 are ReLU). The loader remaps `conv.0`→`conv0`, etc.,
    // so they bind to these flat fields without MLX inferring an array.
    @ModuleInfo(key: "conv0") var conv0: Conv2d  // 1→256, k3 s2 p1
    @ModuleInfo(key: "conv2") var conv2: Conv2d  // dw 256, k3 s2 p1 groups256
    @ModuleInfo(key: "conv3") var conv3: Conv2d  // pw 256→256 k1
    @ModuleInfo(key: "conv5") var conv5: Conv2d  // dw 256, k3 s2 p1 groups256
    @ModuleInfo(key: "conv6") var conv6: Conv2d  // pw 256→256 k1
    @ModuleInfo(key: "out") var out: Linear  // 4096→1024 (has bias)

    init(_ c: ParakeetConfig) {
        let ch = c.subsamplingConvChannels
        let ks = IntOrPair((3, 3))
        let st = IntOrPair((2, 2))
        let pad = IntOrPair((1, 1))
        self._conv0.wrappedValue = Conv2d(
            inputChannels: 1, outputChannels: ch,
            kernelSize: ks, stride: st, padding: pad)
        self._conv2.wrappedValue = Conv2d(
            inputChannels: ch, outputChannels: ch,
            kernelSize: ks, stride: st, padding: pad, groups: ch)
        self._conv3.wrappedValue = Conv2d(
            inputChannels: ch, outputChannels: ch, kernelSize: IntOrPair((1, 1)))
        self._conv5.wrappedValue = Conv2d(
            inputChannels: ch, outputChannels: ch,
            kernelSize: ks, stride: st, padding: pad, groups: ch)
        self._conv6.wrappedValue = Conv2d(
            inputChannels: ch, outputChannels: ch, kernelSize: IntOrPair((1, 1)))
        // final freq dim after 3 stride-2 stages on featIn=128: 128→64→32→16
        let freq = 16
        self._out.wrappedValue = Linear(ch * freq, c.dModel, bias: true)
    }

    /// - x: `[B, time, featIn]` → returns `[B, time/8, dModel]`
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // conv_forward in the reference: input [B, 1, T, F] (NCHW),
        // transposed to NHWC [B, T, F, 1] for MLX Conv2d.
        var h = x.expandedDimensions(axis: -1)  // [B, T, F, 1]
        h = relu(conv0(h))
        h = relu(conv3(conv2(h)))
        h = relu(conv6(conv5(h)))
        // reference: x.swapaxes(1,2).reshape(B, T', -1) where the NCHW view is
        // [B, C, T', F']; in NHWC h is [B, T', F', C]; flatten (C,F') → C*F'.
        let (b, t, f, ch) = (h.dim(0), h.dim(1), h.dim(2), h.dim(3))
        h = h.transposed(0, 1, 3, 2).reshaped(b, t, ch * f)  // [B, T', C*F']
        return out(h)
    }
}

/// Transformer-XL relative positional encoding (audio.py RelPositionalEncoding,
/// scale_input=false). Returns `[1, 2*time-1, dModel]`.
private final class ParakeetRelPosEnc: Module {
    let dModel: Int
    init(dModel: Int) { self.dModel = dModel }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let t = x.dim(1)
        // positions: arange(t-1, -t, -1) → length 2t-1
        let positions = MLXArray(
            stride(from: t - 1, through: -(t - 1), by: -1).map { Float($0) })
        let dim = MLXArray(stride(from: 0, to: dModel, by: 2).map { Float($0) })
        let divTerm = MLX.exp(dim * Float(-log(10000.0) / Double(dModel)))
        let angles =
            positions.expandedDimensions(axis: 1)
            * divTerm.expandedDimensions(axis: 0)
        let sinA = MLX.sin(angles)
        let cosA = MLX.cos(angles)
        // interleave: pe[:,0::2]=sin, pe[:,1::2]=cos
        let pe = MLX.stacked([sinA, cosA], axis: -1)
            .reshaped(positions.dim(0), dModel)
        return pe.expandedDimensions(axis: 0).asType(x.dtype)
    }
}

/// Rel-pos multi-head attention (attention.py RelPositionMultiHeadAttention).
/// bias=false on q/k/v/out; linear_pos has no bias; pos_bias_u/v learned.
private final class ParakeetRelPosAttention: Module {
    let h: Int
    let dK: Int
    let scale: Float

    @ModuleInfo(key: "linear_q") var linearQ: Linear
    @ModuleInfo(key: "linear_k") var linearK: Linear
    @ModuleInfo(key: "linear_v") var linearV: Linear
    @ModuleInfo(key: "linear_out") var linearOut: Linear
    @ModuleInfo(key: "linear_pos") var linearPos: Linear
    @ParameterInfo(key: "pos_bias_u") var posBiasU: MLXArray
    @ParameterInfo(key: "pos_bias_v") var posBiasV: MLXArray

    init(_ c: ParakeetConfig) {
        h = c.nHeads
        dK = c.dModel / c.nHeads
        scale = 1.0 / sqrt(Float(dK))
        self._linearQ.wrappedValue = Linear(c.dModel, c.dModel, bias: false)
        self._linearK.wrappedValue = Linear(c.dModel, c.dModel, bias: false)
        self._linearV.wrappedValue = Linear(c.dModel, c.dModel, bias: false)
        self._linearOut.wrappedValue = Linear(c.dModel, c.dModel, bias: false)
        self._linearPos.wrappedValue = Linear(c.dModel, c.dModel, bias: false)
        self._posBiasU.wrappedValue = MLXArray.zeros([c.nHeads, dK])
        self._posBiasV.wrappedValue = MLXArray.zeros([c.nHeads, dK])
    }

    /// rel_shift (attention.py): pad-left, reshape, drop first row.
    private func relShift(_ x: MLXArray) -> MLXArray {
        let (b, hc, tq, posLen) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        var padded = MLX.padded(
            x, widths: [.init((0, 0)), .init((0, 0)), .init((0, 0)), .init((1, 0))])
        padded = padded.reshaped(b, hc, posLen + 1, tq)
        padded = padded[0..., 0..., 1..., 0...].reshaped(b, hc, tq, posLen)
        return padded
    }

    func callAsFunction(_ x: MLXArray, posEmb: MLXArray) -> MLXArray {
        let (batch, qSeq) = (x.dim(0), x.dim(1))
        let q = linearQ(x)
        let k = linearK(x)
        let v = linearV(x)
        let p = linearPos(posEmb)

        let qR = q.reshaped(batch, qSeq, h, dK)
        let qU = (qR + posBiasU).transposed(0, 2, 1, 3)  // [B,H,T,dK]
        let qV = (qR + posBiasV).transposed(0, 2, 1, 3)
        let kH = k.reshaped(batch, qSeq, h, dK).transposed(0, 2, 1, 3)
        let vH = v.reshaped(batch, qSeq, h, dK).transposed(0, 2, 1, 3)
        let pH = p.reshaped(1, -1, h, dK).transposed(0, 2, 1, 3)

        let matrixAC = MLX.matmul(qU, kH.transposed(0, 1, 3, 2))  // [B,H,T,T]
        var matrixBD = MLX.matmul(qV, pH.transposed(0, 1, 3, 2))  // [B,H,T,posLen]
        matrixBD = relShift(matrixBD)
        matrixBD = matrixBD[0..., 0..., 0..., ..<matrixAC.dim(3)]

        let scores = (matrixAC + matrixBD) * scale
        let attn = softmax(scores, axis: -1)
        let outH = MLX.matmul(attn, vH)  // [B,H,T,dK]
        let merged = outH.transposed(0, 2, 1, 3).reshaped(batch, qSeq, h * dK)
        return linearOut(merged)
    }
}

/// Conformer feed-forward: Linear→SiLU→Linear, bias=false.
private final class ParakeetFeedForward: Module {
    @ModuleInfo var linear1: Linear
    @ModuleInfo var linear2: Linear
    init(d: Int, dFF: Int) {
        self._linear1.wrappedValue = Linear(d, dFF, bias: false)
        self._linear2.wrappedValue = Linear(dFF, d, bias: false)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { linear2(silu(linear1(x))) }
}

/// BatchNorm1d using stored running stats (eval mode).
private final class ParakeetBatchNorm: Module {
    let eps: Float = 1e-5
    var weight: MLXArray
    var bias: MLXArray
    @ParameterInfo(key: "running_mean") var runningMean: MLXArray
    @ParameterInfo(key: "running_var") var runningVar: MLXArray
    init(_ d: Int) {
        weight = MLXArray.ones([d])
        bias = MLXArray.zeros([d])
        self._runningMean.wrappedValue = MLXArray.zeros([d])
        self._runningVar.wrappedValue = MLXArray.ones([d])
    }
    /// x: `[B, T, C]`
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        (x - runningMean) * rsqrt(runningVar + eps) * weight + bias
    }
}

/// Conformer convolution: pointwise→GLU→depthwise→BN→SiLU→pointwise.
/// All convs bias=false (no .bias keys in v3); BN has weight/bias.
/// Conv1d weights load in MLX layout [out,K,in] — no transpose.
private final class ParakeetConvolution: Module {
    let padding: Int
    @ModuleInfo(key: "pointwise_conv1") var pw1: Conv1d
    @ModuleInfo(key: "depthwise_conv") var dw: Conv1d
    @ModuleInfo(key: "batch_norm") var bn: ParakeetBatchNorm
    @ModuleInfo(key: "pointwise_conv2") var pw2: Conv1d

    init(_ c: ParakeetConfig) {
        let d = c.dModel
        padding = (c.convKernel - 1) / 2
        self._pw1.wrappedValue = Conv1d(
            inputChannels: d, outputChannels: d * 2, kernelSize: 1, bias: false)
        self._dw.wrappedValue = Conv1d(
            inputChannels: d, outputChannels: d, kernelSize: c.convKernel,
            padding: 0, groups: d, bias: false)
        self._bn.wrappedValue = ParakeetBatchNorm(d)
        self._pw2.wrappedValue = Conv1d(
            inputChannels: d, outputChannels: d, kernelSize: 1, bias: false)
    }

    /// x: `[B, T, C]`. GLU on the channel axis (axis=-1, = axis 2).
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = pw1(x)  // [B,T,2C]
        let parts = MLX.split(h, parts: 2, axis: -1)
        h = parts[0] * sigmoid(parts[1])  // nn.glu == a * sigmoid(b)
        // explicit symmetric pad (reference pads, conv has padding=0)
        h = MLX.padded(h, widths: [.init((0, 0)), .init((padding, padding)), .init((0, 0))])
        h = dw(h)
        h = bn(h)
        h = silu(h)
        h = pw2(h)
        return h
    }
}

/// One Conformer block (macaron): 0.5*FF1 → attn → conv → 0.5*FF2 → LN_out.
private final class ParakeetConformerLayer: Module {
    let fc: Float = 0.5
    @ModuleInfo(key: "norm_feed_forward1") var nFF1: LayerNorm
    @ModuleInfo(key: "feed_forward1") var ff1: ParakeetFeedForward
    @ModuleInfo(key: "norm_self_att") var nSA: LayerNorm
    @ModuleInfo(key: "self_attn") var selfAttn: ParakeetRelPosAttention
    @ModuleInfo(key: "norm_conv") var nConv: LayerNorm
    @ModuleInfo(key: "conv") var conv: ParakeetConvolution
    @ModuleInfo(key: "norm_feed_forward2") var nFF2: LayerNorm
    @ModuleInfo(key: "feed_forward2") var ff2: ParakeetFeedForward
    @ModuleInfo(key: "norm_out") var nOut: LayerNorm

    init(_ c: ParakeetConfig) {
        let d = c.dModel
        self._nFF1.wrappedValue = LayerNorm(dimensions: d)
        self._ff1.wrappedValue = ParakeetFeedForward(d: d, dFF: c.dFF)
        self._nSA.wrappedValue = LayerNorm(dimensions: d)
        self._selfAttn.wrappedValue = ParakeetRelPosAttention(c)
        self._nConv.wrappedValue = LayerNorm(dimensions: d)
        self._conv.wrappedValue = ParakeetConvolution(c)
        self._nFF2.wrappedValue = LayerNorm(dimensions: d)
        self._ff2.wrappedValue = ParakeetFeedForward(d: d, dFF: c.dFF)
        self._nOut.wrappedValue = LayerNorm(dimensions: d)
    }

    func callAsFunction(_ x: MLXArray, posEmb: MLXArray) -> MLXArray {
        var r = x + fc * ff1(nFF1(x))
        r = r + selfAttn(nSA(r), posEmb: posEmb)
        r = r + conv(nConv(r))
        r = r + fc * ff2(nFF2(r))
        return nOut(r)
    }
}

/// FastConformer encoder. Key prefix `encoder.` ⇒ this module is mounted under
/// "encoder" in the model. pre_encode + layers.N + pos_enc (no weights).
final class ParakeetEncoder: Module {
    @ModuleInfo(key: "pre_encode") fileprivate var preEncode: ParakeetSubsampling
    fileprivate var layers: [ParakeetConformerLayer]
    private let posEnc: ParakeetRelPosEnc

    init(_ c: ParakeetConfig) {
        self._preEncode.wrappedValue = ParakeetSubsampling(c)
        layers = (0..<c.nLayers).map { _ in ParakeetConformerLayer(c) }
        posEnc = ParakeetRelPosEnc(dModel: c.dModel)
    }

    /// mel `[B, time, featIn]` → features `[B, time/8, dModel]`.
    func callAsFunction(_ mel: MLXArray) -> MLXArray {
        var x = preEncode(mel)  // [B, T', dModel]; xscaling=false ⇒ no scale
        let posEmb = posEnc(x)
        for layer in layers { x = layer(x, posEmb: posEmb) }
        return x
    }
}

// MARK: - Decoder (prediction net: embed + 2-layer LSTM)

/// Single LSTM layer, combined gates Wx/Wh + bias, PyTorch gate order i,f,g,o.
/// Reused pattern from PyanLSTMCell, but stepped one timestep at a time with
/// explicit (h,c) carry for the autoregressive RNNT loop.
private final class ParakeetLSTMCell: Module {
    @ParameterInfo(key: "Wx") var wx: MLXArray  // [4H, in]
    @ParameterInfo(key: "Wh") var wh: MLXArray  // [4H, H]
    var bias: MLXArray  // [4H]

    init(inputSize: Int, hiddenSize: Int) {
        self._wx.wrappedValue = MLXArray.zeros([4 * hiddenSize, inputSize])
        self._wh.wrappedValue = MLXArray.zeros([4 * hiddenSize, hiddenSize])
        self.bias = MLXArray.zeros([4 * hiddenSize])
    }

    /// One step. x: `[B, in]`, h/c: `[B, H]` → (hNew, cNew) each `[B, H]`.
    func step(_ x: MLXArray, h: MLXArray, c: MLXArray) -> (MLXArray, MLXArray) {
        var ifgo = addMM(bias, x, wx.T) + MLX.matmul(h, wh.T)  // [B, 4H]
        let g = split(ifgo, parts: 4, axis: -1)
        let i = sigmoid(g[0])
        let f = sigmoid(g[1])
        let gg = tanh(g[2])
        let o = sigmoid(g[3])
        let cNew = f * c + i * gg
        let hNew = o * tanh(cNew)
        return (hNew, cNew)
    }
}

/// Holds the LSTM layer stack under the `dec_rnn.lstm.N.*` key path.
private final class ParakeetDecRNN: Module {
    @ModuleInfo(key: "lstm") var lstm: [ParakeetLSTMCell]
    init(_ c: ParakeetConfig) {
        self._lstm.wrappedValue = (0..<c.predRNNLayers).map { _ in
            ParakeetLSTMCell(inputSize: c.predHidden, hiddenSize: c.predHidden)
        }
    }
}

/// Prediction network: embed(vocab+1, 640) + 2-layer LSTM.
/// Keys: prediction.embed.weight, prediction.dec_rnn.lstm.{0,1}.{Wx,Wh,bias}
private final class ParakeetPredictionInner: Module {
    @ModuleInfo(key: "embed") var embed: Embedding
    @ModuleInfo(key: "dec_rnn") var decRNN: ParakeetDecRNN
    var lstm: [ParakeetLSTMCell] { decRNN.lstm }

    init(_ c: ParakeetConfig) {
        // blank_as_pad ⇒ embedding rows = vocab_size + 1
        self._embed.wrappedValue = Embedding(
            embeddingCount: c.vocabSize + 1, dimensions: c.predHidden)
        self._decRNN.wrappedValue = ParakeetDecRNN(c)
    }
}

final class ParakeetDecoder: Module {
    @ModuleInfo(key: "prediction") fileprivate var prediction: ParakeetPredictionInner
    let predHidden: Int
    let nLayers: Int

    init(_ c: ParakeetConfig) {
        self._prediction.wrappedValue = ParakeetPredictionInner(c)
        predHidden = c.predHidden
        nLayers = c.predRNNLayers
    }

    /// Per-step decode. `lastToken == nil` ⇒ feed zeros (START).
    /// hidden = (h[L,1,H], c[L,1,H]) per layer; returns decOut `[1,1,H]` and
    /// the updated state.
    func stepOut(
        lastToken: Int?, h: [MLXArray], c: [MLXArray]
    ) -> (MLXArray, [MLXArray], [MLXArray]) {
        var x: MLXArray  // [1, H]
        if let t = lastToken {
            x = prediction.embed(MLXArray([Int32(t)]))  // [1, H]
        } else {
            x = MLXArray.zeros([1, predHidden])
        }
        var newH = [MLXArray]()
        var newC = [MLXArray]()
        var input = x
        for l in 0..<nLayers {
            let (hl, cl) = prediction.lstm[l].step(input, h: h[l], c: c[l])
            newH.append(hl)
            newC.append(cl)
            input = hl
        }
        // decoder output is the last layer's hidden, shaped [1,1,H]
        return (input.expandedDimensions(axis: 1), newH, newC)
    }

    func zeroState() -> ([MLXArray], [MLXArray]) {
        let h = (0..<nLayers).map { _ in MLXArray.zeros([1, predHidden]) }
        let c = (0..<nLayers).map { _ in MLXArray.zeros([1, predHidden]) }
        return (h, c)
    }
}

// MARK: - Joint

/// Joint net: enc(1024→640)+bias, pred(640→640)+bias, ReLU,
/// joint_net.2(640→8198)+bias. Output split: [:8193]=tokens, [8193:]=durations.
final class ParakeetJoint: Module {
    @ModuleInfo(key: "enc") var enc: Linear
    @ModuleInfo(key: "pred") var pred: Linear
    // Python joint_net = [ReLU, Identity, Linear]; only index 2 has weights.
    // The loader remaps `joint_net.2.` → `joint_net2.` so this binds flat.
    @ModuleInfo(key: "joint_net2") var jointNet2: Linear

    init(_ c: ParakeetConfig) {
        self._enc.wrappedValue = Linear(c.dModel, c.jointHidden, bias: true)
        self._pred.wrappedValue = Linear(c.predHidden, c.jointHidden, bias: true)
        let outDim = c.vocabSize + 1 + c.numDurations  // 8193 + 5 = 8198
        self._jointNet2.wrappedValue = Linear(c.jointHidden, outDim, bias: true)
    }

    /// encFrame `[1,1,1024]`, decOut `[1,1,640]` → logits `[8198]` (flat).
    func callAsFunction(_ encFrame: MLXArray, _ decOut: MLXArray) -> MLXArray {
        let e = enc(encFrame)  // [1,1,640]
        let pr = pred(decOut)  // [1,1,640]
        // enc[:,:,None,:] + pred[:,None,:,:] → [1,1,1,640]
        let x = e.expandedDimensions(axis: 2) + pr.expandedDimensions(axis: 1)
        let r = relu(x)
        let logits = jointNet2(r)  // [1,1,1,8198]
        return logits.reshaped(-1)  // [8198]
    }
}

// MARK: - Top-level model

public final class ParakeetTDTModel: Module {
    let cfg = ParakeetConfig()
    @ModuleInfo(key: "encoder") var encoder: ParakeetEncoder
    @ModuleInfo(key: "decoder") var decoder: ParakeetDecoder
    @ModuleInfo(key: "joint") var joint: ParakeetJoint
    let vocabulary: [String]

    init(vocabulary: [String]) {
        let c = ParakeetConfig()
        self.vocabulary = vocabulary
        self._encoder.wrappedValue = ParakeetEncoder(c)
        self._decoder.wrappedValue = ParakeetDecoder(c)
        self._joint.wrappedValue = ParakeetJoint(c)
    }

    public struct Result {
        public var transcript: String
        public var tokenIds: [Int]
        /// TDT-aligned tokens (timing from the durations, specials excluded) —
        /// the source for segment/word timestamps (ADR 020 S3).
        public var tokens: [ParakeetAlignment.Token]
        public var decodeSteps: Int
        public var encoderSeconds: Double
        public var decodeSeconds: Double
    }

    /// Run encoder once, then greedy TDT decode (parakeet.py decode_greedy).
    public func transcribe(_ samples: [Float]) -> Result {
        let mel = ParakeetMel.logMel(samples, cfg)  // [1, T, 128]

        let encStart = CFAbsoluteTimeGetCurrent()
        let features = encoder(mel)  // [1, T/8, 1024]
        features.eval()
        let encSeconds = CFAbsoluteTimeGetCurrent() - encStart

        let T = features.dim(1)
        let durations = cfg.durations
        let blank = cfg.blankId

        let decStart = CFAbsoluteTimeGetCurrent()
        var step = 0
        var newSymbols = 0
        var lastToken: Int? = nil
        var (h, c) = decoder.zeroState()
        var hyp = [Int]()
        var tokens = [ParakeetAlignment.Token]()
        var steps = 0

        while step < T {
            let (decOut, newH, newC) = decoder.stepOut(lastToken: lastToken, h: h, c: c)
            let encFrame = features[0..., step..<(step + 1), 0...]  // [1,1,1024]
            let logits = joint(encFrame, decOut)  // [8198]
            // split: tokens [:8193], durations [8193:]
            let tokenLogits = logits[0..<(blank + 1)]  // 8193
            let durLogits = logits[(blank + 1)...]  // 5
            let predToken = MLX.argMax(tokenLogits).item(Int.self)
            let decision = MLX.argMax(durLogits).item(Int.self)
            // Frame index at emission → time (before `step` advances). The TDT
            // duration is the token's own time span (S3).
            let tokenStart = Double(step) * cfg.timeRatio
            let tokenDur = Double(durations[decision]) * cfg.timeRatio

            if predToken != blank {
                hyp.append(predToken)
                lastToken = predToken
                h = newH
                c = newC
                // Record the aligned token for timestamping — predicted-token
                // softmax probability as confidence; specials/out-of-range are
                // excluded so segment/word text matches the detokenized output.
                if predToken >= 0, predToken < vocabulary.count {
                    let piece = vocabulary[predToken]
                    if !ParakeetDetokenizer.isSpecial(piece) {
                        let conf = softmax(tokenLogits, axis: -1)[predToken]
                            .item(Float.self)
                        tokens.append(
                            ParakeetAlignment.Token(
                                id: predToken,
                                text: piece.replacingOccurrences(
                                    of: "\u{2581}", with: " "),
                                start: tokenStart, duration: tokenDur,
                                confidence: Double(conf)))
                    }
                }
            }

            step += durations[decision]
            newSymbols += 1
            if durations[decision] != 0 {
                newSymbols = 0
            } else if newSymbols >= cfg.maxSymbols {
                step += 1
                newSymbols = 0
            }
            steps += 1
        }
        let decSeconds = CFAbsoluteTimeGetCurrent() - decStart

        let transcript = ParakeetTDTModel.decode(hyp, vocabulary: vocabulary)
        MLX.Memory.clearCache()
        return Result(
            transcript: transcript, tokenIds: hyp, tokens: tokens,
            decodeSteps: steps, encoderSeconds: encSeconds,
            decodeSeconds: decSeconds)
    }

    /// SentencePiece detokenize via the MLX-free `ParakeetDetokenizer`
    /// (special-token stripping + leading-space trim; unit-pinned, ADR 008/009).
    static func decode(_ ids: [Int], vocabulary: [String]) -> String {
        ParakeetDetokenizer.detokenize(ids, vocabulary: vocabulary)
    }
}
