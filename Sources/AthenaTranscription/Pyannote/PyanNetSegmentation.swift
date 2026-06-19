// Athena-native MLX port of the pyannote PyanNet speaker-segmentation model
// (SincNet → 4-layer BiLSTM → 2 linear → 7-class powerset head). Adapted from
// soniqo/speech-swift (Apache-2.0) — names `Pyan`-prefixed to avoid colliding
// with the vendored Sortformer types — and cross-checked against pyannote.audio
// `sincnet.py` / `PyanNet.py` / `powerset.py`. ADR 018.
//
// Weights: aufklarer/Pyannote-Segmentation-MLX (MIT) — `model.safetensors`
// (sinc filters pre-computed into a standard Conv1d), loaded via Athena's
// governed #hubDownloader. Conv weights are stored `[Cout, K, Cin]` = MLX
// layout, so they load without transpose. The forward output `[frames × 7]` is
// reduced to plain Swift and decoded by `PyannoteSegmentationDecode`
// (MLX-free, unit-pinned).
import AthenaCore
import Foundation
import HuggingFace
import MLX
import MLXHuggingFace
import MLXLMCommon
import MLXNN

/// Affine instance-norm over the time axis (channels-last `[B, L, C]`), no
/// running stats — matches PyTorch `InstanceNorm1d(affine=True)` (eps 1e-5).
final class PyanInstanceNorm: Module {
    var weight: MLXArray
    var bias: MLXArray

    init(_ dimensions: Int) {
        self.weight = MLXArray.ones([dimensions])
        self.bias = MLXArray.zeros([dimensions])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let mean = x.mean(axis: 1, keepDims: true)
        let variance = x.variance(axis: 1, keepDims: true)
        return (x - mean) * rsqrt(variance + 1e-5) * weight + bias
    }
}

/// LeakyReLU (slope 0.01 — PyTorch default, as pyannote uses).
private func pyanLeakyRelu(_ x: MLXArray) -> MLXArray {
    maximum(x, x * 0.01)
}

/// MaxPool1d over the time axis of channels-last `[B, L, C]` (kernel = stride,
/// padding 0). Slice-and-max — proven against the reference.
private func pyanMaxPool1d(_ x: MLXArray, kernel: Int) -> MLXArray {
    let length = x.dim(-2)
    let outLen = (length - kernel) / kernel + 1
    var slices: [MLXArray] = []
    for k in 0..<kernel {
        slices.append(
            x[0..., .stride(from: k, to: k + outLen * kernel, by: kernel), 0...])
    }
    return MLX.stacked(slices, axis: -1).max(axis: -1)
}

/// One LSTM layer: combined gate weights `Wx [4H, in]`, `Wh [4H, H]`, fused
/// `bias [4H]`; PyTorch gate order i,f,g,o.
final class PyanLSTMCell: Module {
    @ParameterInfo(key: "Wx") var wx: MLXArray
    @ParameterInfo(key: "Wh") var wh: MLXArray
    var bias: MLXArray

    init(inputSize: Int, hiddenSize: Int) {
        self._wx.wrappedValue = MLXArray.zeros([4 * hiddenSize, inputSize])
        self._wh.wrappedValue = MLXArray.zeros([4 * hiddenSize, hiddenSize])
        self.bias = MLXArray.zeros([4 * hiddenSize])
    }

    /// - Parameter x: `[B, L, in]` → `[B, L, H]`.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let projected = addMM(bias, x, wx.T)  // [B, L, 4H]
        let seqLen = x.dim(-2)
        var h: MLXArray?
        var c: MLXArray?
        var outs: [MLXArray] = []
        for t in 0..<seqLen {
            var ifgo = projected[.ellipsis, t, 0...]
            if let h { ifgo = ifgo + matmul(h, wh.T) }
            let g = split(ifgo, parts: 4, axis: -1)
            let i = sigmoid(g[0])
            let f = sigmoid(g[1])
            let gg = tanh(g[2])
            let o = sigmoid(g[3])
            let cNew = c.map { f * $0 + i * gg } ?? (i * gg)
            c = cNew
            h = o * tanh(cNew)
            outs.append(h!)
        }
        return stacked(outs, axis: -2)
    }
}

/// A stack of same-direction LSTM layers (`layers.0`, `layers.1`, …).
final class PyanLSTMStack: Module {
    let layers: [PyanLSTMCell]
    init(_ layers: [PyanLSTMCell]) { self.layers = layers }
}

/// Stacked bidirectional LSTM: per layer run forward and (reversed) backward,
/// concat `[fwd, bwd]` along features → the next layer's input.
private func runPyanBiLSTM(
    _ x: MLXArray, fwd: PyanLSTMStack, bwd: PyanLSTMStack
) -> MLXArray {
    var input = x
    for i in 0..<fwd.layers.count {
        let fwdOut = fwd.layers[i](input)
        let seqLen = input.dim(-2)
        let idx = MLXArray(Array((0..<seqLen).reversed()))
        let rev = input.take(idx, axis: -2)
        let bwdOut = bwd.layers[i](rev).take(idx, axis: -2)
        input = concatenated([fwdOut, bwdOut], axis: -1)
    }
    return input
}

/// SincNet frontend: 3× (conv → [abs on layer 0] → maxpool(3) → instance-norm
/// → leaky-relu) over a wav-normalized waveform.
final class PyanSincNet: Module {
    @ModuleInfo(key: "wav_norm") var wavNorm: PyanInstanceNorm
    let conv: [Conv1d]
    let norm: [PyanInstanceNorm]

    override init() {
        let filters = [80, 60, 60]
        let kernels = [251, 5, 5]
        let strides = [10, 1, 1]
        let inCh = [1, 80, 60]
        // The first (sinc) conv carries no bias in the checkpoint.
        self.conv = (0..<3).map { i in
            Conv1d(
                inputChannels: inCh[i], outputChannels: filters[i],
                kernelSize: kernels[i], stride: strides[i], bias: i != 0)
        }
        self.norm = filters.map { PyanInstanceNorm($0) }
        self._wavNorm.wrappedValue = PyanInstanceNorm(1)
    }

    /// - Parameter x: `[B, 1, samples]` → `[B, frames, 60]`.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = x.transposed(0, 2, 1)  // [B, samples, 1]
        out = wavNorm(out)
        for i in 0..<conv.count {
            out = conv[i](out)
            if i == 0 { out = abs(out) }
            out = pyanMaxPool1d(out, kernel: 3)
            out = norm[i](out)
            out = pyanLeakyRelu(out)
        }
        return out  // [B, frames, 60]
    }
}

/// Full PyanNet: SincNet → BiLSTM(4) → linear(2) → classifier(7) → softmax.
final class PyanNetSegmentationNetwork: Module {
    let sincnet: PyanSincNet
    @ModuleInfo(key: "lstm_fwd") var lstmFwd: PyanLSTMStack
    @ModuleInfo(key: "lstm_bwd") var lstmBwd: PyanLSTMStack
    let linear: [Linear]
    let classifier: Linear

    override init() {
        self.sincnet = PyanSincNet()
        let hidden = 128
        let fwd = (0..<4).map { PyanLSTMCell(inputSize: $0 == 0 ? 60 : 256, hiddenSize: hidden) }
        let bwd = (0..<4).map { PyanLSTMCell(inputSize: $0 == 0 ? 60 : 256, hiddenSize: hidden) }
        self.linear = (0..<2).map { Linear($0 == 0 ? 256 : 128, 128) }
        self.classifier = Linear(128, 7)
        super.init()
        self._lstmFwd.wrappedValue = PyanLSTMStack(fwd)
        self._lstmBwd.wrappedValue = PyanLSTMStack(bwd)
    }

    /// - Parameter waveform: `[B, 1, samples]` (16 kHz mono).
    /// - Returns: `[B, frames, 7]` softmax powerset posteriors.
    func callAsFunction(_ waveform: MLXArray) -> MLXArray {
        var x = sincnet(waveform)  // [B, frames, 60]
        x = runPyanBiLSTM(x, fwd: lstmFwd, bwd: lstmBwd)  // [B, frames, 256]
        for layer in linear { x = pyanLeakyRelu(layer(x)) }
        return softmax(classifier(x), axis: -1)  // [B, frames, 7]
    }
}

/// Loader + sliding-window segmentation runner. Not an actor — the governed
/// `MLXDiarizationModule` owns the single instance and serialises access.
final class PyanNetSegmentationModel {
    let network: PyanNetSegmentationNetwork
    /// 10 s analysis window at 16 kHz.
    static let windowSamples = 160_000
    static let sampleRate = 16_000
    static let windowSeconds = 10.0

    private init(network: PyanNetSegmentationNetwork) {
        self.network = network
    }

    /// Download (`*.json` + `*.safetensors`) and load via Athena's governed,
    /// proxied #hubDownloader (HF cache root follows `HF_HOME`).
    static func fromPretrained(_ repoId: String) async throws -> PyanNetSegmentationModel {
        let modelURL = try await #hubDownloader(
            HuggingFace.HubClient(session: AthenaProxy.proxiedURLSession())
        ).download(
            id: repoId, revision: nil,
            matching: ["*.json", "*.safetensors"],
            useLatest: false, progressHandler: { _ in })
        return try fromModelDirectory(modelURL)
    }

    static func fromModelDirectory(_ url: URL) throws -> PyanNetSegmentationModel {
        let st = try FileManager.default.contentsOfDirectory(
            at: url, includingPropertiesForKeys: nil
        ).first { $0.pathExtension == "safetensors" }
        guard let st else {
            throw AthenaError.moduleLoadFailed(
                .diarization,
                reason: "no .safetensors weights in \(url.path)")
        }
        let weights = try loadArrays(url: st)
        let network = PyanNetSegmentationNetwork()
        try network.update(
            parameters: ModuleParameters.unflattened(weights),
            verify: .noUnusedKeys)
        eval(network.parameters())
        return PyanNetSegmentationModel(network: network)
    }

    /// Diarization-mode sliding window (50% overlap), powerset-decoded into
    /// per-window locally-tagged speaker regions. Global identity is resolved
    /// downstream (embed + cluster). `step` defaults to half the window.
    /// Window start sample offsets for the sliding segmentation. A clip at most
    /// one window long yields a single window at 0 (zero-padded up to `W` by
    /// `paddedWindow`); longer clips tile by `stepSamples`, right-aligning a
    /// final window to the end so the tail is covered exactly once. MLX-free,
    /// unit-pinned.
    static func windowStarts(
        total: Int, windowSamples W: Int, stepSamples: Int
    ) -> [Int] {
        guard total > 0 else { return [] }
        let step = max(1, stepSamples)
        if total <= W { return [0] }
        var starts: [Int] = []
        var s = 0
        while s + W <= total {
            starts.append(s)
            s += step
        }
        if let last = starts.last, last + W < total {
            starts.append(total - W)
        }
        return starts
    }

    /// Exactly `W` samples starting at `s`, zero-padded when the clip ends
    /// first. pyannote's SincNet has a FIXED receptive field (the first Conv1d
    /// has kernel 251); feeding it a shorter raw window makes the convolution
    /// degenerate (`(L-251)/10+1 ≤ 0`), MLX raises an internal error, and its
    /// default error handler ABORTS the whole process (EXC_BREAKPOINT) — so a
    /// short, silent, or corrupt file in a batch would crash the daemon, which
    /// launchd then restarts "fresh". Padding to the window matches the
    /// reference (pyannote zero-pads short chunks) and keeps the conv valid.
    /// MLX-free, unit-pinned.
    static func paddedWindow(
        _ samples: [Float], start s: Int, windowSamples W: Int
    ) -> [Float] {
        let lo = max(0, s)
        let hi = min(lo + W, samples.count)
        var win = lo < hi ? Array(samples[lo..<hi]) : []
        if win.count < W {
            win.append(contentsOf: repeatElement(0, count: W - win.count))
        }
        return win
    }

    func segment(
        _ samples: [Float],
        params: PyannoteSegmentationParams = .default,
        stepSamples: Int = PyanNetSegmentationModel.windowSamples / 2
    ) -> [SpeakerActivityRegion] {
        let total = samples.count
        guard total > 0 else { return [] }
        let W = Self.windowSamples
        let step = max(1, stepSamples)
        let totalSeconds = Double(total) / Double(Self.sampleRate)

        let starts = Self.windowStarts(
            total: total, windowSamples: W, stepSamples: step)

        var regions: [SpeakerActivityRegion] = []
        for (idx, s) in starts.enumerated() {
            // Always exactly `W` samples (zero-padded) so the SincNet conv can
            // never go degenerate and abort the process (see `paddedWindow`).
            let win = Self.paddedWindow(samples, start: s, windowSamples: W)
            let arr = MLXArray(win).reshaped(1, 1, W)
            let probs = network(arr)
            eval(probs)
            let frames = probs.dim(1)
            let flat = probs[0].asArray(Float.self)  // frames*7 row-major
            MLX.Memory.clearCache()
            guard frames > 0 else { continue }
            var posteriors = [[Float]]()
            posteriors.reserveCapacity(frames)
            for f in 0..<frames {
                posteriors.append(Array(flat[(f * 7)..<(f * 7 + 7)]))
            }
            let windowStart = Double(s) / Double(Self.sampleRate)
            let frameDuration = Self.windowSeconds / Double(frames)
            let own = PyannoteSegmentationDecode.ownership(
                index: idx, count: starts.count,
                step: Double(step) / Double(Self.sampleRate),
                windowSeconds: Self.windowSeconds,
                windowStart: windowStart, totalSeconds: totalSeconds)
            regions.append(
                contentsOf: PyannoteSegmentationDecode.regions(
                    posteriors: posteriors, frameDuration: frameDuration,
                    windowStart: windowStart, ownStart: own.start,
                    ownEnd: own.end, window: idx, params: params))
        }
        regions.sort { $0.start < $1.start }
        return regions
    }
}
