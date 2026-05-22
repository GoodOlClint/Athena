// Vendored from soniqo/speech-swift (Apache-2.0), adapted for Athena:
// kept in-module under AthenaTranscription, names prefixed to avoid
// collisions with the vendored Sortformer types. Original architecture:
// WeSpeaker ResNet34-LM (https://github.com/wenet-e2e/wespeaker),
// converted from pyannote/wespeaker-voxceleb-resnet34-LM. M25.1.
//
// BatchNorm is fused into the preceding Conv2d (each conv carries a
// bias and no separate BN layer); the Athena weight loader folds the
// BN-separate mlx-community `weights.npz` into this shape at load time
// (see WeSpeakerModel.swift).
import MLX
import MLXNN

/// ResNet BasicBlock with BatchNorm fused into each Conv2d.
///
/// Two 3×3 convs (each with bias = folded BN); a 1×1 shortcut conv is
/// present only when the stride or channel count changes.
final class WeSpeakerBasicBlock: Module {
    let conv1: Conv2d
    let conv2: Conv2d
    let shortcut: Conv2d?

    init(inChannels: Int, outChannels: Int, stride: Int = 1) {
        self.conv1 = Conv2d(
            inputChannels: inChannels, outputChannels: outChannels,
            kernelSize: 3, stride: IntOrPair((stride, stride)),
            padding: 1, bias: true)
        self.conv2 = Conv2d(
            inputChannels: outChannels, outputChannels: outChannels,
            kernelSize: 3, stride: 1, padding: 1, bias: true)
        if stride != 1 || inChannels != outChannels {
            self.shortcut = Conv2d(
                inputChannels: inChannels, outputChannels: outChannels,
                kernelSize: 1, stride: IntOrPair((stride, stride)),
                padding: 0, bias: true)
        } else {
            self.shortcut = nil
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = relu(conv1(x))
        out = conv2(out)
        let residual = shortcut.map { $0(x) } ?? x
        return relu(out + residual)
    }
}

/// WeSpeaker ResNet34-LM speaker-embedding network (BN-fused).
///
/// ```
/// Input: [B, T, 80, 1] log-mel
/// → Conv2d(1→32, k=3, p=1) + ReLU
/// → layer1: 3× BasicBlock(32→32)
/// → layer2: 4× BasicBlock(32→64, s=2)
/// → layer3: 6× BasicBlock(64→128, s=2)
/// → layer4: 3× BasicBlock(128→256, s=2)
/// → temporal statistics pooling (mean ⊕ std) → [B, 5120]
/// → Linear(5120→256) → L2-normalize
/// Output: [B, 256]
/// ```
final class WeSpeakerNetwork: Module {
    let conv1: Conv2d
    let layer1: [WeSpeakerBasicBlock]
    let layer2: [WeSpeakerBasicBlock]
    let layer3: [WeSpeakerBasicBlock]
    let layer4: [WeSpeakerBasicBlock]
    let embedding: Linear

    override init() {
        self.conv1 = Conv2d(
            inputChannels: 1, outputChannels: 32,
            kernelSize: 3, stride: 1, padding: 1, bias: true)
        self.layer1 = Self.makeLayer(
            inChannels: 32, outChannels: 32, blocks: 3, stride: 1)
        self.layer2 = Self.makeLayer(
            inChannels: 32, outChannels: 64, blocks: 4, stride: 2)
        self.layer3 = Self.makeLayer(
            inChannels: 64, outChannels: 128, blocks: 6, stride: 2)
        self.layer4 = Self.makeLayer(
            inChannels: 128, outChannels: 256, blocks: 3, stride: 2)
        // Pooling: time-pooled mean ⊕ std over the 10 surviving freq
        // bins × 256 channels = 2 × 10 × 256 = 5120.
        self.embedding = Linear(5120, 256)
    }

    private static func makeLayer(
        inChannels: Int, outChannels: Int, blocks: Int, stride: Int
    ) -> [WeSpeakerBasicBlock] {
        var out: [WeSpeakerBasicBlock] = []
        for i in 0..<blocks {
            out.append(
                WeSpeakerBasicBlock(
                    inChannels: i == 0 ? inChannels : outChannels,
                    outChannels: outChannels,
                    stride: i == 0 ? stride : 1))
        }
        return out
    }

    /// - Parameter mel: `[B, T, 80, 1]` log-mel (channels-last).
    /// - Returns: `[B, 256]` L2-normalized speaker embeddings.
    func callAsFunction(_ mel: MLXArray) -> MLXArray {
        // PyTorch WeSpeaker permutes (B,T,F)→(B,F,T); in MLX NHWC that
        // is [B,80,T,1] = [B, freq, time, channels].
        var x = mel.transposed(0, 2, 1, 3)
        x = relu(conv1(x))
        for b in layer1 { x = b(x) }
        for b in layer2 { x = b(x) }
        for b in layer3 { x = b(x) }
        for b in layer4 { x = b(x) }
        // x: [B, F'=10, T'=T/8, 256]. Fold freq into channels in
        // (channel-major) order to match the PyTorch reshape, then pool
        // mean ⊕ std over time.
        let bDim = x.dim(0)
        let tDim = x.dim(2)
        x = x.transposed(0, 2, 3, 1).reshaped(bDim, tDim, -1)  // [B, T', 2560]
        let mean = x.mean(axis: 1)
        let std = sqrt(x.variance(axis: 1) + 1e-10)
        let pooled = concatenated([mean, std], axis: -1)  // [B, 5120]
        var emb = embedding(pooled)
        let norm = sqrt((emb * emb).sum(axis: -1, keepDims: true) + 1e-10)
        emb = emb / norm
        return emb
    }
}
