// Athena-native loader + inference wrapper for the WeSpeaker ResNet34-LM
// speaker-embedding model. Sources the ungated safetensors weights
// (BatchNorm already fused into each Conv2d) via Athena's governed
// #hubDownloader and runs the vendored WeSpeakerNetwork.
//
// NOTE on the weight repo: the substrate's MLX `loadArrays` reads
// safetensors/gguf only — not the `.npz` that
// `mlx-community/wespeaker-voxceleb-resnet34-LM` ships — so Athena
// sources the identical model (pyannote/wespeaker-voxceleb-resnet34-LM,
// 256-d) from the safetensors mirror, which is the validated
// network+weights+frontend trio. M25.1.
import AthenaCore
import Foundation
import HuggingFace
import MLX
import MLXHuggingFace
import MLXLMCommon
import MLXNN

/// Loads + runs WeSpeaker ResNet34-LM. Not an actor — the governed
/// `MLXSpeakerEmbeddingModule` owns the single instance and serialises
/// access.
final class WeSpeakerModel {
    let network: WeSpeakerNetwork
    let features = WeSpeakerFeatures()
    let embeddingDimension = 256

    private init(network: WeSpeakerNetwork) {
        self.network = network
    }

    /// Download (`*.json` + `*.safetensors`) and load. HF cache root
    /// follows `HF_HOME` (SSD/local fallback); fetch goes through the
    /// egress proxy like every other model pull.
    static func fromPretrained(_ repoId: String) async throws -> WeSpeakerModel {
        let modelURL = try await #hubDownloader(
            HuggingFace.HubClient(session: AthenaProxy.proxiedURLSession())
        ).download(
            id: repoId, revision: nil,
            matching: ["*.json", "*.safetensors"],
            useLatest: false, progressHandler: { _ in })
        return try fromModelDirectory(modelURL)
    }

    static func fromModelDirectory(_ url: URL) throws -> WeSpeakerModel {
        let st = try FileManager.default.contentsOfDirectory(
            at: url, includingPropertiesForKeys: nil
        ).first { $0.pathExtension == "safetensors" }
        guard let st else {
            throw AthenaError.moduleLoadFailed(
                .speakerEmbedding,
                reason: "no .safetensors weights in \(url.path)")
        }
        let weights = try loadArrays(url: st)
        let network = WeSpeakerNetwork()
        try network.update(
            parameters: ModuleParameters.unflattened(weights),
            verify: .noUnusedKeys)
        eval(network.parameters())
        return WeSpeakerModel(network: network)
    }

    /// Extract one 256-d L2-normalized embedding from 16 kHz mono PCM.
    func embed(_ samples: [Float]) -> [Float] {
        let mel = features.extract(samples)  // [T, 80]
        guard mel.dim(0) > 0 else {
            return [Float](repeating: 0, count: embeddingDimension)
        }
        let input = mel.reshaped(1, mel.dim(0), mel.dim(1), 1)  // [1,T,80,1]
        let emb = network(input)
        eval(emb)
        let out = emb[0].asArray(Float.self)
        // End-of-call allocator-pool flush (M50.2). The wrapper module
        // calls `embed` once per audio segment in a loop; without this,
        // per-call ResNet activations accumulate in MLX's pool exactly
        // like the embedder did pre-M46.6. `out` is already a Swift
        // [Float] copy so nothing downstream needs the MLXArray.
        MLX.Memory.clearCache()
        return out
    }
}
