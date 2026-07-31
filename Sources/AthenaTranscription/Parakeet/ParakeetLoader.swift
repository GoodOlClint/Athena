// Loader for the Parakeet-TDT-0.6B-v3 MLX port (ADR 020). The serve path loads
// from the local model-store dir via `fromModelDirectory` (inference never
// auto-downloads); `fromPretrained` (governed #hubDownloader, same mechanism as
// WeSpeakerModel/SortformerModel) backs the operator pull + the gated test. It
// reads `config.json` for the vocabulary, loads the safetensors and maps keys
// onto the module tree.
//
// Conv weights in v3 are already MLX-layout (no transpose). The only key
// surgery: BatchNorm `num_batches_tracked` is dropped (not a parameter), and
// we assert a handful of critical tensors are present & non-zero so a silent
// key mismatch can't masquerade as a working forward on random init.
import AthenaCore
import Foundation
import HuggingFace
import MLX
import MLXHuggingFace
import MLXLMCommon
import MLXNN

public enum ParakeetLoadError: Error, CustomStringConvertible {
    case noWeights(String)
    case noConfig(String)
    case missingCriticalTensor(String)
    case zeroCriticalTensor(String)
    public var description: String {
        switch self {
        case .noWeights(let p): return "no .safetensors in \(p)"
        case .noConfig(let p): return "no config.json in \(p)"
        case .missingCriticalTensor(let k):
            return "critical tensor missing after load: \(k)"
        case .zeroCriticalTensor(let k):
            return "critical tensor is all-zero (random init?): \(k)"
        }
    }
}

private struct ParakeetConfigJSON: Decodable {
    struct Joint: Decodable { let vocabulary: [String] }
    let joint: Joint
}

public enum ParakeetLoader {
    public static func fromPretrained(_ repoId: String) async throws
        -> ParakeetTDTModel
    {
        let url = try await #hubDownloader(
            HuggingFace.HubClient(session: AthenaProxy.proxiedURLSession())
        ).download(
            id: repoId, revision: nil,
            matching: ["*.json", "*.safetensors"],
            useLatest: false, progressHandler: { _ in })
        return try fromModelDirectory(url)
    }

    public static func fromModelDirectory(_ url: URL) throws -> ParakeetTDTModel {
        let configURL = url.appendingPathComponent("config.json")
        guard FileManager.default.fileExists(atPath: configURL.path) else {
            throw ParakeetLoadError.noConfig(url.path)
        }
        let cfg = try JSONDecoder().decode(
            ParakeetConfigJSON.self, from: Data(contentsOf: configURL))
        let vocab = cfg.joint.vocabulary

        let stFiles = try FileManager.default.contentsOfDirectory(
            at: url, includingPropertiesForKeys: nil
        ).filter { $0.pathExtension == "safetensors" }
        guard !stFiles.isEmpty else {
            throw ParakeetLoadError.noWeights(url.path)
        }

        var raw = [String: MLXArray]()
        for f in stFiles {
            for (k, v) in try loadArrays(url: f) { raw[k] = v }
        }

        // Drop non-parameter bookkeeping tensors and remap the two list-style
        // key groups whose numeric children would otherwise make MLX infer a
        // (gappy) array that can't bind to our flat fields:
        //   encoder.pre_encode.conv.{0,2,3,5,6}. → ...conv{0,2,3,5,6}.
        //   joint.joint_net.2.                   → joint.joint_net2.
        // The dec_rnn.lstm.{0,1} group IS a real contiguous array in our model,
        // so it is left as-is.
        var weights = [String: MLXArray]()
        for (k, v) in raw where !k.contains("num_batches_tracked") {
            var nk = k
            if nk.contains("pre_encode.conv.") {
                for i in [0, 2, 3, 5, 6] {
                    nk = nk.replacingOccurrences(
                        of: "pre_encode.conv.\(i).", with: "pre_encode.conv\(i).")
                }
            }
            nk = nk.replacingOccurrences(
                of: "joint_net.2.", with: "joint_net2.")
            weights[nk] = v
        }

        // Critical-tensor presence + non-zero guard (so a key mismatch that
        // routes update(verify:.none) past silent gaps can't report a
        // benchmark on random init).
        // Post-remap names.
        let critical = [
            "encoder.pre_encode.conv0.weight",
            "encoder.pre_encode.out.weight",
            "encoder.layers.0.self_attn.linear_q.weight",
            "encoder.layers.0.self_attn.pos_bias_u",
            "encoder.layers.23.norm_out.weight",
            "decoder.prediction.embed.weight",
            "decoder.prediction.dec_rnn.lstm.0.Wx",
            "joint.enc.weight",
            "joint.joint_net2.weight",
        ]
        for key in critical {
            guard let t = weights[key] else {
                throw ParakeetLoadError.missingCriticalTensor(key)
            }
            let s = MLX.sum(MLX.abs(t)).item(Float.self)
            if s == 0 {
                throw ParakeetLoadError.zeroCriticalTensor(key)
            }
        }

        let model = ParakeetTDTModel(vocabulary: vocab)
        try model.update(
            parameters: ModuleParameters.unflattened(weights),
            verify: .noUnusedKeys)
        eval(model.parameters())
        return model
    }
}
