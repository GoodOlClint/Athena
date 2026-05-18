import AthenaCore
import Foundation
import HuggingFace
import MLX
import MLXHuggingFace
import MLXLMCommon
import MLXNN
import Tokenizers

/// Downloads an `mlx-community/whisper-*` checkpoint (config.json +
/// weights.safetensors) via the substrate Hub downloader and loads it
/// into a `WhisperModel`. The checkpoint key layout matches the module
/// property keys exactly, so the safetensors map is applied directly
/// (only the non-parameter `alignment_heads` tensor is dropped). M4.2b.
public enum WhisperLoader {
    public enum LoadError: Error, CustomStringConvertible {
        case missingFile(String)
        case badConfig(String)
        public var description: String {
            switch self {
            case .missingFile(let f): return "whisper: missing \(f)"
            case .badConfig(let s): return "whisper: bad config — \(s)"
            }
        }
    }

    /// Download (or cache-hit) and load. `modelId` defaults to the
    /// 4-layer-decoder turbo model. Cache root follows `HF_HOME`
    /// (SSD-or-local — set by the serve entrypoint).
    public static func load(
        modelId: String = "mlx-community/whisper-large-v3-turbo"
    ) async throws -> WhisperModel {
        let dir = try await #hubDownloader(
            HuggingFace.HubClient(
                session: AthenaProxy.proxiedURLSession())
        ).download(
            id: modelId, revision: nil,
            matching: ["*.json", "*.safetensors"],
            useLatest: false, progressHandler: { _ in })

        let configURL = dir.appending(component: "config.json")
        guard let configData = try? Data(contentsOf: configURL) else {
            throw LoadError.missingFile("config.json")
        }
        let config: WhisperConfig
        do {
            config = try JSONDecoder().decode(
                WhisperConfig.self, from: configData)
        } catch {
            throw LoadError.badConfig(String(describing: error))
        }

        let weightsURL = dir.appending(component: "weights.safetensors")
        guard FileManager.default.fileExists(atPath: weightsURL.path)
        else { throw LoadError.missingFile("weights.safetensors") }

        var weights = try MLX.loadArrays(url: weightsURL)
        // Non-parameter tensors (word-timestamp alignment table).
        weights.removeValue(forKey: "alignment_heads")

        let model = WhisperModel(config)
        try model.update(
            parameters: ModuleParameters.unflattened(weights),
            verify: .none)
        eval(model)
        return model
    }

    /// The Whisper GPT2-BPE tokenizer. The `mlx-community/whisper-*`
    /// repos ship NO tokenizer files, so it is sourced from
    /// `openai/whisper-large-v3` (tokenizer.json/vocab/merges) and loaded
    /// with the substrate HF tokenizer loader. Used only to decode
    /// generated ids → text (the forced prefix is built by id).
    public static func loadTokenizer(
        from tokenizerRepo: String = "openai/whisper-large-v3"
    ) async throws -> any MLXLMCommon.Tokenizer {
        let dir = try await #hubDownloader(
            HuggingFace.HubClient(
                session: AthenaProxy.proxiedURLSession())
        ).download(
            id: tokenizerRepo, revision: nil,
            matching: [
                "tokenizer.json", "tokenizer_config.json", "vocab.json",
                "merges.txt", "special_tokens_map.json",
                "added_tokens.json", "normalizer.json",
            ],
            useLatest: false, progressHandler: { _ in })
        return try await #huggingFaceTokenizerLoader().load(from: dir)
    }
}
