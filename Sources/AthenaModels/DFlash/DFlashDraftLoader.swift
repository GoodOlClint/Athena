import Foundation
import MLX
import MLXNN

// Standalone loader for a z-lab DFlash drafter checkpoint (config.json +
// model.safetensors). Mirrors the `WhisperLoader` precedent: the
// checkpoint key layout matches the `DFlashDraftModel` property keys
// exactly, so the safetensors map is applied directly. The drafter carries
// NO embed/head tensors (it shares the target's), so nothing is dropped.
// M63.1.
public enum DFlashDraftLoader {
    public enum LoadError: Error, CustomStringConvertible {
        case missingFile(String)
        case badConfig(String)
        public var description: String {
            switch self {
            case .missingFile(let f): return "dflash-draft: missing \(f)"
            case .badConfig(let s): return "dflash-draft: bad config — \(s)"
            }
        }
    }

    /// Load a drafter from a local checkpoint directory.
    public static func load(directory dir: URL) throws -> DFlashDraftModel {
        let configURL = dir.appending(component: "config.json")
        guard let configData = try? Data(contentsOf: configURL) else {
            throw LoadError.missingFile("config.json")
        }
        let config: DFlashDraftConfiguration
        do {
            config = try JSONDecoder().decode(
                DFlashDraftConfiguration.self, from: configData)
        } catch {
            throw LoadError.badConfig(String(describing: error))
        }

        let weightsURL = dir.appending(component: "model.safetensors")
        guard FileManager.default.fileExists(atPath: weightsURL.path) else {
            throw LoadError.missingFile("model.safetensors")
        }

        let weights = try MLX.loadArrays(url: weightsURL)
        let model = DFlashDraftModel(config)
        try model.update(
            parameters: ModuleParameters.unflattened(weights), verify: .all)
        eval(model)
        return model
    }
}
