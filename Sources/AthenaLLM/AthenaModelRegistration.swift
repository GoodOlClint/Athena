import AthenaModels
import Foundation
import MLXLLM
import MLXLMCommon

/// Registers Athena's vendored Qwen3.5 model into the substrate's public,
/// last-write-wins model-type registry, overriding the upstream
/// `qwen3_5*` mappings. This keeps `mlx-swift-lm` a pristine path
/// dependency: the local-dir loader resolves `model_type` through this
/// same registry, so a Qwen3.5 directory loads Athena's class. M2 grows
/// the MTP head on `AthenaQwen35*` without touching the clone.
enum AthenaModelRegistration {
    private static func creator<C: Codable, M: LanguageModel>(
        _ type: C.Type, _ make: @escaping (C) -> M
    ) -> (Data) throws -> LanguageModel {
        { data in
            let config: C
            if let json5 = try? JSONDecoder.json5().decode(C.self, from: data)
            {
                config = json5
            } else {
                config = try JSONDecoder().decode(C.self, from: data)
            }
            return make(config)
        }
    }

    /// Idempotent (registry is last-write-wins); safe to call before every
    /// load.
    static func install() async {
        let registry = LLMModelFactory.shared.typeRegistry
        await registry.registerModelType(
            "qwen3_5",
            creator: creator(
                AthenaQwen35Configuration.self, AthenaQwen35Model.init))
        await registry.registerModelType(
            "qwen3_5_moe",
            creator: creator(
                AthenaQwen35Configuration.self, AthenaQwen35MoEModel.init))
        await registry.registerModelType(
            "qwen3_5_text",
            creator: creator(
                AthenaQwen35TextConfiguration.self,
                AthenaQwen35TextModel.init))
    }
}
