//
//  Qwen35MoE.swift
//  mlx-swift-lm
//
//  Created by John Mai on 2026/2/9.
//
//  Port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen3_5_moe.py
//

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

public struct AthenaQwen35Configuration: Codable, Sendable {
    var modelType: String
    var textConfig: AthenaQwen35TextConfiguration

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case textConfig = "text_config"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.modelType = try container.decode(String.self, forKey: .modelType)

        if let textConfig = try container.decodeIfPresent(
            AthenaQwen35TextConfiguration.self, forKey: .textConfig)
        {
            self.textConfig = textConfig
        } else {
            self.textConfig = try AthenaQwen35TextConfiguration(from: decoder)
        }
    }

    /// A copy with the MTP head disabled (see
    /// `AthenaQwen35TextConfiguration.withMTPDisabled`).
    public func withMTPDisabled() -> AthenaQwen35Configuration {
        var copy = self
        copy.textConfig = copy.textConfig.withMTPDisabled()
        return copy
    }
}

public class AthenaQwen35MoEModel: AthenaQwen35Model {

    override public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var newWeights = [String: MLXArray]()
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
            newWeights[key] = value
        }

        for l in 0 ..< languageModel.configuration.hiddenLayers {
            let prefix = "language_model.model.layers.\(l).mlp"
            let gateUpKey = "\(prefix).experts.gate_up_proj"
            if let gateUp = newWeights[gateUpKey] {
                newWeights[gateUpKey] = nil
                let mid = gateUp.dim(-2) / 2
                newWeights["\(prefix).switch_mlp.gate_proj.weight"] =
                    gateUp[.ellipsis, ..<mid, 0...]
                newWeights["\(prefix).switch_mlp.up_proj.weight"] =
                    gateUp[.ellipsis, mid..., 0...]
                if let downProj = newWeights["\(prefix).experts.down_proj"] {
                    newWeights["\(prefix).experts.down_proj"] = nil
                    newWeights["\(prefix).switch_mlp.down_proj.weight"] = downProj
                }
            }
        }

        // MTP layer(s): unlike the backbone layers (shipped pre-fused as
        // `experts.gate_up_proj`), some checkpoints store the MTP MoE
        // experts PER-EXPERT — `mtp.layers.L.mlp.experts.{0..E-1}.{gate,
        // up,down}_proj.weight` (e.g. Qwen3.5-122B-A10B). The
        // `SparseMoeBlock` binds the stacked `switch_mlp.*` tensors, so
        // fold the per-expert weights together here (routed experts only;
        // the router `gate`, `shared_expert.*`, and `shared_expert_gate`
        // map directly). Stacked along axis 0, matching the backbone
        // path's `[num_experts, …]` shape.
        let cfg = languageModel.configuration
        if cfg.numExperts > 0 {
            for l in 0 ..< cfg.mtpNumHiddenLayers {
                let prefix = "language_model.mtp.layers.\(l).mlp"
                for proj in ["gate_proj", "up_proj", "down_proj"] {
                    let perExpert = (0 ..< cfg.numExperts).map {
                        "\(prefix).experts.\($0).\(proj).weight"
                    }
                    guard perExpert.allSatisfy({ newWeights[$0] != nil })
                    else { continue }
                    let fused = stacked(
                        perExpert.map { newWeights[$0]! }, axis: 0)
                    for k in perExpert { newWeights[k] = nil }
                    newWeights["\(prefix).switch_mlp.\(proj).weight"] = fused
                }
            }
        }

        return languageModel.sanitize(weights: newWeights)
    }
}
