import ArgumentParser
import AthenaClient
import AthenaCore
import AthenaDeploy
import AthenaLLM
import Foundation

/// `athena default [--module M] [NAME]` — show or set a module's default
/// served model (ADR 026). The allowlist is retired; availability is the
/// model store, and the per-module default lives in the TOML config. Each
/// module maps to a config key (the LLM keeps the historical `model` key):
///   llm → model, textEmbedding → embedding_model,
///   transcription → transcription_model, diarization → diarization_model,
///   speakerEmbedding → speaker_embedding_model.
/// No NAME prints the effective default; NAME writes the key via the shared
/// `ConfigEditor`. M9.5d / ADR 026.
struct Default: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "default",
        abstract: "Show or set a module's default served model.")

    @Option(
        help:
            "Module class: llm, textEmbedding, transcription, diarization, speakerEmbedding."
    )
    var module: String = "llm"

    @Argument(help: "Model name to make default (omit to show).")
    var name: String?
    @Option(help: "Config file (overrides auto-resolution).")
    var config: String?

    @OptionGroup var daemon: DaemonOptions

    /// The TOML config key + the AthenaConfig accessor for a module class.
    private static func configKey(for module: ModuleID) -> String {
        switch module {
        case .llm: return "model"
        case .textEmbedding: return "embedding_model"
        case .transcription: return "transcription_model"
        case .diarization: return "diarization_model"
        case .speakerEmbedding: return "speaker_embedding_model"
        }
    }

    private static func configuredDefault(
        for module: ModuleID, in cfg: AthenaConfig
    ) -> String? {
        switch module {
        case .llm: return cfg.model
        case .textEmbedding: return cfg.embeddingModel
        case .transcription: return cfg.transcriptionModel
        case .diarization: return cfg.diarizationModel
        case .speakerEmbedding: return cfg.speakerEmbeddingModel
        }
    }

    func run() async throws {
        guard let moduleId = ModuleID(rawValue: module) else {
            throw ValidationError(
                "unknown module '\(module)' — one of "
                    + ModuleID.allCases.map(\.rawValue)
                    .joined(separator: ", "))
        }

        if daemon.isRemote {
            // ADR 026 — a module's default is a config key on the daemon HOST;
            // there is no runtime API to set it (the allowlist-default endpoint
            // is retired). The LLM default is still readable/settable via the
            // model-default endpoint; other modules must be edited in the
            // daemon's TOML on the host.
            guard moduleId == .llm else {
                throw ValidationError(
                    "setting the \(module) default on a remote daemon is not "
                        + "supported — edit "
                        + "\(Self.configKey(for: moduleId)) in the daemon's "
                        + "athena.toml on its host, then restart it.")
            }
            if let name {
                try await RemoteModels.setDefault(daemon, name: name)
            } else {
                try await RemoteModels.getDefault(daemon)
            }
            return
        }

        let url = ConfigEditor.resolvePath(config)
        let key = Self.configKey(for: moduleId)
        if let name {
            ConfigEditor.setScalar(key: key, value: name, in: url)
            print("default \(module) model = \(name)  (\(url.path))")
            return
        }
        if let cfg = try? AthenaConfig.parse(file: url),
            let def = Self.configuredDefault(for: moduleId, in: cfg)
        {
            print(def)
        } else if moduleId == .llm {
            print(
                "\(ModelStore.defaultModelName)  "
                    + "(built-in default; unset in config)")
        } else {
            print(
                "(no default configured for \(module); the store's sole model "
                    + "of this class is used, else the request must name one)")
        }
    }
}
