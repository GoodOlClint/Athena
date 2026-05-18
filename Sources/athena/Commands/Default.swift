import ArgumentParser
import AthenaClient
import AthenaDeploy
import AthenaLLM
import Foundation

/// `athena default [NAME]` — show or set the model the single-model
/// shim serves. No arg prints the effective default (config `model`,
/// or the built-in fallback when unset). With NAME, writes `model`
/// into the TOML via the shared `ConfigEditor`. M9.5d.
struct Default: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "default",
        abstract: "Show or set the default served model.")

    @Argument(help: "Model name to make default (omit to show).")
    var name: String?
    @Option(help: "Config file (overrides auto-resolution).")
    var config: String?

    @OptionGroup var daemon: DaemonOptions

    func run() async throws {
        if daemon.isRemote {
            if let name {
                try await RemoteModels.setDefault(
                    daemon, name: name)
            } else {
                try await RemoteModels.getDefault(daemon)
            }
            return
        }
        let url = ConfigEditor.resolvePath(config)
        if let name {
            ConfigEditor.setScalar(key: "model", value: name, in: url)
            print("default model = \(name)  (\(url.path))")
            return
        }
        if let cfg = try? AthenaConfig.parse(file: url),
            let model = cfg.model
        {
            print(model)
        } else {
            print(
                "\(ModelStore.defaultModelName)  "
                    + "(built-in default; unset in config)")
        }
    }
}
