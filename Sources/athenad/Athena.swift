import ArgumentParser
import AthenaClient

/// `athenad` — the Project the platform inference daemon + Apple-host
/// operator CLI (MLX/Metal; macOS-only). The portable `athena`
/// client is a separate cross-platform binary. M14.2a.
@main
struct Athenad: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "athenad",
        abstract:
            "Project the platform inference daemon (passive oracle).",
        version: "0.9.40",
        subcommands: [
            Load.self, Install.self, ListModels.self, Ps.self,
            Run.self, Pull.self, Convert.self, Verify.self,
            Prune.self, Cp.self, Default.self, Rm.self, Show.self,
            Unload.self,
            Start.self, Stop.self, Status.self, Config.self,
            Doctor.self, Logs.self, Uninstall.self, Auth.self,
            Hf.self, Proxy.self, Queue.self, Vectors.self,
            Store.self,
        ],
        defaultSubcommand: Load.self
    )
}
