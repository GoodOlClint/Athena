import ArgumentParser
import AthenaClient

/// `athena` — the Project the platform inference appliance: one unified CLI.
/// On macOS it carries everything — local daemon lifecycle
/// (`load`/`start`/`stop`/`status`), Apple-host operator ops
/// (`install`/`pull`/`convert`/…), and the HTTP client verbs
/// (`run`/`queue`/`vectors`/`store`/`auth`) that work against a local
/// OR remote daemon. The daemon process itself is `athenad` (M14.2d;
/// users never type it). Linux/Windows get the same `athena` command
/// built from the portable client subset only — there is no local
/// daemon to manage there, so those verbs are remote-only. M14.2c.
@main
struct Athena: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "athena",
        abstract:
            "Project the platform inference appliance (passive oracle).",
        version: "0.9.78",
        subcommands: [
            Load.self, Init.self, Install.self, ListModels.self, Ps.self,
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
