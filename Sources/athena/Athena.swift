import ArgumentParser
import AthenaClient

/// `athena` — the Project the platform inference appliance: one unified CLI.
/// On macOS it carries everything — local daemon lifecycle
/// (`load`/`start`/`stop`/`status`), Apple-host operator ops
/// (`install`/`pull`/`convert`/…), and the HTTP client verbs
/// (`run`/`queue`/`vectors`/`store`/`auth`) that work against a local
/// OR remote daemon. Linux/Windows get the same `athena` command
/// built from the portable client subset only — there is no local
/// daemon to manage there, so those verbs are remote-only.
@main
struct Athena: AsyncParsableCommand {
    /// Single source of the daemon version — also stamped into the
    /// served OpenAPI document (`info.version`) so the spec can never
    /// report a version other than the build that serves it.
    static let appVersion = "0.10.142"

    static let configuration = CommandConfiguration(
        commandName: "athena",
        abstract:
            "Project the platform inference appliance (passive oracle).",
        version: Athena.appVersion,
        subcommands: [
            Load.self, Init.self, Install.self, ListModels.self, Ps.self,
            Run.self, Pull.self, Convert.self, Verify.self,
            Prune.self, Cp.self, Default.self, Rm.self, Show.self,
            Unload.self, ResidentCmd.self, AllowlistCmd.self,
            Start.self, Stop.self, Restart.self, Status.self,
            Config.self,
            Doctor.self, Logs.self, Uninstall.self, Auth.self,
            Hf.self, Proxy.self, Queue.self, Vectors.self,
            Store.self, UsageCommand.self, AuditCommand.self,
            CacheCmd.self,
        ],
        defaultSubcommand: Load.self
    )
}
