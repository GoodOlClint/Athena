import ArgumentParser
import AthenaClient

/// `athena` — the portable Project the platform client (Linux/Windows, and
/// the macOS client subset). Same command everywhere; this build
/// carries only the verbs that work against a (local or remote)
/// daemon over HTTP. There is no local daemon to manage off-Apple,
/// so lifecycle/operator verbs are intentionally absent here — the
/// full macOS `athena` (the monorepo package) adds them. M14.4.
@main
struct Athena: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "athena",
        abstract:
            "Project the platform client — remote inference, model store, "
            + "and RBAC admin over a daemon's HTTP API.",
        version: "0.10.3",
        subcommands: [
            Run.self, Ps.self, CStatus.self, Unload.self,
            ListCmd.self, ShowCmd.self, RmCmd.self, CpCmd.self,
            DefaultCmd.self, PullCmd.self, ConvertCmd.self,
            PruneCmd.self,
            Queue.self, Vectors.self, Store.self, AuthClient.self,
            UsageCmd.self,
        ])
}
