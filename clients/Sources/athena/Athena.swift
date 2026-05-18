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
        abstract: "Project the platform client (talks to a daemon).",
        version: "0.9.49",
        subcommands: [
            Run.self, Ps.self, Unload.self,
            Queue.self, Vectors.self, Store.self, AuthClient.self,
        ])
}
