import ArgumentParser
import AthenaClient

/// `athena` — the portable Project the platform client. Cross-platform thin
/// CLI that talks to a (local or remote) `athenad` daemon over HTTP;
/// no MLX/Metal. The daemon + Apple-host operator tooling lives in the
/// separate `athenad` binary. M14.2a (more verbs land in M14.2b/M17).
@main
struct Athena: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "athena",
        abstract: "Project the platform client (talks to an athenad).",
        version: "0.9.40",
        subcommands: [
            Queue.self, Vectors.self, Store.self, AuthClient.self,
        ])
}
