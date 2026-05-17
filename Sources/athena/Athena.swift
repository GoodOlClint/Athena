import ArgumentParser

/// `athena` — the Project the platform inference appliance. Ollama-style CLI
/// (M6) + `queue` (M9.1); `vectors`/`store` land in later M9 slices.
@main
struct Athena: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "athena",
        abstract: "Project the platform inference appliance (passive oracle).",
        version: "0.9.3",
        subcommands: [
            Serve.self, Install.self, ListModels.self, Ps.self,
            Run.self, Pull.self, Rm.self, Show.self, Stop.self,
            Queue.self, Vectors.self, Store.self,
        ],
        defaultSubcommand: Serve.self
    )
}
