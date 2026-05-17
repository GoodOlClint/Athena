import ArgumentParser

/// `athena` — the Project the platform inference appliance. Ollama-style CLI
/// (M6) + queue/vectors/store (M9.1–9.3) + daemon lifecycle (M9.4).
@main
struct Athena: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "athena",
        abstract: "Project the platform inference appliance (passive oracle).",
        version: "0.9.10",
        subcommands: [
            Load.self, Install.self, ListModels.self, Ps.self,
            Run.self, Pull.self, Convert.self, Verify.self, Rm.self,
            Show.self, Unload.self,
            Start.self, Stop.self, Status.self, Config.self,
            Queue.self, Vectors.self, Store.self,
        ],
        defaultSubcommand: Load.self
    )
}
