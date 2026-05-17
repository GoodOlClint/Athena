import ArgumentParser

/// `athena` — the Project the platform inference appliance. Ollama-style CLI
/// (M6): `serve`, `install`, `list`/`ls`, `ps`, `run`, `pull`, `rm`,
/// `show`; `stop` lands later.
@main
struct Athena: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "athena",
        abstract: "Project the platform inference appliance (passive oracle).",
        version: "0.8.1",
        subcommands: [
            Serve.self, Install.self, ListModels.self, Ps.self,
            Run.self, Pull.self, Rm.self, Show.self,
        ],
        defaultSubcommand: Serve.self
    )
}
