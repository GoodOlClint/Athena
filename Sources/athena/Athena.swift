import ArgumentParser

/// `athena` — the Project the platform inference appliance. Ships `serve` and
/// `install`; the rest of the Ollama-style surface (run/pull/list/ps/
/// stop/show/rm) lands across later milestones.
@main
struct Athena: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "athena",
        abstract: "Project the platform inference appliance (passive oracle).",
        version: "0.3.0",
        subcommands: [Serve.self, Install.self],
        defaultSubcommand: Serve.self
    )
}
