import ArgumentParser

/// `athena` — the Project the platform inference appliance. M0 ships `serve`; the
/// full Ollama-style surface (run/pull/list/ps/stop/show/rm) lands across
/// later milestones.
@main
struct Athena: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "athena",
        abstract: "Project the platform inference appliance (passive oracle).",
        version: "0.1.0",
        subcommands: [Serve.self],
        defaultSubcommand: Serve.self
    )
}
