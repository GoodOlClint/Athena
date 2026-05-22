import ArgumentParser
import AthenaClient
import AthenaStore
import Foundation

/// `athena usage` — per-principal token accounting (M27.3). Overloads
/// local/remote like the other unified verbs: a loopback `--host`
/// reads THIS box's store directly (the operator sees every
/// principal); an off-box `--host` routes to that daemon's
/// `/api/usage` over HTTP, where the server owner-scopes the result
/// (a member sees its own row, an admin sees all). Pull only — the
/// passive oracle never pushes usage out.
// Named `UsageCommand` (not `Usage`) to avoid colliding with the
// OpenAI `Usage` response DTO in the same module.
struct UsageCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "usage",
        abstract: "Show per-principal token usage.")

    @OptionGroup var daemon: DaemonOptions
    @Option(help: "Data dir (default: configured / ~/.athena).")
    var dataDir: String?

    func run() async throws {
        if daemon.isRemote {
            try await RemoteUsage.show(daemon)
            return
        }
        // Local: read the store directly (works while the daemon holds
        // it open — SQLite WAL allows concurrent readers).
        guard let db = try? AthenaStore(path: storeDBPath(dataDir))
        else {
            print("no usage recorded")
            return
        }
        RemoteUsage.render(
            await db.allUsage().map {
                RemoteUsage.Entry(
                    principal: $0.principal, requests: $0.requests,
                    prompt_tokens: $0.promptTokens,
                    completion_tokens: $0.completionTokens,
                    total_tokens: $0.totalTokens, updated: $0.updated)
            })
    }
}
