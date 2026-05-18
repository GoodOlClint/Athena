import ArgumentParser
import AthenaCore
import Foundation

/// `athena ps` — governed module state from a running daemon
/// (`GET /healthz`), formatted ollama-style.
struct Ps: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "ps",
        abstract: "Show governed module state from a running daemon."
    )

    @Option(help: "Daemon host.")
    var host: String = "127.0.0.1"

    @Option(help: "Daemon port.")
    var port: Int = GovernorConfig.defaultPort

    func run() async throws {
        let url = URL(string: "http://\(host):\(port)/healthz")!
        let snap: GovernorSnapshot
        do {
            let (data, _) = try await URLSession.shared.data(from: url)
            snap = try JSONDecoder().decode(
                GovernorSnapshot.self, from: data)
        } catch {
            print(
                "no running athena daemon at \(host):\(port) "
                    + "(\(error.localizedDescription))")
            throw ExitCode.failure
        }

        print(
            "budget \(humanBytes(snap.totalBudgetBytes)) | "
                + "reserved \(humanBytes(snap.reservedBytes)) | "
                + "free \(humanBytes(snap.freeBytes)) | "
                + "prompt-cache cap "
                + "\(humanBytes(snap.promptCacheCapBytes))")
        print(
            "MODULE".padding(toLength: 16, withPad: " ", startingAt: 0)
                + "STATE".padding(
                    toLength: 12, withPad: " ", startingAt: 0)
                + "RESERVED".padding(
                    toLength: 12, withPad: " ", startingAt: 0)
                + "EVICTABLE")
        for m in snap.modules.sorted(by: {
            $0.id.rawValue < $1.id.rawValue
        }) {
            print(
                m.id.rawValue.padding(
                    toLength: 16, withPad: " ", startingAt: 0)
                    + "\(m.state)".padding(
                        toLength: 12, withPad: " ", startingAt: 0)
                    + humanBytes(m.reservedBytes).padding(
                        toLength: 12, withPad: " ", startingAt: 0)
                    + (m.evictable ? "yes" : "no"))
        }
    }

    private func humanBytes(_ n: Int) -> String {
        ListModels.humanBytes(n)
    }
}
