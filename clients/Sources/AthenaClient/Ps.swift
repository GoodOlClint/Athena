import ArgumentParser
import Foundation

#if canImport(FoundationNetworking)
    import FoundationNetworking
#endif

/// `athena ps` — governed module state from a running daemon
/// (`GET /healthz`), formatted compactly. Decodes a local mirror of
/// the governor snapshot so the portable client carries no dependency
/// on the Apple-only `AthenaCore` (M14.3).
public struct Ps: AsyncParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "ps",
        abstract: "Show governed module state from a running daemon."
    )

    @Option(help: "Daemon host.")
    public var host: String = "127.0.0.1"

    @Option(help: "Daemon port.")
    public var port: Int = athenaDefaultPort

    public init() {}

    /// The subset of `/healthz` the client renders.
    private struct HealthSnapshot: Decodable {
        struct Module: Decodable {
            let id: String
            let state: String
            let reservedBytes: Int
            let evictable: Bool
        }
        let totalBudgetBytes: Int
        let reservedBytes: Int
        let freeBytes: Int
        let promptCacheCapBytes: Int
        let modules: [Module]
    }

    public func run() async throws {
        let url = URL(string: "http://\(host):\(port)/healthz")!
        let snap: HealthSnapshot
        do {
            let (data, _) = try await URLSession.shared.data(
                from: url)
            snap = try JSONDecoder().decode(
                HealthSnapshot.self, from: data)
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
            "MODULE".padding(
                toLength: 16, withPad: " ", startingAt: 0)
                + "STATE".padding(
                    toLength: 12, withPad: " ", startingAt: 0)
                + "RESERVED".padding(
                    toLength: 12, withPad: " ", startingAt: 0)
                + "EVICTABLE")
        for m in snap.modules.sorted(by: { $0.id < $1.id }) {
            print(
                m.id.padding(
                    toLength: 16, withPad: " ", startingAt: 0)
                    + m.state.padding(
                        toLength: 12, withPad: " ", startingAt: 0)
                    + humanBytes(m.reservedBytes).padding(
                        toLength: 12, withPad: " ", startingAt: 0)
                    + (m.evictable ? "yes" : "no"))
        }
    }
}
