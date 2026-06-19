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
            let residentBytes: Int
            let evictable: Bool
            /// Id of the model resident in this module's slot (nil when
            /// unloaded). Optional so the client still renders against a daemon
            /// that predates the field.
            let model: String?
            /// ADR 023 G3 — true ⇒ residentBytes is a measured footprint;
            /// false/absent ⇒ the pre-load estimate. Optional for back-compat;
            /// rendered as a leading `~` on the RESIDENT value when estimated.
            let measured: Bool?
        }
        let totalBudgetBytes: Int
        let residentBytes: Int
        let freeBytes: Int
        let promptCacheCapBytes: Int
        /// M55 — live phys_footprint (Activity Monitor "Memory"; counts
        /// GPU/Metal buffers RSS misses). Optional so the portable client
        /// still renders against a daemon that predates the field.
        let physFootprintBytes: Int?
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

        let physFootprint = snap.physFootprintBytes.map(humanBytes) ?? "n/a"
        print(
            "budget \(humanBytes(snap.totalBudgetBytes)) | "
                + "resident \(humanBytes(snap.residentBytes)) | "
                + "phys-footprint \(physFootprint) | "
                + "free \(humanBytes(snap.freeBytes)) | "
                + "prompt-cache cap "
                + "\(humanBytes(snap.promptCacheCapBytes))")
        print(
            "MODULE".padding(
                toLength: 16, withPad: " ", startingAt: 0)
                + "STATE".padding(
                    toLength: 12, withPad: " ", startingAt: 0)
                + "RESIDENT".padding(
                    toLength: 12, withPad: " ", startingAt: 0)
                + "MODEL".padding(
                    toLength: 40, withPad: " ", startingAt: 0)
                + "EVICTABLE")
        for m in snap.modules.sorted(by: { $0.id < $1.id }) {
            print(
                m.id.padding(
                    toLength: 16, withPad: " ", startingAt: 0)
                    + m.state.padding(
                        toLength: 12, withPad: " ", startingAt: 0)
                    + ((m.measured == false && m.residentBytes > 0
                        ? "~" : "") + humanBytes(m.residentBytes)).padding(
                        toLength: 12, withPad: " ", startingAt: 0)
                    + (m.model ?? "-").padding(
                        toLength: 40, withPad: " ", startingAt: 0)
                    + (m.evictable ? "yes" : "no"))
        }
        // ADR 023 G3 — note the marker only when something is still estimated.
        if snap.modules.contains(where: {
            $0.measured == false && $0.residentBytes > 0
        }) {
            print("(~ = estimated footprint; not yet measured at load)")
        }
    }
}
