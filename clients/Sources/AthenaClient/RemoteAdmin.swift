import ArgumentParser
import Foundation

#if canImport(FoundationNetworking)
    import FoundationNetworking
#endif

// M17.3 — remote daemon/RBAC posture. `GET /api/admin/status` (M16.5)
// is the native admin view (model, listen address, auth posture,
// user/token/admin counts) — distinct from the always-open
// `/healthz` governor snapshot. Gated `daemon.admin` server-side; the
// client only surfaces the outcome. On macOS `status` keeps its
// local pidfile + `/healthz` view and routes here when `--host` is
// off-box; the portable `status` is this, always.

public enum RemoteAdmin {
    public static func status(_ d: DaemonOptions) async throws {
        let (code, data): (Int, Data)
        do {
            (code, data) = try await HTTPClient.send(
                "GET", d.base + "/api/admin/status", key: d.authKey)
        } catch { HTTPClient.noDaemon(d, error) }
        HTTPClient.printJSON(data)
        if code >= 400 { throw ExitCode.failure }
    }
}

/// `athena status` on the portable client — the daemon's native admin
/// posture over HTTP. (Distinct Swift type name: the monorepo
/// `athena` target imports AthenaClient AND defines its own local
/// `Status`, so a shared `Status` would be ambiguous there.)
public struct CStatus: AsyncParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "status",
        abstract: "Daemon model + RBAC posture (/api/admin/status).")
    @OptionGroup public var daemon: DaemonOptions
    public init() {}
    public func run() async throws {
        try await RemoteAdmin.status(daemon)
    }
}
