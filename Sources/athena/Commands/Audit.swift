import ArgumentParser
import AthenaClient
import AthenaStore
import Foundation

/// `athena audit` — the append-only RBAC/admin audit trail (M30.2).
/// Overloads local/remote like the other unified verbs: a loopback
/// `--host` reads THIS box's store directly (the operator sees the
/// whole trail); an off-box `--host` routes to that daemon's
/// `/api/audit` over HTTP, which is admin-only. Pull only — the
/// passive oracle never pushes the trail out.
// Named `AuditCommand` (not `Audit`) to mirror UsageCommand and avoid
// any name collision with the portable `AuditCmd`.
struct AuditCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "audit",
        abstract: "Show the RBAC/admin audit trail.")

    @OptionGroup var daemon: DaemonOptions
    @Option(help: "Filter by acting principal (e.g. u:admin).")
    var principal: String?
    @Option(help: "Filter by action (e.g. user.create).")
    var action: String?
    @Option(help: "Only entries at/after this epoch-seconds time.")
    var since: Double?
    @Option(help: "Max rows (default 100, capped 1000).")
    var limit: Int?
    @Option(help: "Data dir (default: configured / ~/.athena).")
    var dataDir: String?

    func run() async throws {
        if daemon.isRemote {
            try await RemoteAudit.show(
                daemon, principal: principal, action: action,
                since: since, limit: limit)
            return
        }
        // Local: read the store directly (works while the daemon holds
        // it open — SQLite WAL allows concurrent readers).
        guard let db = try? AthenaStore(path: storeDBPath(dataDir))
        else {
            print("no audit entries")
            return
        }
        let rows = await db.listAudit(
            principal: principal, action: action, since: since,
            limit: min(1000, max(1, limit ?? 100)))
        RemoteAudit.render(
            rows.map {
                RemoteAudit.Entry(
                    id: $0.id, ts: $0.ts, principal: $0.principal,
                    action: $0.action, target: $0.target,
                    result: $0.result, detail: $0.detail)
            })
    }
}
