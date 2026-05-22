import ArgumentParser
import Foundation

#if canImport(FoundationNetworking)
    import FoundationNetworking
#endif

// M30.2 — remote RBAC/admin audit trail. The portable `athena`
// (Linux/Windows: no local daemon) reads a daemon's `/api/audit` over
// HTTP; the bearer is `DaemonOptions.authKey`. The endpoint is
// admin-only (daemon.admin) and filterable. On macOS the same `audit`
// verb reads the local store directly by default and routes here when
// `--host` is off-box (`DaemonOptions.isRemote`); the portable struct
// below ALWAYS goes remote. Pull only — the passive oracle never
// pushes the trail out.
public enum RemoteAudit {
    /// One audit entry. Public + memberwise so the macOS local path
    /// (reading AthenaStore) can reuse `render`.
    public struct Entry: Decodable {
        public let id: Int
        public let ts: Double
        public let principal: String
        public let action: String
        public let target: String?
        public let result: String
        public let detail: String?
        public init(
            id: Int, ts: Double, principal: String, action: String,
            target: String?, result: String, detail: String?
        ) {
            self.id = id
            self.ts = ts
            self.principal = principal
            self.action = action
            self.target = target
            self.result = result
            self.detail = detail
        }
    }
    private struct Report: Decodable { let audit: [Entry] }

    /// `/api/audit` query string from the filters (values percent-
    /// encoded so `u:admin` / `user.create` survive the wire).
    static func query(
        principal: String?, action: String?, since: Double?,
        limit: Int?
    ) -> String {
        func enc(_ s: String) -> String {
            s.addingPercentEncoding(
                withAllowedCharacters: .alphanumerics) ?? s
        }
        var parts: [String] = []
        if let p = principal, !p.isEmpty {
            parts.append("principal=\(enc(p))")
        }
        if let a = action, !a.isEmpty {
            parts.append("action=\(enc(a))")
        }
        if let s = since { parts.append("since=\(s)") }
        if let l = limit { parts.append("limit=\(l)") }
        return parts.isEmpty ? "" : "?" + parts.joined(separator: "&")
    }

    public static func show(
        _ d: DaemonOptions, principal: String? = nil,
        action: String? = nil, since: Double? = nil,
        limit: Int? = nil
    ) async throws {
        let q = query(
            principal: principal, action: action, since: since,
            limit: limit)
        let (code, data): (Int, Data)
        do {
            (code, data) = try await HTTPClient.send(
                "GET", d.base + "/api/audit" + q, key: d.authKey)
        } catch { HTTPClient.noDaemon(d, error) }
        guard code < 400,
            let r = try? JSONDecoder().decode(Report.self, from: data)
        else {
            HTTPClient.printJSON(data)
            if code >= 400 { throw ExitCode.failure }
            return
        }
        render(r.audit)
    }

    private static let stamp: DateFormatter = {
        let f = DateFormatter()
        f.dateFormat = "yyyy-MM-dd HH:mm:ss"
        return f
    }()

    /// Render the audit table (shared by the remote and macOS-local
    /// paths). Left-aligned, fixed columns; long fields truncate.
    public static func render(_ entries: [Entry]) {
        guard !entries.isEmpty else {
            print("no audit entries")
            return
        }
        func pad(_ s: String, _ w: Int) -> String {
            s.count >= w
                ? String(s.prefix(w))
                : s + String(repeating: " ", count: w - s.count)
        }
        print(
            pad("TIME", 20) + pad("PRINCIPAL", 18)
                + pad("ACTION", 20) + pad("TARGET", 22)
                + pad("RESULT", 9) + "DETAIL")
        for e in entries {
            let t = stamp.string(
                from: Date(timeIntervalSince1970: e.ts))
            print(
                pad(t, 20) + pad(e.principal, 18)
                    + pad(e.action, 20) + pad(e.target ?? "-", 22)
                    + pad(e.result, 9) + (e.detail ?? ""))
        }
    }
}

// MARK: - Portable command struct (remote-only)

/// `athena audit` on the portable client — always remote. On macOS the
/// same name is the local-overloading verb in Sources/athena/Commands.
public struct AuditCmd: AsyncParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "audit",
        abstract: "Show the RBAC/admin audit trail from the daemon.")
    @OptionGroup public var daemon: DaemonOptions
    @Option(help: "Filter by acting principal (e.g. u:admin).")
    public var principal: String?
    @Option(help: "Filter by action (e.g. user.create).")
    public var action: String?
    @Option(help: "Only entries at/after this epoch-seconds time.")
    public var since: Double?
    @Option(help: "Max rows (default 100, capped 1000).")
    public var limit: Int?
    public init() {}
    public func run() async throws {
        try await RemoteAudit.show(
            daemon, principal: principal, action: action,
            since: since, limit: limit)
    }
}
