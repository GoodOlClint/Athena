import ArgumentParser
import Foundation

#if canImport(FoundationNetworking)
    import FoundationNetworking
#endif

// M27.3 — remote per-principal token usage. The portable `athena`
// (Linux/Windows: no local daemon) reads a daemon's `/api/usage` over
// HTTP; the bearer is `DaemonOptions.authKey`. The SERVER owner-scopes
// (a member sees its own row, an admin sees all). On macOS the same
// `usage` verb reads the local store directly by default and routes
// here when `--host` is off-box (`DaemonOptions.isRemote`); the
// portable struct below ALWAYS goes remote. Pull only — the passive
// oracle never pushes usage out.
public enum RemoteUsage {
    /// One principal's cumulative usage. Public + memberwise so the
    /// macOS local path (reading AthenaStore) can reuse `render`.
    public struct Entry: Decodable {
        public let principal: String
        public let requests: Int
        public let prompt_tokens: Int
        public let completion_tokens: Int
        public let total_tokens: Int
        public let updated: Double
        /// ADR 041 — budget state; all nil when the principal has no budget
        /// (or, on the LOCAL path, when read straight from the store without
        /// the daemon's configured default in hand).
        public let budget: Int?
        public let period_tokens: Int?
        public let period_reset: String?
        public init(
            principal: String, requests: Int, prompt_tokens: Int,
            completion_tokens: Int, total_tokens: Int, updated: Double,
            budget: Int? = nil, period_tokens: Int? = nil,
            period_reset: String? = nil
        ) {
            self.principal = principal
            self.requests = requests
            self.prompt_tokens = prompt_tokens
            self.completion_tokens = completion_tokens
            self.total_tokens = total_tokens
            self.updated = updated
            self.budget = budget
            self.period_tokens = period_tokens
            self.period_reset = period_reset
        }
    }
    private struct Report: Decodable { let usage: [Entry] }

    public static func show(_ d: DaemonOptions) async throws {
        let (code, data): (Int, Data)
        do {
            (code, data) = try await HTTPClient.send(
                "GET", d.base + "/api/usage", key: d.authKey)
        } catch { HTTPClient.noDaemon(d, error) }
        guard code < 400,
            let r = try? JSONDecoder().decode(Report.self, from: data)
        else {
            HTTPClient.printJSON(data)
            if code >= 400 { throw ExitCode.failure }
            return
        }
        render(r.usage)
    }

    /// Render the usage table (shared by the remote and macOS-local
    /// paths). Right-aligned counts; columns sized for readability.
    public static func render(_ entries: [Entry]) {
        guard !entries.isEmpty else {
            print("no usage recorded")
            return
        }
        func r(_ s: String, _ w: Int) -> String {
            s.count >= w
                ? s
                : String(repeating: " ", count: w - s.count) + s
        }
        // ADR 041 — the budget columns appear only when at least one row has
        // one, so an appliance with no quotas renders exactly as before.
        let budgeted = entries.contains { ($0.budget ?? 0) > 0 }
        print(
            "PRINCIPAL".padding(toLength: 28, withPad: " ", startingAt: 0)
                + r("REQS", 8) + r("PROMPT", 12) + r("COMPL", 12)
                + r("TOTAL", 12)
                + (budgeted ? r("PERIOD", 12) + r("BUDGET", 12) : ""))
        for e in entries {
            var line =
                e.principal.padding(
                    toLength: 28, withPad: " ", startingAt: 0)
                + r("\(e.requests)", 8)
                + r("\(e.prompt_tokens)", 12)
                + r("\(e.completion_tokens)", 12)
                + r("\(e.total_tokens)", 12)
            if budgeted {
                line +=
                    r(e.period_tokens.map(String.init) ?? "-", 12)
                    + r(e.budget.map(String.init) ?? "unlimited", 12)
            }
            print(line)
        }
        // One reset line for the table: the period is daemon-wide.
        if let reset = entries.compactMap(\.period_reset).first {
            print("period resets \(reset)")
        }
    }
}

// MARK: - Portable command struct (remote-only)

/// `athena usage` on the portable client — always remote. On macOS the
/// same name is the local-overloading verb in Sources/athena/Commands.
public struct UsageCmd: AsyncParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "usage",
        abstract: "Show per-principal token usage from the daemon.")
    @OptionGroup public var daemon: DaemonOptions
    public init() {}
    public func run() async throws {
        try await RemoteUsage.show(daemon)
    }
}
