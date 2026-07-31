import ArgumentParser
import AthenaClient
import AthenaCore
import Foundation

/// `athena logs` — operator surface for the daemon's unified-log
/// entries. M45.5 promoted this from a thin /usr/bin/log shell-out
/// to a `/api/logs` (one-shot) / `/api/logs/stream` (SSE) client so
/// it works against a REMOTE daemon and respects RBAC
/// (`.daemonAdmin`). Off-box operators get the same surface as the
/// local box.
///
/// `--offline` keeps the direct `/usr/bin/log` shell-out path for the
/// triage scenario where the daemon is DOWN and you need to see
/// what crashed. The offline path runs locally only and bypasses
/// the API permission gate.
struct Logs: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "logs",
        abstract:
            "Show / stream the daemon's unified-log entries (via /api/logs by default; --offline for direct local query)."
    )

    @OptionGroup var daemon: DaemonOptions
    @Flag(name: .shortAndLong, help: "Follow new entries as they arrive (SSE).")
    var follow = false
    @Option(
        help:
            "How far back to look (default 1h). E.g. 5m, 1h, 1d. Ignored under --follow."
    )
    var since: String = "1h"
    @Option(
        name: .customLong("category"),
        parsing: .singleValue,
        help:
            "Filter by category (repeatable). E.g. daemon, audit, model.llm, model.diarization."
    )
    var categories: [String] = []
    @Flag(
        help:
            "Include info+debug entries (memory-only by default — only persists if `sudo log config --mode \"level:debug\" --subsystem athena` was set)."
    )
    var debug = false
    @Option(help: "Max entries returned by /api/logs (default 200, capped 5000).")
    var limit: Int?
    @Flag(
        help:
            "Bypass the daemon and shell out to /usr/bin/log directly. For triage when the daemon is down — no RBAC, local box only."
    )
    var offline = false
    @Option(
        help: "Output style for --offline: compact (default) | syslog | json | ndjson.")
    var style: String = "compact"

    func run() async throws {
        if offline {
            try await runOffline()
            return
        }
        // Default path: hit the daemon's /api/logs endpoint (works
        // against a local OR remote daemon, depending on
        // DaemonOptions). RBAC-gated to daemon.admin server-side;
        // the client just relays.
        if follow {
            try await RemoteLogs.stream(
                daemon, categories: categories, debug: debug)
        } else {
            try await RemoteLogs.show(
                daemon, since: since, categories: categories,
                debug: debug, limit: limit)
        }
    }

    /// `--offline` escape hatch: pre-M45.5 behavior. Wraps
    /// `/usr/bin/log show / stream` directly on the LOCAL box, no
    /// daemon involved. Useful when the daemon is down and you need
    /// to read the crash trail; pointless against a remote daemon
    /// because we'd be reading the operator's local unified log,
    /// not the daemon's.
    private func runOffline() async throws {
        let logBin = "/usr/bin/log"
        guard FileManager.default.isExecutableFile(atPath: logBin) else {
            FailableExit.die(
                "error: \(logBin) not found — --offline requires "
                    + "macOS's unified-log query tool")
        }
        var predicate = "subsystem == \"athena\""
        if !categories.isEmpty {
            let list =
                categories
                .map { "\"\($0)\"" }
                .joined(separator: ", ")
            predicate += " AND category IN { \(list) }"
        }
        var args: [String] = []
        if follow {
            args.append("stream")
        } else {
            args.append("show")
            args.append("--last")
            args.append(since)
        }
        args.append("--predicate")
        args.append(predicate)
        args.append("--style")
        args.append(style)
        if debug {
            args.append("--info")
            args.append("--debug")
        }
        let p = Process()
        p.executableURL = URL(fileURLWithPath: logBin)
        p.arguments = args
        try p.run()
        p.waitUntilExit()
        if p.terminationStatus != 0 {
            throw ExitCode(p.terminationStatus)
        }
    }
}
