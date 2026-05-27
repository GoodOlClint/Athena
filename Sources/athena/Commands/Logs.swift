import ArgumentParser
import AthenaClient
import AthenaCore
import Foundation

/// `athena logs` — operator-friendly wrapper over `/usr/bin/log show`
/// and `/usr/bin/log stream`. Athena's diagnostic surface is the macOS
/// unified log (subsystem "athena"); this command pre-bakes the
/// canonical predicate so operators don't have to memorize the
/// query DSL. The full cheatsheet (raw `log show` recipes, off-box
/// shipping configs, persistence gotcha) lives in `docs/logging.md`.
///
/// M45.4: replaced the pre-M45 file-tail surface (which read
/// `athena.out.log` / `athena.err.log` / `athena.log`). M45.1
/// dropped those file sinks; `athena.err.log` survives only as a
/// crash-dump capture, not a diagnostic log.
struct Logs: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "logs",
        abstract:
            "Stream / show the unified-log entries Athena emitted.")

    @Flag(name: .shortAndLong, help: "Follow new entries as they arrive (`log stream`).")
    var follow = false
    @Option(
        help:
            "How far back to look (`log show --last`). E.g. 5m, 1h, 1d. Default 1h."
    )
    var since: String = "1h"
    @Option(
        name: .customLong("category"),
        parsing: .singleValue,
        help:
            "Filter by category (repeatable). E.g. daemon, audit, model.llm, model.diarization. Omitted ⇒ all categories under subsystem athena."
    )
    var categories: [String] = []
    @Flag(
        help:
            "Include info+debug entries (memory-only by default — only persists if `sudo log config --mode \"level:debug\" --subsystem athena` was set)."
    )
    var debug = false
    @Option(
        help: "Output style: compact (default) | syslog | json | ndjson.")
    var style: String = "compact"

    func run() async throws {
        let logBin = "/usr/bin/log"
        guard FileManager.default.isExecutableFile(atPath: logBin) else {
            FailableExit.die(
                "error: \(logBin) not found — `athena logs` requires "
                    + "macOS's unified-log query tool")
        }

        var predicate = "subsystem == \"athena\""
        if !categories.isEmpty {
            // `category IN { "a", "b" }` predicate dialect.
            let list = categories
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
        // Inherit stdio so the operator sees output live and can
        // Ctrl-C `log stream` cleanly.
        try p.run()
        p.waitUntilExit()
        if p.terminationStatus != 0 {
            throw ExitCode(p.terminationStatus)
        }
    }
}
