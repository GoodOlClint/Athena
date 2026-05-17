import ArgumentParser
import AthenaDeploy
import Foundation

/// `athena logs` — tail (optionally follow) the daemon's log. Two
/// sources: a launchd-installed daemon writes `<log_dir>/athena.{err,
/// out}.log`; an `athena start`-managed daemon writes
/// `<data-dir|~/.athena>/athena.log`. M9.6b.
struct Logs: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "logs",
        abstract: "Tail (or follow) the daemon log.")

    enum Source: String, ExpressibleByArgument, CaseIterable {
        case err, out, start
    }

    @Option(help: "Which log: err | out (launchd) | start.")
    var source: Source = .err
    @Option(
        name: [.customShort("n"), .long],
        help: "Lines to show (default 50).")
    var lines: Int = 50
    @Flag(name: .shortAndLong, help: "Follow appended output.")
    var follow = false
    @Option(help: "Config file (overrides auto-resolution).")
    var config: String?
    @Option(help: "Runtime/data dir for `start` logs (default ~/.athena).")
    var dataDir: String?

    func run() async throws {
        let url: URL
        switch source {
        case .start:
            let dir =
                dataDir.map {
                    URL(
                        fileURLWithPath:
                            ($0 as NSString).expandingTildeInPath,
                        isDirectory: true)
                }
                ?? FileManager.default.homeDirectoryForCurrentUser
                    .appendingPathComponent(
                        ".athena", isDirectory: true)
            url = dir.appendingPathComponent("athena.log")
        case .err, .out:
            let cfgURL = ConfigEditor.resolvePath(config)
            guard
                let cfg = try? AthenaConfig.parse(file: cfgURL)
            else {
                FailableExit.die(
                    "error: cannot read config at \(cfgURL.path) "
                        + "(need log_dir for \(source.rawValue) logs)")
            }
            url = URL(fileURLWithPath: cfg.logDir, isDirectory: true)
                .appendingPathComponent("athena.\(source.rawValue).log")
        }

        guard FileManager.default.fileExists(atPath: url.path) else {
            FailableExit.die(
                "error: no log at \(url.path) — is the daemon "
                    + "\(source == .start ? "started" : "installed")?")
        }

        // Delegate to `tail` (battle-tested follow/rotation). `-F`
        // follows by name so a rotated log keeps streaming.
        let p = Process()
        p.executableURL = URL(fileURLWithPath: "/usr/bin/tail")
        p.arguments =
            ["-n", String(max(0, lines))]
            + (follow ? ["-F"] : []) + [url.path]
        try p.run()
        p.waitUntilExit()
        if p.terminationStatus != 0 {
            throw ExitCode(p.terminationStatus)
        }
    }
}
