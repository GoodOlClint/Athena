import ArgumentParser
import AthenaClient
import AthenaCore
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
        case err, out, start, auto
    }

    @Option(help: "Which log: auto (default) | err | out (launchd) | start.")
    var source: Source = .auto
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
        case .auto:
            // M43.4 #6 — pick the first log that actually exists so an
            // operator who started the daemon via `athena start`
            // (user-context, writes to <data-dir>/athena.log) doesn't
            // hit `is the daemon installed?` because we hard-coded the
            // launchd-managed `.err` path. Order: launchd err → launchd
            // out → user-context start. Dead end names every candidate.
            url = try Self.firstExistingLogURL(
                config: config, dataDir: dataDir)
        case .start:
            let dir = Self.startLogDir(dataDir: dataDir)
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

    private static func startLogDir(dataDir: String?) -> URL {
        dataDir.map {
            URL(
                fileURLWithPath:
                    ($0 as NSString).expandingTildeInPath,
                isDirectory: true)
        } ?? AthenaEnv.userHome()
            .appendingPathComponent(".athena", isDirectory: true)
    }

    /// M43.4 #6 — auto-detect routing: try the launchd-managed paths
    /// first (every installed daemon has `log_dir` in its TOML), then
    /// the user-context `<data-dir>/athena.log` from `athena start`.
    /// Returns the URL of the first one that exists on disk; throws a
    /// dead-end error that NAMES every candidate so the operator
    /// doesn't have to guess.
    private static func firstExistingLogURL(
        config: String?, dataDir: String?
    ) throws -> URL {
        var candidates: [URL] = []
        let cfgURL = ConfigEditor.resolvePath(config)
        if let cfg = try? AthenaConfig.parse(file: cfgURL) {
            let logDir = URL(
                fileURLWithPath: cfg.logDir, isDirectory: true)
            candidates.append(
                logDir.appendingPathComponent("athena.err.log"))
            candidates.append(
                logDir.appendingPathComponent("athena.out.log"))
        }
        let startURL =
            startLogDir(dataDir: dataDir)
            .appendingPathComponent("athena.log")
        candidates.append(startURL)
        if let live = candidates.first(where: {
            FileManager.default.fileExists(atPath: $0.path)
        }) {
            return live
        }
        let list = candidates.map { "    " + $0.path }
            .joined(separator: "\n")
        FailableExit.die(
            "error: no athena log found at any known path. Tried:\n"
                + list + "\n  Is the daemon running? "
                + "`athena status` reports its lifecycle.")
    }
}
