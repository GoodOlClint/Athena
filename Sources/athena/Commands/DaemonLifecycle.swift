import ArgumentParser
import AthenaCore
import Darwin
import Foundation

// `athena start | stop | status` — control the daemon PROCESS (M9.4).
// Distinct from `unload` (frees the model, daemon keeps running) and
// `load`/`serve` (runs the daemon in the foreground). The daemon
// lazy-loads the LLM on first inference, so `start` brings the surface
// up *without* serving a model until one is requested.
//
// This is the manual/dev lifecycle; boot-time supervision is still
// `athena install` (launchd). A launchd-managed daemon has no pidfile
// here — `stop` says so rather than guessing.

/// Runtime dir holding the pidfile + log. Mirrors the store default
/// (`~/.athena`); `--data-dir` overrides both, matching `load`.
private func runtimeDir(_ dataDir: String?) -> URL {
    dataDir.map {
        URL(
            fileURLWithPath: ($0 as NSString).expandingTildeInPath,
            isDirectory: true)
    }
        ?? FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".athena", isDirectory: true)
}

private func pidFile(_ dataDir: String?) -> URL {
    runtimeDir(dataDir).appendingPathComponent("athena.pid")
}

/// True if `pid` is a live process we can see.
private func alive(_ pid: Int32) -> Bool {
    // kill(_, 0) probes existence: 0 ⇒ alive; EPERM ⇒ alive but not
    // ours; ESRCH ⇒ gone.
    kill(pid, 0) == 0 || errno == EPERM
}

/// The running daemon's pid from the pidfile, if it is still alive.
/// Cleans a stale pidfile as a side effect.
private func livePid(_ dataDir: String?) -> Int32? {
    let f = pidFile(dataDir)
    guard
        let s = try? String(contentsOf: f, encoding: .utf8),
        let pid = Int32(s.trimmingCharacters(in: .whitespacesAndNewlines))
    else { return nil }
    if alive(pid) { return pid }
    try? FileManager.default.removeItem(at: f)
    return nil
}

/// This binary's absolute path, for re-exec as the daemon.
private func selfExecutable() -> String {
    if let p = Bundle.main.executablePath { return p }
    let arg0 = CommandLine.arguments[0]
    if arg0.hasPrefix("/") { return arg0 }
    return FileManager.default.currentDirectoryPath + "/" + arg0
}

struct Start: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "start",
        abstract: "Start the daemon process in the background.")
    @Option(help: "Listen host.") var host: String = "127.0.0.1"
    @Option(help: "Listen port.")
    var port: Int = GovernorConfig.defaultPort
    @Option(help: "LLM engine: mlx or stub.") var engine: String?
    @Option(help: "Model dir or store name.") var model: String?
    @Option(help: "Runtime/data dir (default ~/.athena).")
    var dataDir: String?
    @Option(help: "Log level (trace|debug|info|…; default info).")
    var logLevel: String?
    @Option(help: "Opt-in remote syslog udp://host[:514] (logs only).")
    var syslogRemote: String?

    func run() async throws {
        if let pid = livePid(dataDir) {
            FailableExit.die(
                "error: daemon already running (pid \(pid))")
        }
        let dir = runtimeDir(dataDir)
        try FileManager.default.createDirectory(
            at: dir, withIntermediateDirectories: true)
        let logURL = dir.appendingPathComponent("athena.log")
        if !FileManager.default.fileExists(atPath: logURL.path) {
            FileManager.default.createFile(
                atPath: logURL.path, contents: nil)
        }
        guard let logFH = try? FileHandle(forWritingTo: logURL) else {
            FailableExit.die(
                "error: cannot open log \(logURL.path)")
        }
        logFH.seekToEndOfFile()

        var args = ["load", "--host", host, "--port", String(port)]
        if let engine { args += ["--engine", engine] }
        if let model { args += ["--model", model] }
        if let dataDir { args += ["--data-dir", dataDir] }
        if let logLevel { args += ["--log-level", logLevel] }
        if let syslogRemote {
            args += ["--syslog-remote", syslogRemote]
        }

        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: selfExecutable())
        proc.arguments = args
        proc.standardOutput = logFH
        proc.standardError = logFH
        proc.standardInput = FileHandle.nullDevice
        do {
            try proc.run()
        } catch {
            FailableExit.die("error: spawn failed: \(error)")
        }
        let pid = proc.processIdentifier
        try? Data("\(pid)\n".utf8).write(to: pidFile(dataDir))
        print(
            "started athena daemon (pid \(pid)) on "
                + "\(host):\(port) — log: \(logURL.path)")
    }
}

struct Stop: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "stop",
        abstract: "Stop the daemon process.")
    @Option(help: "Runtime/data dir (default ~/.athena).")
    var dataDir: String?

    func run() async throws {
        guard let pid = livePid(dataDir) else {
            FailableExit.die(
                "error: no daemon pidfile at "
                    + "\(pidFile(dataDir).path) — if a daemon is "
                    + "running it is launchd-managed; use "
                    + "`launchctl` / `athena install`")
        }
        kill(pid, SIGTERM)
        // Up to ~5s for a graceful exit, then SIGKILL.
        for _ in 0..<50 {
            if !alive(pid) { break }
            try? await Task.sleep(nanoseconds: 100_000_000)
        }
        if alive(pid) {
            kill(pid, SIGKILL)
            try? await Task.sleep(nanoseconds: 200_000_000)
        }
        try? FileManager.default.removeItem(at: pidFile(dataDir))
        print("stopped athena daemon (pid \(pid))")
    }
}

struct Status: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "status",
        abstract: "Report daemon process + governor state.")
    @OptionGroup var daemon: DaemonOptions
    @Option(help: "Runtime/data dir (default ~/.athena).")
    var dataDir: String?

    func run() async throws {
        let pid = livePid(dataDir)
        if let pid {
            print("daemon: running (pid \(pid))")
        } else {
            print("daemon: no managed pidfile")
        }
        var req = URLRequest(
            url: URL(string: daemon.base + "/healthz")!)
        req.timeoutInterval = 2
        guard
            let (data, _) = try? await URLSession.shared.data(for: req)
        else {
            print("endpoint: unreachable at \(daemon.host):\(daemon.port)")
            if pid == nil { throw ExitCode.failure }
            return
        }
        print("endpoint: \(daemon.host):\(daemon.port) (healthy)")
        HTTPClient.printJSON(data)
    }
}
