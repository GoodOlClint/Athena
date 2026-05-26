import ArgumentParser
import AthenaClient
import AthenaCore
import AthenaDeploy
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
        ?? AthenaEnv.userHome()
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
    @Option(help: "Bearer-auth keys file (see `athena load --help`).")
    var authKeysFile: String?

    @Option(help: "launchd label for a system (boot) daemon.")
    var label: String = "me.goodolclint.athena"

    func run() async throws {
        // Root + an installed plist ⇒ delegate to launchctl so the
        // daemon runs as the plist's `UserName` (the service user the
        // installer recorded), not root. The pre-fix `Process()` spawn
        // inherited the sudo'ing caller's euid, leaving the daemon as
        // root — which under macOS hardened-runtime spawn semantics
        // broke Swift's `Bundle.main.bundleURL` resolution and made
        // mlx-c print "Failed to load the default metallib library
        // not found …" four times on first MLX init. The same binary
        // worked foreground because the shell launched it as a normal
        // user with argv[0] = the full executable path. `athena stop`
        // already symmetrically goes through launchctl bootout; this
        // closes the loop on start.
        if geteuid() == 0, isValidLabel(label) {
            let plist = InstallPlan.plistPath(label: label)
            if FileManager.default.fileExists(atPath: plist.path) {
                // bootout is non-fatal (returns 3 when the service
                // isn't loaded); bootstrap is the actual load.
                _ = launchctl(["bootout", "system", plist.path])
                let rc = launchctl(
                    ["bootstrap", "system", plist.path])
                if rc != 0 {
                    FailableExit.die(
                        "error: launchctl bootstrap "
                            + "\(plist.path) failed (status \(rc))")
                }
                print(
                    "started system athena daemon via launchd "
                        + "(\(label)) — logs in "
                        + "/usr/local/var/log/athena/")
                return
            }
            // Root, but no plist → fall through to Process() spawn
            // with a warning so the operator notices the gap.
            FileHandle.standardError.write(
                Data(
                    ("warning: running as root but no installed "
                        + "launchd plist at \(plist.path). Falling "
                        + "back to a root-owned Process() daemon — "
                        + "`athena install` first to get a "
                        + "user-owned launchd service.\n").utf8))
        }
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

        var flags = ["--host", host, "--port", String(port)]
        if let engine { flags += ["--engine", engine] }
        if let model { flags += ["--model", model] }
        if let dataDir { flags += ["--data-dir", dataDir] }
        if let logLevel { flags += ["--log-level", logLevel] }
        if let syslogRemote {
            flags += ["--syslog-remote", syslogRemote]
        }
        if let authKeysFile {
            flags += ["--auth-keys-file", authKeysFile]
        }

        // Spawn `athena load …` directly. Was: spawn the sibling
        // `athenad` (M14.2d) which `execv`-ed `athena` with
        // `argv[0]="athena"` (bare). That argv[0] / kernel-binary-path
        // discrepancy made Swift's `Bundle.main.bundleURL` resolve to
        // the wrong directory under macOS hardened-runtime spawn
        // semantics, so mlx-c's SwiftPM-bundle lookup couldn't find
        // `mlx-swift_Cmlx.bundle` and `MLX.Memory.memoryLimit = …`
        // crashed the daemon with "Failed to load the default metallib
        // library not found …" on first MLX init. Spawning the
        // self-binary keeps argv[0] aligned with the resolved path —
        // same fix as the M42 launchd plist.
        let selfPath = selfExecutable()
        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: selfPath)
        proc.arguments = ["load"] + flags
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

/// A launchd label is reverse-DNS-ish: ASCII letters/digits and the
/// separators `.`, `-`, `_`. We pass it to `launchctl` as argv (never a
/// shell string), but validate defensively so a malformed `--label`
/// fails fast rather than reaching launchctl.
private func isValidLabel(_ s: String) -> Bool {
    guard !s.isEmpty, s.count <= 255 else { return false }
    let allowed = Set(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-_")
    return s.allSatisfy { allowed.contains($0) }
}

/// Run `/bin/launchctl` with `args` as argv (no shell), returning its
/// exit status. Output is discarded — callers only need the status.
private func launchctl(_ args: [String]) -> Int32 {
    let p = Process()
    p.executableURL = URL(fileURLWithPath: "/bin/launchctl")
    p.arguments = args
    p.standardOutput = FileHandle.nullDevice
    p.standardError = FileHandle.nullDevice
    do { try p.run() } catch { return -1 }
    p.waitUntilExit()
    return p.terminationStatus
}

struct Stop: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "stop",
        abstract:
            "Stop the daemon process (user pidfile, else system launchd)."
    )
    @Option(help: "Runtime/data dir (default ~/.athena).")
    var dataDir: String?
    @Option(help: "launchd label for a system (boot) daemon.")
    var label: String = "me.goodolclint.athena"

    func run() async throws {
        // 1. A user-context daemon (from `athena start`) owns a pidfile;
        //    stop it directly. This path needs no root and no install.
        if let pid = livePid(dataDir) {
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
            return
        }

        // 2. No user pidfile → maybe a system (launchd-managed) daemon.
        guard isValidLabel(label) else {
            FailableExit.die(
                "error: invalid --label '\(label)' (expected reverse-DNS: "
                    + "letters, digits, '.', '-', '_')")
        }
        let plist = InstallPlan.plistPath(label: label)
        let systemPresent =
            FileManager.default.fileExists(atPath: plist.path)
            || launchctl(["print", "system/\(label)"]) == 0
        guard systemPresent else {
            FailableExit.die(
                "error: no athena daemon to stop — no user pidfile at "
                    + "\(pidFile(dataDir).path) and no system LaunchDaemon "
                    + "'\(label)'. Start one with `athena start`, or "
                    + "install the boot daemon with `athena install`.")
        }

        // 3. A system daemon exists. Removing it needs root; we surface
        //    that and never self-escalate (no spawning sudo ourselves).
        guard geteuid() == 0 else {
            FailableExit.die(
                "error: '\(label)' is a system LaunchDaemon — stopping it "
                    + "needs root. Re-run: sudo athena stop --label \(label)")
        }
        let status = launchctl(["bootout", "system/\(label)"])
        guard status == 0 else {
            FailableExit.die(
                "error: launchctl bootout system/\(label) failed "
                    + "(status \(status))")
        }
        print("stopped system athena daemon (\(label))")
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
        // Off-box: no local pidfile to read — the native admin
        // posture (/api/admin/status) is the remote equivalent.
        if daemon.isRemote {
            try await RemoteAdmin.status(daemon)
            return
        }
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
