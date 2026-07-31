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
        abstract:
            "Start the daemon. Installed (root): launchd-managed; "
            + "uninstalled: foreground-attached (Ctrl-C to stop).")
    @Option(help: "Listen host.") var host: String = "127.0.0.1"
    @Option(help: "Listen port.")
    var port: Int = GovernorConfig.defaultPort
    @Option(help: "LLM engine: mlx or stub.") var engine: String?
    @Option(help: "Model dir or store name.") var model: String?
    @Option(help: "Runtime/data dir (default ~/.athena).")
    var dataDir: String?
    @Option(help: "Terminal verbosity (trace|debug|info|…; default info). Foreground only.")
    var logLevel: String?
    @Option(help: "Bearer-auth keys file (see `athena load --help`).")
    var authKeysFile: String?

    @Option(help: "launchd label for a system (boot) daemon.")
    var label: String = "me.goodolclint.athena"

    func run() async throws {
        // NB1 (M66.3): validate the label BEFORE any euid branching. The
        // old `if geteuid() == 0, isValidLabel(label)` skipped the entire
        // root block — including the M43.1 hard-fail below — on a malformed
        // label, falling through to the user-context Process() spawn and
        // running the MLX daemon as ROOT (the metallib-bundle breakage
        // M43.1 exists to prevent). Fail fast up front, like Stop/Restart.
        guard isValidLabel(label) else {
            FailableExit.die(
                "error: invalid --label '\(label)' — use reverse-DNS "
                    + "form (ASCII letters/digits and . - _)")
        }
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
        if geteuid() == 0 {
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
            // M43.1 — refuse: a root-owned Process() spawn re-hits the
            // same metallib-bundle lookup bug that drove the v0.10.38–41
            // chase (mlx-c's SwiftPM-resource lookup mis-resolves under
            // hardened-runtime spawn from root). The prior stderr-only
            // warning here was invisible to scripts that
            // `2>/dev/null`, so the daemon would come up broken with
            // `athena status` reporting healthy. Hard fail with the
            // install gate is the unambiguous remedy.
            FailableExit.die(
                "error: cannot start a root-owned daemon — no installed "
                    + "launchd plist at \(plist.path). Run "
                    + "`sudo athena install` first to install the system "
                    + "LaunchDaemon, or re-run without sudo to start a "
                    + "user-context daemon.")
        }
        if let pid = livePid(dataDir) {
            FailableExit.die(
                "error: daemon already running (pid \(pid))")
        }
        let dir = runtimeDir(dataDir)
        try FileManager.default.createDirectory(
            at: dir, withIntermediateDirectories: true)

        var flags = ["--host", host, "--port", String(port)]
        if let engine { flags += ["--engine", engine] }
        if let model { flags += ["--model", model] }
        if let dataDir { flags += ["--data-dir", dataDir] }
        if let logLevel { flags += ["--log-level", logLevel] }
        if let authKeysFile {
            flags += ["--auth-keys-file", authKeysFile]
        }

        // Spawn `athena load …` directly, inheriting our stdio so the
        // child's TerminalLogHandler emits to the operator's terminal
        // (M45.1 — no `--background`, so the daemon installs the
        // TerminalLogHandler alongside the unified-log handler;
        // TerminalLogHandler writes to stderr).
        //
        // Spawn-and-return (NOT waitUntilExit): keeps the historical
        // `athena start` contract — scripts and the e2e gate expect
        // start to fork the daemon and return so the shell can
        // continue with curl tests. The child's stdio FDs survive
        // the parent's exit (the operator's terminal stays connected
        // to the daemon until they close it or run `athena stop`).
        // Pre-M45.1 the child's stdout/stderr were redirected to a
        // file under the runtime dir; M45.1 drops the file in favor
        // of terminal inheritance + the unified-log handler.
        //
        // Re. argv[0]: spawning the self-binary keeps argv[0] aligned
        // with the kernel's view, so Swift's `Bundle.main.bundleURL`
        // resolves correctly under macOS hardened-runtime spawn
        // semantics — same fix as the M42 launchd plist.
        let selfPath = selfExecutable()
        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: selfPath)
        proc.arguments = ["load"] + flags
        proc.standardInput = FileHandle.nullDevice
        // standardOutput / standardError left as `nil` (Process
        // default) so the child inherits our terminal FDs.
        do {
            try proc.run()
        } catch {
            FailableExit.die("error: spawn failed: \(error)")
        }
        let pid = proc.processIdentifier
        try? Data("\(pid)\n".utf8).write(to: pidFile(dataDir))
        print(
            "started athena daemon (pid \(pid)) on "
                + "\(host):\(port) — `athena stop` to halt; "
                + "logs flow to this terminal until close. "
                + "For a system daemon: `sudo athena install` then "
                + "`sudo athena start`.")
    }
}

// `isValidLabel` moved to AthenaDeploy (NB4 / M70.1b) so it is unit-testable
// under `swift test`; the call sites here are unchanged (DaemonLifecycle
// imports AthenaDeploy).

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
            for _ in 0 ..< 50 {
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

/// `athena restart` — bootout + bootstrap a system LaunchDaemon so the
/// updated plist (config / install) takes effect. `launchctl kickstart
/// -k` does NOT re-read the plist; operators reach for that and silently
/// keep running the old args. M43.4 fragility #7.
struct Restart: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "restart",
        abstract:
            "Re-bootstrap the system LaunchDaemon (re-reads the plist)."
    )
    @Option(help: "launchd label for a system (boot) daemon.")
    var label: String = "me.goodolclint.athena"
    @OptionGroup var daemon: DaemonOptions
    @Flag(
        name: .customLong("sudo"),
        help: "Force the root bootout/bootstrap path, skipping the daemon API.")
    var forceSudo: Bool = false

    func run() async throws {
        guard isValidLabel(label) else {
            FailableExit.die(
                "error: invalid --label '\(label)' (expected reverse-DNS: "
                    + "letters, digits, '.', '-', '_')")
        }
        // ADR 037 — prefer the daemon-mediated restart: a reachable daemon
        // drains + exit(0)s and launchd's KeepAlive relaunches it, so no sudo
        // is needed. Fall back to root bootout/bootstrap when unreachable (or
        // `--sudo`).
        if !forceSudo, await RemoteConfig.restart(daemon) {
            print(
                "restart requested via daemon at \(daemon.base) — draining, "
                    + "then launchd relaunches (~10s). Poll "
                    + "`curl \(daemon.base)/healthz`.")
            return
        }
        let plist = InstallPlan.plistPath(label: label)
        guard FileManager.default.fileExists(atPath: plist.path) else {
            FailableExit.die(
                "error: no installed LaunchDaemon plist at "
                    + "\(plist.path). Run `sudo athena install` first.")
        }
        guard geteuid() == 0 else {
            FailableExit.die(
                "error: restart needs root to bootout + bootstrap "
                    + "system/\(label). Re-run: sudo athena restart "
                    + "--label \(label)")
        }
        // bootout is non-fatal (returns 3 when the service isn't
        // loaded — same shape `athena start` accepts).
        _ = launchctl(["bootout", "system", plist.path])
        let rc = launchctl(["bootstrap", "system", plist.path])
        if rc != 0 {
            FailableExit.die(
                "error: launchctl bootstrap \(plist.path) failed "
                    + "(status \(rc))")
        }
        print(
            "restarted system athena daemon via launchd (\(label)) — "
                + "logs in /usr/local/var/log/athena/")
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
