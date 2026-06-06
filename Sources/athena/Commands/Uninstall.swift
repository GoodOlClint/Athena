import ArgumentParser
import AthenaDeploy
import Darwin
import Foundation

/// `athena uninstall` — reverse `athena install`. Stops + unloads the
/// launchd daemon, removes the plist, the `bin/athena` symlink, and the
/// `libexec/athena` payload. Config, working dir, and logs are KEPT by
/// default (they hold user data); `--purge` removes those too. M9.6c.
struct Uninstall: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "uninstall",
        abstract: "Remove the launchd daemon install (reverse install).")

    @Option(help: "Install prefix.")
    var prefix: String = "/usr/local"
    @Option(help: "launchd label.")
    var label: String = "me.goodolclint.athena"
    @Option(help: "Config path (for log_dir under --purge).")
    var config: String?
    @Flag(help: "Also remove config, working dir, and logs.")
    var purge = false
    @Flag(help: "Print the plan; change nothing.")
    var dryRun = false

    func run() throws {
        let plan = InstallPlan(
            sourceDir: URL(fileURLWithPath: "/"),
            prefix: URL(fileURLWithPath: prefix, isDirectory: true),
            label: label)
        let fm = FileManager.default

        // Resolve log_dir (only needed for --purge) from the installed
        // config, or an explicit --config.
        var logDir: URL?
        if purge {
            let cfgURL =
                config.map {
                    URL(
                        fileURLWithPath:
                            ($0 as NSString).expandingTildeInPath)
                } ?? plan.installedConfig
            if let cfg = try? AthenaConfig.parse(file: cfgURL) {
                logDir = URL(
                    fileURLWithPath: cfg.logDir, isDirectory: true)
            }
        }

        var removals: [URL] = [
            plan.plistPath, plan.binLauncher, plan.libexecDir,
        ]
        if purge {
            removals += [plan.configDir, plan.workingDir]
            if let logDir { removals.append(logDir) }
        }

        if dryRun {
            print("athena uninstall (dry-run)")
            print("  bootout: system \(plan.plistPath.path)")
            for u in removals { print("  rm:      \(u.path)") }
            if !purge {
                print(
                    "  keep:    \(plan.configDir.path), "
                        + "\(plan.workingDir.path) (use --purge to "
                        + "remove)")
            }
            return
        }

        guard geteuid() == 0 else {
            throw ValidationError(
                "must run as root (sudo athena uninstall …)")
        }

        // Stop + unload the daemon (ignore failure: may not be loaded).
        _ = Self.launchctl(["bootout", "system", plan.plistPath.path])

        var removed = 0
        for u in removals {
            // lstat semantics: present even if it's a dangling symlink
            // (a dead bin/athena link must still be removed).
            guard
                (try? fm.attributesOfItem(atPath: u.path)) != nil
            else { continue }
            do {
                try fm.removeItem(at: u)
                removed += 1
                print("removed \(u.path)")
            } catch {
                FileHandle.standardError.write(
                    Data("error: \(u.path): \(error)\n".utf8))
            }
        }
        print(
            "uninstalled \(label) (\(removed) item(s) removed"
                + (purge ? ", purged" : ", config/data kept") + ").")
    }

    @discardableResult
    private static func launchctl(_ args: [String]) -> Int32 {
        let p = Process()
        p.executableURL = URL(fileURLWithPath: "/bin/launchctl")
        p.arguments = args
        p.standardError = FileHandle.nullDevice
        try? p.run()
        p.waitUntilExit()
        return p.terminationStatus
    }
}
