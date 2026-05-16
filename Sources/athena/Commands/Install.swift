import ArgumentParser
import AthenaDeploy
import Darwin
import Foundation

/// `athena install` — install this binary as the boot-time launchd daemon.
/// Collapses the old deploy/athena-install.sh. The one thing it cannot do
/// is *build* (MLX's Metal shaders need xcodebuild — a binary can't build
/// itself); run `deploy/build.sh` first, then this from the build output.
struct Install: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "install",
        abstract: "Install Athena as a boot-time launchd system daemon."
    )

    @Option(help: "Path to the TOML config.")
    var config: String = "deploy/athena.toml"

    @Option(help: "Install prefix.")
    var prefix: String = "/usr/local"

    @Option(help: "launchd label.")
    var label: String = "me.goodolclint.athena"

    @Option(help: "Service account (default: $SUDO_USER or current user).")
    var user: String?

    @Option(help: "Build-output dir to install from (default: this binary's dir).")
    var from: String?

    @Flag(help: "Print the plan and rendered plist; change nothing.")
    var dryRun = false

    func run() throws {
        let cfg = try AthenaConfig.parse(
            file: URL(fileURLWithPath: config))

        let sourceDir =
            from.map { URL(fileURLWithPath: $0, isDirectory: true) }
            ?? Bundle.main.executableURL!.resolvingSymlinksInPath()
                .deletingLastPathComponent()

        let plan = InstallPlan(
            sourceDir: sourceDir,
            prefix: URL(fileURLWithPath: prefix, isDirectory: true),
            label: label)
        let serviceUser =
            user ?? ProcessInfo.processInfo.environment["SUDO_USER"]
            ?? NSUserName()
        let fm = FileManager.default

        let plistData = try LaunchdPlist.xmlData(
            label: label,
            executablePath: plan.installedBinary.path,
            user: serviceUser,
            workingDirectory: plan.workingDir.path,
            config: cfg)

        if dryRun {
            print("athena install (dry-run)")
            print("  from:    \(plan.sourceDir.path)")
            print("  libexec: \(plan.libexecDir.path)")
            print("  symlink: \(plan.binSymlink.path)")
            print("  config:  \(plan.installedConfig.path)")
            print("  plist:   \(plan.plistPath.path)")
            print("  user:    \(serviceUser)")
            print("  copy:    \(plan.artifactNames(fileManager: fm).joined(separator: ", "))")
            print("--- \(label).plist ---")
            print(String(data: plistData, encoding: .utf8) ?? "")
            return
        }

        guard geteuid() == 0 else {
            throw ValidationError(
                "must run as root (sudo athena install …)")
        }
        guard fm.fileExists(atPath: plan.sourceMetallib.path) else {
            throw ValidationError(
                "no Metal library next to this binary "
                    + "(\(plan.sourceMetallib.path)). This is a `swift "
                    + "build` binary and cannot run MLX — build with "
                    + "deploy/build.sh (xcodebuild) and install from there.")
        }
        if plan.sourceDir.standardizedFileURL == plan.libexecDir.standardizedFileURL {
            throw ValidationError(
                "refusing to install onto itself; run from the build "
                    + "output or pass --from")
        }

        for dir in [plan.libexecDir, plan.configDir, plan.workingDir,
                    URL(fileURLWithPath: cfg.logDir, isDirectory: true)] {
            try fm.createDirectory(
                at: dir, withIntermediateDirectories: true)
        }

        for name in plan.artifactNames(fileManager: fm) {
            let src = plan.sourceDir.appendingPathComponent(name)
            let dst = plan.libexecDir.appendingPathComponent(name)
            if fm.fileExists(atPath: dst.path) {
                try fm.removeItem(at: dst)
            }
            try fm.copyItem(at: src, to: dst)
        }

        try Data(contentsOf: URL(fileURLWithPath: config))
            .write(to: plan.installedConfig)

        try? fm.removeItem(at: plan.binSymlink)
        try fm.createSymbolicLink(
            at: plan.binSymlink, withDestinationURL: plan.installedBinary)

        try plistData.write(to: plan.plistPath)
        try fm.setAttributes(
            [.posixPermissions: 0o644], ofItemAtPath: plan.plistPath.path)

        // Service-owned writable dirs (logs, working dir).
        for dir in [plan.workingDir.path, cfg.logDir] {
            try? fm.setAttributes(
                [.ownerAccountName: serviceUser], ofItemAtPath: dir)
        }

        _ = Self.launchctl(["bootout", "system", plan.plistPath.path])
        try Self.launchctlChecked(["bootstrap", "system", plan.plistPath.path])
        _ = Self.launchctl(["enable", "system/\(label)"])
        _ = Self.launchctl(["kickstart", "-k", "system/\(label)"])

        print("installed \(label).")
        print(
            "  health: curl -s http://\(cfg.listenHost):\(cfg.listenPort)/healthz")
        print("  logs:   tail -f \(cfg.logDir)/athena.err.log")
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

    private static func launchctlChecked(_ args: [String]) throws {
        let status = launchctl(args)
        guard status == 0 else {
            throw ValidationError(
                "launchctl \(args.joined(separator: " ")) failed "
                    + "(status \(status))")
        }
    }
}
