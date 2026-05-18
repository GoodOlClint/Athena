import ArgumentParser
import AthenaCore
import AthenaDeploy
import AthenaStore
import Crypto
import Darwin
import Foundation

/// `athena install` — install this binary as the boot-time launchd daemon.
/// Collapses the old deploy/athena-install.sh. The one thing it cannot do
/// is *build* (MLX's Metal shaders need xcodebuild — a binary can't build
/// itself); run `deploy/build.sh` first, then this from the build output.
struct Install: AsyncParsableCommand {
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

    func run() async throws {
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
            executablePath: plan.installedDaemon.path,
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

        // The service user's home — NOT root's. The daemon runs as
        // this user and resolves `~/.athena[/models]` against it; the
        // installer must create those exact paths so the daemon can
        // write them.
        let svcHome =
            AthenaEnv.homeDirectory(ofUser: serviceUser)
            ?? URL(
                fileURLWithPath: "/Users/\(serviceUser)",
                isDirectory: true)
        func resolve(_ p: String?, default def: URL) -> URL {
            guard let p, !p.isEmpty else { return def }
            if p.hasPrefix("~") {
                return svcHome.appendingPathComponent(
                    String(p.dropFirst()).drop(while: { $0 == "/" })
                        .description, isDirectory: true)
            }
            return URL(fileURLWithPath: p, isDirectory: true)
        }
        let dataDir = resolve(
            cfg.dataDir,
            default: svcHome.appendingPathComponent(
                ".athena", isDirectory: true))
        let modelStore = resolve(
            cfg.modelStore,
            default: svcHome.appendingPathComponent(
                ".athena/models", isDirectory: true))
        let logURL = URL(
            fileURLWithPath: cfg.logDir, isDirectory: true)

        // Create everything. Service-writable dirs are chowned to the
        // service user; every dir is 0755 so the service user can
        // TRAVERSE the path (the launchd EX_CONFIG bug was
        // /usr/local/var/log at mode 744 — unreadable by the
        // non-root service user, so launchd couldn't open the log).
        func ensureDir(_ url: URL, owner: String?) throws {
            try fm.createDirectory(
                at: url, withIntermediateDirectories: true)
            try? fm.setAttributes(
                [.posixPermissions: 0o755],
                ofItemAtPath: url.path)
            if let owner {
                try? fm.setAttributes(
                    [.ownerAccountName: owner],
                    ofItemAtPath: url.path)
            }
        }
        /// Make every ancestor of `leaf` that lies under `prefix`
        /// traversable (0755) — fixes umask-restricted intermediates
        /// like /usr/local/var/log created during this install.
        func makeTraversable(toward leaf: URL) {
            let base = plan.prefix.standardizedFileURL.path
            var cur = leaf.standardizedFileURL
            while cur.path.hasPrefix(base), cur.path != base {
                if fm.fileExists(atPath: cur.path) {
                    try? fm.setAttributes(
                        [.posixPermissions: 0o755],
                        ofItemAtPath: cur.path)
                }
                cur = cur.deletingLastPathComponent()
            }
        }

        try ensureDir(plan.libexecDir, owner: nil)
        try ensureDir(plan.configDir, owner: nil)
        try ensureDir(plan.workingDir, owner: serviceUser)
        try ensureDir(logURL, owner: serviceUser)
        makeTraversable(toward: logURL)
        try ensureDir(dataDir, owner: serviceUser)
        try ensureDir(modelStore, owner: serviceUser)

        // Seed a default `admin` account (with the `admin` role) and
        // one admin bearer token on a fresh store, so the appliance
        // is administrable immediately. Idempotent: never touch an
        // existing install (no password reset / extra token on
        // reinstall — gated on userCount==0).
        let dbURL = dataDir.appendingPathComponent("athena.sqlite")
        var seededPassword: String?
        var seededToken: String?
        if let db = try? AthenaStore(path: dbURL) {
            if await db.userCount() == 0 {
                let pw = Self.simplePassword()
                let salt = Passwords.randomSalt()
                let hash = Passwords.derive(
                    password: pw, salt: salt,
                    iters: Passwords.defaultIterations)
                try? await db.putUser(
                    username: "admin", salt: salt, hash: hash,
                    iters: Passwords.defaultIterations)
                try? await db.grantRole(
                    username: "admin", role: "admin")
                seededPassword = pw
                let rawKey = SymmetricKey(size: .bits256)
                    .withUnsafeBytes { Data($0) }
                let key =
                    "sk-athena-"
                    + rawKey.base64EncodedString()
                    .replacingOccurrences(of: "+", with: "-")
                    .replacingOccurrences(of: "/", with: "_")
                    .replacingOccurrences(of: "=", with: "")
                try? await db.putToken(
                    hash: Data(AuthConfig.sha(key)),
                    username: "admin", scopedRoles: nil,
                    label: "install-seed")
                seededToken = key
            }
        }
        // The daemon runs as the service user and must own the DB
        // (install runs as root, so it would otherwise be root-owned
        // and unwritable by the daemon).
        for ext in ["", "-wal", "-shm"] {
            let p = dbURL.path + ext
            if fm.fileExists(atPath: p) {
                try? fm.setAttributes(
                    [.ownerAccountName: serviceUser],
                    ofItemAtPath: p)
            }
        }

        for name in plan.artifactNames(fileManager: fm) {
            let src = plan.sourceDir.appendingPathComponent(name)
            let dst = plan.libexecDir.appendingPathComponent(name)
            guard fm.fileExists(atPath: src.path) else {
                throw ValidationError(
                    "build artifact '\(name)' not found in "
                        + "\(plan.sourceDir.path). Build the whole "
                        + "package (deploy/build.sh — it uses the "
                        + "`athena-Package` scheme, which also "
                        + "produces `athenad`), then install from "
                        + "that output.")
            }
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

        // (dirs already created + chowned + made traversable above.)

        _ = Self.launchctl(["bootout", "system", plan.plistPath.path])
        try Self.launchctlChecked(["bootstrap", "system", plan.plistPath.path])
        _ = Self.launchctl(["enable", "system/\(label)"])
        _ = Self.launchctl(["kickstart", "-k", "system/\(label)"])

        print("installed \(label) (service user: \(serviceUser)).")
        print("  model store: \(modelStore.path)")
        print("  data dir:    \(dataDir.path)")
        if let pw = seededPassword {
            print("")
            print("  ┌──────────────────────────────────────────┐")
            print("  │ admin account created (role: admin)       │")
            print("  │   username: admin                         │")
            print("  │   password: \(pw.padding(toLength: 30, withPad: " ", startingAt: 0))│")
            print("  │ SAVE THIS — shown once. Change it via     │")
            print("  │ `athena auth user passwd admin`.          │")
            print("  └──────────────────────────────────────────┘")
            if let tok = seededToken {
                print("")
                print("  admin bearer token (SAVE NOW — shown once):")
                print("    \(tok)")
                print(
                    "    use:  Authorization: Bearer <token>")
                print(
                    "    mint more: `athena auth token add --user "
                        + "<u>`")
            }
            print("")
        } else {
            print(
                "  auth: existing accounts kept (no admin seeded)")
            print(
                "  locked out? reset offline (no token needed):")
            print(
                "    \(plan.binSymlink.path) auth user passwd admin "
                    + "--data-dir \(dataDir.path)")
        }
        print(
            "  health: curl -s http://\(cfg.listenHost):\(cfg.listenPort)/healthz")
        print("  logs:   tail -f \(cfg.logDir)/athena.err.log")
        print(
            "  note: the daemon serves /healthz + /ui even with no "
                + "model loaded; pull/convert a model for inference.")
    }

    /// A human-typeable random password (~62 bits): 12 chars from an
    /// unambiguous alphabet (no l/1/o/0). PBKDF2-hashed at rest;
    /// meant to be rotated.
    static func simplePassword() -> String {
        let alpha = Array(
            "abcdefghijkmnpqrstuvwxyz23456789ABCDEFGHJKLMNPQRSTUVWXYZ")
        var rng = SystemRandomNumberGenerator()
        return String(
            (0..<12).map { _ in alpha.randomElement(using: &rng)! })
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
