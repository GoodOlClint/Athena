import ArgumentParser
import AthenaClient
import AthenaCore
import AthenaDeploy
import AthenaServerKit
import AthenaStore
import Crypto
import Darwin
import Foundation

// MARK: - Privilege-boundary helpers (M66.3)

/// Symlink-safe ownership/permission changes (B3/B7). `FileManager`'s
/// `setAttributes(…, ofItemAtPath:)` and the POSIX `chown`/`chmod` calls
/// resolve symlinks, so between the installer creating a path (as root)
/// and changing its owner/mode by PATH, a hostile service account could
/// swap that path for a symlink and redirect the chown/chmod onto an
/// arbitrary file. Opening with `O_NOFOLLOW` first and operating on the
/// resulting fd closes that TOCTOU: a symlink at the final component makes
/// `open` fail (ELOOP) instead of being followed.
enum FsOwn {
    /// Resolve a username to its (uid, gid); nil if unknown.
    static func ids(of user: String) -> (uid: uid_t, gid: gid_t)? {
        guard let pw = getpwnam(user) else { return nil }
        return (pw.pointee.pw_uid, pw.pointee.pw_gid)
    }

    /// Open `path` without following a final symlink. `O_DIRECTORY` is
    /// added for directories. Returns the fd (caller closes) or nil.
    private static func openNoFollow(_ path: String, isDir: Bool) -> Int32 {
        var flags = O_RDONLY | O_NOFOLLOW
        if isDir { flags |= O_DIRECTORY }
        return open(path, flags)
    }

    /// `fchown` the path's own inode (never a symlink target). Returns
    /// false (and leaves ownership unchanged) if the path is a symlink or
    /// can't be opened.
    @discardableResult
    static func chown(
        _ path: String, uid: uid_t, gid: gid_t, isDir: Bool
    ) -> Bool {
        let fd = openNoFollow(path, isDir: isDir)
        guard fd >= 0 else { return false }
        defer { close(fd) }
        return fchown(fd, uid, gid) == 0
    }

    /// `fchmod` the path's own inode (never a symlink target).
    @discardableResult
    static func chmod(_ path: String, mode: mode_t, isDir: Bool) -> Bool {
        let fd = openNoFollow(path, isDir: isDir)
        guard fd >= 0 else { return false }
        defer { close(fd) }
        return fchmod(fd, mode) == 0
    }
}

/// B6 (M66.3): a config-supplied `data_dir`/`model_store` that the
/// installer would `chown` to the service user must not be a shared system
/// directory — `data_dir = "/etc"` would otherwise hand `/etc` to the
/// service account. A path is rejected when it is the filesystem root or a
/// bare top-level directory (`/etc`, `/usr`, `/var`, …) or sits under a
/// known system subtree. The install prefix (e.g. `/usr/local/...`) and
/// the service user's home stay allowed; external volumes (depth ≥ 2,
/// e.g. `/Volumes/SSD/models`) are fine.
func isSafeToOwn(_ url: URL) -> Bool {
    let std = url.standardizedFileURL
    let comps = std.pathComponents.filter { $0 != "/" }
    // Root or a single top-level component (`/etc`, `/usr`, `/`).
    if comps.count <= 1 { return false }
    // Known-dangerous system subtrees (does NOT include `/usr/local`).
    let forbidden = [
        "/System", "/bin", "/sbin", "/dev", "/Network", "/cores",
        "/private/etc", "/private/var/db", "/usr/bin", "/usr/sbin",
        "/usr/lib", "/usr/libexec", "/usr/share", "/usr/include",
    ]
    let p = std.path
    for f in forbidden where p == f || p.hasPrefix(f + "/") {
        return false
    }
    return true
}

/// `athena install` — install this binary as the boot-time launchd daemon.
/// Collapses the old deploy/athena-install.sh. The one thing it cannot do
/// is *build* (MLX's Metal shaders need xcodebuild — a binary can't build
/// itself); run `deploy/build.sh` first, then this from the build output.
struct Install: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "install",
        abstract: "Install Athena as a boot-time launchd system daemon."
    )

    /// B17 (M66.2): lifetime (days) of the bootstrap admin token seeded at
    /// install. A quarter-year rotation window — long enough for setup,
    /// short of "forever". Recoverable via the seeded admin password.
    static let installSeedTokenTTLDays = 90

    @Option(
        help:
            "Path to the TOML config. Omit to use ./deploy/athena.toml if present, else synthesize a documented default."
    )
    var config: String?

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
        // Resolve the config. Explicit --config is unchanged: parse it,
        // throwing if absent. With no --config, fall back to the in-repo
        // dev copy if present, else synthesize a documented default from
        // the built-in per-key defaults so a bare appliance installs
        // with no config authoring. `configData` is what gets written to
        // the install dir, so the operator always has a file to edit.
        let cfg: AthenaConfig
        let configData: Data
        let synthesized: Bool
        if let config {
            let url = URL(fileURLWithPath: config)
            cfg = try AthenaConfig.parse(file: url)
            configData = try Data(contentsOf: url)
            synthesized = false
        } else {
            let dev = URL(fileURLWithPath: "deploy/athena.toml")
            if FileManager.default.fileExists(atPath: dev.path) {
                cfg = try AthenaConfig.parse(file: dev)
                configData = try Data(contentsOf: dev)
                synthesized = false
            } else {
                let logDir = URL(fileURLWithPath: prefix)
                    .appendingPathComponent("var/log/athena").path
                let toml = DefaultConfig.toml(
                    listenPort: GovernorConfig.defaultPort,
                    logDir: logDir)
                cfg = try AthenaConfig.parse(toml: toml)
                configData = Data(toml.utf8)
                synthesized = true
            }
        }

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

        // M38: version/upgrade guard. A prior install leaves a marker in
        // the config dir; classify this install against it so the
        // operator sees fresh / reinstall / upgrade and is WARNED on a
        // DOWNGRADE (an older build can reintroduce a fixed bug or expect
        // an older store schema). Detection only — never blocks.
        let versionMarker = plan.configDir.appendingPathComponent(
            "installed-version")
        let previousVersion =
            (try? String(contentsOf: versionMarker, encoding: .utf8))?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let versionSummary = VersionGuard.summary(
            from: previousVersion, to: Athena.appVersion)
        let isDowngrade =
            VersionGuard.classify(
                from: previousVersion, to: Athena.appVersion)
            == .downgrade

        // The LaunchDaemon's cwd MUST be the libexec dir (the binary's
        // enclosing directory + every `*.bundle` resource sits there).
        // mlx-swift's `Bundle.module` resolver falls back through a
        // chain of cwd-adjacent candidates to locate
        // `mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib`;
        // when cwd is the data dir (`/usr/local/var/athena`) those
        // candidates all miss and the daemon prints "Failed to load
        // the default metallib library not found …" four times at
        // first MLX call. Foreground `athena load` worked because the
        // operator's shell cwd happened to be the build/install dir.
        // The daemon still addresses its data dir explicitly via
        // `--data-dir`, so cwd doesn't need to BE the data dir.
        let plistData = try LaunchdPlist.xmlData(
            label: label,
            executablePath: plan.installedBinary.path,
            user: serviceUser,
            workingDirectory: plan.libexecDir.path,
            config: cfg,
            configPath: plan.installedConfig.path)

        if dryRun {
            print("athena install (dry-run)")
            print("  \(versionSummary)")
            print("  from:    \(plan.sourceDir.path)")
            print("  libexec: \(plan.libexecDir.path)")
            print("  launcher: \(plan.binLauncher.path)")
            print(
                "  config:  \(plan.installedConfig.path)"
                    + (synthesized
                        ? "  (synthesized defaults — none supplied)" : ""))
            print("  plist:   \(plan.plistPath.path)")
            print("  user:    \(serviceUser)")
            print("  copy:    \(plan.artifactNames(fileManager: fm).joined(separator: ", "))")
            if synthesized {
                print("--- athena.toml (synthesized) ---")
                print(String(data: configData, encoding: .utf8) ?? "")
            }
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
            // B7 (M66.3): chmod/chown on an O_NOFOLLOW fd, not by path, so
            // a symlink swapped in at the final component can't redirect
            // the mode/owner change onto another file.
            FsOwn.chmod(url.path, mode: 0o755, isDir: true)
            if let owner, let ids = FsOwn.ids(of: owner) {
                FsOwn.chown(
                    url.path, uid: ids.uid, gid: ids.gid, isDir: true)
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
                    FsOwn.chmod(cur.path, mode: 0o755, isDir: true)
                }
                cur = cur.deletingLastPathComponent()
            }
        }

        // B6 (M66.3): refuse to chown a config-supplied directory that is a
        // shared system path (e.g. `data_dir = "/etc"`) to the service
        // user. The install prefix and the service home stay allowed.
        for (label, url) in [
            ("data_dir", dataDir), ("model_store", modelStore),
            ("log_dir", logURL),
        ] where !isSafeToOwn(url) {
            throw ValidationError(
                "\(label) resolves to '\(url.standardizedFileURL.path)', "
                    + "a system directory the installer refuses to chown "
                    + "to the service user — choose a path under the "
                    + "service home or a dedicated data volume")
        }

        try ensureDir(plan.libexecDir, owner: nil)
        // The launcher's parent (<prefix>/bin) is NOT guaranteed to exist: on
        // Apple Silicon Homebrew lives in /opt/homebrew, so a clean Mac often
        // has no /usr/local/bin, and writing the launcher there fails late
        // ("The file 'athena' doesn't exist") after artifacts are already
        // copied. Create it like the other root-owned system dirs.
        try ensureDir(plan.binLauncher.deletingLastPathComponent(), owner: nil)
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
        // B8 (M66.3): install runs as root via sudo, so the inherited env
        // is the invoker's, not the service user's — don't source the store
        // key from it. A fresh install seeds plaintext; the daemon encrypts
        // on first boot (as the service user, with its own Keychain key).
        if let db = try? AthenaStore(
            path: dbURL, key: StoreKey.resolve(trustEnv: false))
        {
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
                // B17 (M66.2): the bootstrap token gets a default TTL (a
                // 90-day rotation window) instead of living forever — a
                // long-lived all-powerful credential is a standing risk. If
                // it lapses the operator still has the seeded admin
                // PASSWORD (/ui login + `athena auth`) to mint a fresh one,
                // so a TTL can't lock anyone out.
                let seedExpires =
                    Date().timeIntervalSince1970
                    + Double(Self.installSeedTokenTTLDays) * 86_400
                try? await db.putToken(
                    hash: Data(AuthConfig.sha(key)),
                    username: "admin", scopedRoles: nil,
                    label: "install-seed", expires: seedExpires)
                seededToken = key
            }
        }
        // The daemon runs as the service user and must own the DB
        // (install runs as root, so it would otherwise be root-owned
        // and unwritable by the daemon). B3 (M66.3): chown via an
        // O_NOFOLLOW fd, not by path — between seeding the DB (as root) and
        // this chown, a hostile service account could swap the file for a
        // symlink; following it would chown an arbitrary target to the
        // service user. The fd open fails on a symlink instead.
        if let ids = FsOwn.ids(of: serviceUser) {
            for ext in ["", "-wal", "-shm"] {
                let p = dbURL.path + ext
                if fm.fileExists(atPath: p) {
                    FsOwn.chown(
                        p, uid: ids.uid, gid: ids.gid, isDir: false)
                }
            }
        }

        for name in plan.artifactNames(fileManager: fm) {
            let src = plan.sourceDir.appendingPathComponent(name)
            let dst = plan.libexecDir.appendingPathComponent(name)
            guard fm.fileExists(atPath: src.path) else {
                throw ValidationError(
                    "build artifact '\(name)' not found in "
                        + "\(plan.sourceDir.path). Build the whole "
                        + "package via deploy/build.sh, then install "
                        + "from that output.")
            }
            if fm.fileExists(atPath: dst.path) {
                try fm.removeItem(at: dst)
            }
            try fm.copyItem(at: src, to: dst)
        }

        try configData.write(to: plan.installedConfig)
        // ADR 037 — the daemon (running as the service user) must be able to
        // REWRITE its own TOML for `PUT /api/config` / `athena config set` /
        // the WebUI editor. Install runs as root, so the file is root-owned and
        // those daemon-mediated writes would fail `writeFailed`; chown it to the
        // service user (same O_NOFOLLOW-fd pattern as the DB above). This is
        // what removes `sudo` from config changes.
        if let ids = FsOwn.ids(of: serviceUser),
            fm.fileExists(atPath: plan.installedConfig.path)
        {
            FsOwn.chown(
                plan.installedConfig.path, uid: ids.uid, gid: ids.gid,
                isDir: false)
        }
        // Record the installed version so the NEXT install can classify
        // the transition (M38). Best-effort — a failure here must not
        // abort an otherwise-good install.
        try? Data(Athena.appVersion.utf8).write(to: versionMarker)

        // M62 — write a thin `exec` WRAPPER (not a symlink). A symlink at
        // <prefix>/bin/athena leaves `argv[0]` pointing at this bin dir, so
        // MLX's `Bundle.main` lookup for
        // `mlx-swift_Cmlx.bundle/.../default.metallib` searches a dir with no
        // bundle and a foreground `athena load` throws "Failed to load the
        // default metallib". `exec`ing the real libexec binary keeps `argv[0]`
        // beside its resource bundles (the launchd plist already does this for
        // the daemon — see LaunchdPlist).
        try? fm.removeItem(at: plan.binLauncher)
        let launcher = "#!/bin/sh\nexec \"\(plan.installedBinary.path)\" \"$@\"\n"
        try Data(launcher.utf8).write(to: plan.binLauncher)
        try fm.setAttributes(
            [.posixPermissions: 0o755], ofItemAtPath: plan.binLauncher.path)

        try plistData.write(to: plan.plistPath)
        try fm.setAttributes(
            [.posixPermissions: 0o644], ofItemAtPath: plan.plistPath.path)

        // (dirs already created + chowned + made traversable above.)

        _ = Self.launchctl(["bootout", "system", plan.plistPath.path])
        try Self.launchctlChecked(["bootstrap", "system", plan.plistPath.path])
        _ = Self.launchctl(["enable", "system/\(label)"])
        _ = Self.launchctl(["kickstart", "-k", "system/\(label)"])

        print("installed \(label) (service user: \(serviceUser)).")
        print("  \(versionSummary)")
        if isDowngrade {
            print(
                "  WARNING: downgrade — an older build may reintroduce "
                    + "fixed bugs or expect an older store schema; "
                    + "verify compatibility before serving.")
        }
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
                // M45.6 auto-stash: write the seeded admin token to
                // the INVOKING operator's Keychain so `athena <verb>`
                // works from their shell with no --key / ATHENA_KEY /
                // `athena auth login` step. Failures (locked Keychain,
                // weird sudo path) fall back to the manual hint.
                let port = cfg.listenPort
                var stashed = false
                do {
                    try Credentials.storeAsInvokingOperator(
                        tok, host: "127.0.0.1", port: port)
                    stashed = true
                } catch {
                    // Non-fatal; we still print the raw token.
                }
                print("")
                print("  admin bearer token (SAVE NOW — shown once):")
                print("    \(tok)")
                print(
                    "    use:  Authorization: Bearer <token>")
                print(
                    "    mint more: `athena auth token add --user "
                        + "<u>`")
                print(
                    "    expires in \(Self.installSeedTokenTTLDays)d — "
                        + "rotate before then: `athena auth token rotate "
                        + "<hashprefix>` (the seeded password recovers "
                        + "access if it lapses).")
                if stashed {
                    let who =
                        ProcessInfo.processInfo
                        .environment["SUDO_USER"] ?? "current user"
                    print(
                        "    cached in \(who)'s Keychain at "
                            + "127.0.0.1:\(port) — `athena logs` "
                            + "etc. work with no --key.")
                } else {
                    print(
                        "    cache for repeat use: `athena auth "
                            + "login --host 127.0.0.1 --port "
                            + "\(port)` (then paste the token).")
                }
            }
            print("")
        } else {
            print(
                "  auth: existing accounts kept (no admin seeded)")
            print(
                "  locked out? reset offline (no token needed):")
            print(
                "    \(plan.binLauncher.path) auth user passwd admin "
                    + "--data-dir \(dataDir.path)")
            print(
                "  no admin token? mint one offline:")
            print(
                "    \(plan.binLauncher.path) auth token add --user "
                    + "admin --data-dir \(dataDir.path)")
        }
        print(
            "  health: curl -s http://\(cfg.listenHost):\(cfg.listenPort)/healthz")
        print("  logs:   athena logs --follow   # or `log stream --predicate 'subsystem == \"athena\"'`")
        print(
            "  console: http://\(cfg.listenHost):\(cfg.listenPort)/ui"
                + " — sign in to control this RUNNING daemon "
                + "(models, daemon, users, config)")
        print(
            "  note: the daemon serves /healthz + the /ui console "
                + "even with no model loaded; the console controls a "
                + "running daemon — it cannot cold-start a stopped "
                + "one (use launchctl / `athena start`).")
        // M43.4 #2 — a fresh install ships with no LLM in the store;
        // the first /v1/chat/completions returns 400 model_not_available
        // with an empty list. Flag the gap up front and name the fix.
        if Self.modelStoreEmpty(modelStore) {
            print("")
            print(
                "  no LLM in store — `athena pull <id>` then `athena "
                    + "default <id>` before the first chat request.")
        }
    }

    /// True when the model-store root holds zero entries (so any
    /// `/v1/chat/completions` will return `model_not_available`).
    /// Best-effort: a permissions error counts as "not empty" so the
    /// banner doesn't lie about a store it couldn't read.
    private static func modelStoreEmpty(_ url: URL) -> Bool {
        let entries =
            (try? FileManager.default.contentsOfDirectory(
                at: url, includingPropertiesForKeys: nil)) ?? []
        // Ignore dotfiles + bundle directories so a `.DS_Store` doesn't
        // disguise an empty store.
        let visible = entries.filter {
            !$0.lastPathComponent.hasPrefix(".")
        }
        return visible.isEmpty
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
