import ArgumentParser
import AthenaClient
import AthenaDeploy
import Foundation

/// `athena config …` — inspect and edit the daemon's flat TOML config
/// (M9.4b). `get` reads through the canonical `AthenaConfig` parser;
/// `set` rewrites a single scalar in place via `ConfigEditor`,
/// preserving comments/layout.
struct Config: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "config",
        abstract: "Inspect and edit the daemon TOML config.",
        subcommands: [
            ConfigPath.self, ConfigShow.self, ConfigGet.self,
            ConfigSet.self,
        ])
}

struct ConfigPath: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "path", abstract: "Print the resolved config path.")
    @Option(help: "Config file (overrides auto-resolution).")
    var config: String?
    func run() async throws {
        print(ConfigEditor.resolvePath(config).path)
    }
}

struct ConfigShow: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "show", abstract: "Print the raw config file.")
    @Option(help: "Config file (overrides auto-resolution).")
    var config: String?
    func run() async throws {
        print(
            ConfigEditor.read(ConfigEditor.resolvePath(config))
                .trimmingCharacters(in: .newlines))
    }
}

struct ConfigGet: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "get",
        abstract: "Print one effective value (parsed).")
    @Option(help: "Config file (overrides auto-resolution).")
    var config: String?
    @Argument(
        help: "Key: \(ConfigEditor.knownKeys.sorted().joined(separator: "|")).")
    var key: String

    func run() async throws {
        let url = ConfigEditor.resolvePath(config)
        let cfg: AthenaConfig
        do {
            cfg = try AthenaConfig.parse(file: url)
        } catch {
            FailableExit.die("error: \(url.path): \(error)")
        }
        guard let value = ConfigEditor.value(key, in: cfg) else {
            FailableExit.die("\(key) is unset (built-in default)")
        }
        print(value)
    }
}

struct ConfigSet: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "set",
        abstract: "Set one scalar in place (keeps comments/layout).")
    @Option(help: "Config file (overrides auto-resolution).")
    var config: String?
    @Option(help: "launchd label (for re-bootstrap if installed).")
    var label: String = "me.goodolclint.athena"
    @Flag(
        name: .customLong("no-apply"),
        help: """
            Write the TOML only; skip the re-render + bootstrap (useful \
            for staging changes before a manual restart).
            """)
    var noApply: Bool = false
    @OptionGroup var daemon: DaemonOptions
    @Argument(help: "Key to set.") var key: String
    @Argument(help: "Value.") var value: String

    func run() async throws {
        // ADR 037 — prefer the daemon-mediated path: a reachable daemon owns
        // its TOML (chowned to the service user at install) and writes it via
        // `PUT /api/config`, so `athena config set` needs no sudo. Fall back to
        // the direct/sudo path only when no daemon answers. `--no-apply` keeps
        // the old local-write-only staging behavior.
        if !noApply {
            switch await RemoteConfig.set(daemon, key: key, value: value) {
            case .ok:
                print("\(key) = \(value)  (via daemon at \(daemon.base))")
                print(
                    "note: run `athena restart` to apply (no sudo needed on an "
                        + "installed daemon).")
                return
            case .rejected(let code, let data):
                // The daemon validated it and said no (unknown/denied/bad value)
                // — surface verbatim, don't silently fall back to a local write.
                HTTPClient.printJSON(data)
                throw ExitCode(Int32(code >= 400 ? 1 : 0))
            case .unreachable:
                break  // no daemon — fall through to the local/sudo path
            }
        }
        let url = ConfigEditor.resolvePath(config)
        ConfigEditor.setScalar(key: key, value: value, in: url)
        print("\(key) = \(value)  (\(url.path))")
        // M43.4 #1 — the launchd plist hard-codes every config arg, so
        // a TOML edit alone has no effect on a launchd-managed install.
        // When the plist exists, re-render it from the updated TOML
        // and run `launchctl bootout + bootstrap` so the daemon picks
        // the change up on the next boot of the service. Without root
        // we refuse + name the remedy instead of writing surprise
        // partial state.
        guard !noApply else {
            print(
                "note: --no-apply set; run `sudo athena restart "
                    + "--label \(label)` to take effect.")
            return
        }
        let plist = InstallPlan.plistPath(label: label)
        guard FileManager.default.fileExists(atPath: plist.path) else {
            print(
                "note: no installed LaunchDaemon plist at "
                    + "\(plist.path); the TOML edit applies only to a "
                    + "future install or a user-context `athena load`.")
            return
        }
        guard geteuid() == 0 else {
            FailableExit.die(
                "error: an installed LaunchDaemon at \(plist.path) "
                    + "freezes config values into the plist; re-render "
                    + "needs root. Re-run: sudo athena config set "
                    + "\(key) \(value) (or --no-apply to skip)")
        }
        try Self.rerenderAndBootstrap(
            configURL: url, label: label, plistPath: plist)
        print(
            "applied: plist re-rendered and bootstrapped "
                + "(\(plist.path))")
    }

    /// Re-render the LaunchDaemon plist from the current TOML and
    /// `launchctl bootout + bootstrap` it. Mirrors what `athena install`
    /// does for the plist-write portion; assumes the standard
    /// `/usr/local` prefix the installer uses (a custom `--prefix`
    /// install needs a fresh `athena install` to update its plist).
    private static func rerenderAndBootstrap(
        configURL: URL, label: String, plistPath: URL
    ) throws {
        let cfg: AthenaConfig
        do {
            cfg = try AthenaConfig.parse(file: configURL)
        } catch {
            FailableExit.die(
                "error: cannot reparse \(configURL.path) for plist "
                    + "re-render: \(error)")
        }
        // Preserve the service user + executable path that the
        // INSTALLED plist already encodes. (A custom-prefix install
        // shows up here too; re-derive the libexec path from the
        // existing plist's first ProgramArguments entry.)
        let existing =
            (try? Data(contentsOf: plistPath)).flatMap {
                String(data: $0, encoding: .utf8)
            } ?? ""
        let user = Self.extractUserName(existing) ?? "_athena"
        let exec = Self.extractExecutablePath(existing)
            ?? "/usr/local/libexec/athena/athena"
        let workdir =
            (exec as NSString).deletingLastPathComponent
        let plistData = try LaunchdPlist.xmlData(
            label: label,
            executablePath: exec,
            user: user,
            workingDirectory: workdir,
            config: cfg,
            configPath: configURL.path)  // NJ2: keep ATHENA_CONFIG correct
        try plistData.write(to: plistPath)
        try? FileManager.default.setAttributes(
            [.posixPermissions: 0o644],
            ofItemAtPath: plistPath.path)
        // bootout is non-fatal (returns 3 when the service isn't
        // loaded).
        _ = Self.launchctl(["bootout", "system", plistPath.path])
        let rc = Self.launchctl(
            ["bootstrap", "system", plistPath.path])
        if rc != 0 {
            FailableExit.die(
                "error: launchctl bootstrap \(plistPath.path) failed "
                    + "(status \(rc))")
        }
    }

    /// Pull the `<key>UserName</key><string>VALUE</string>` value out
    /// of the plist XML — used to preserve the service user across a
    /// re-render. Returns nil if the key isn't present.
    private static func extractUserName(_ xml: String) -> String? {
        Self.extractAfterKey(xml, "UserName")
    }

    /// First `<string>` after `ProgramArguments`'s opening `<array>`
    /// — i.e., the daemon binary path that launchd execs. Mirrors what
    /// `athena install` originally wrote into the plist.
    private static func extractExecutablePath(_ xml: String) -> String? {
        guard
            let argsRange = xml.range(
                of: "<key>ProgramArguments</key>")
        else { return nil }
        let tail = xml[argsRange.upperBound...]
        guard let strOpen = tail.range(of: "<string>")
        else { return nil }
        let afterOpen = tail[strOpen.upperBound...]
        guard let strClose = afterOpen.range(of: "</string>")
        else { return nil }
        return String(afterOpen[..<strClose.lowerBound])
    }

    /// `<key>NAME</key>\s*<string>VALUE</string>` extractor — small
    /// hand-roll so this avoids a PropertyListSerialization round-trip
    /// (which would reshape the file and lose comments). The plist is
    /// machine-written and well-formed, so substring lookup is safe.
    private static func extractAfterKey(_ xml: String, _ name: String)
        -> String?
    {
        guard
            let keyRange = xml.range(of: "<key>\(name)</key>")
        else { return nil }
        let tail = xml[keyRange.upperBound...]
        guard let strOpen = tail.range(of: "<string>")
        else { return nil }
        let afterOpen = tail[strOpen.upperBound...]
        guard let strClose = afterOpen.range(of: "</string>")
        else { return nil }
        return String(afterOpen[..<strClose.lowerBound])
    }

    /// `/bin/launchctl` runner with argv (no shell). Output is
    /// discarded; callers only need the exit status.
    private static func launchctl(_ args: [String]) -> Int32 {
        let p = Process()
        p.executableURL = URL(fileURLWithPath: "/bin/launchctl")
        p.arguments = args
        p.standardOutput = FileHandle.nullDevice
        p.standardError = FileHandle.nullDevice
        do { try p.run() } catch { return -1 }
        p.waitUntilExit()
        return p.terminationStatus
    }
}
