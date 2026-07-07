import Foundation

/// Generates the Athena launchd daemon plist directly as a property list —
/// no string template, no `sed` sentinels. Boot-time system daemon
/// (RunAtLoad + KeepAlive); logs to local files for Vector → Clio (Athena
/// initiates nothing to Crete — passive-oracle contract).
public enum LaunchdPlist {

    public static func dictionary(
        label: String,
        executablePath: String,
        user: String,
        workingDirectory: String,
        config: AthenaConfig,
        configPath: String = ""
    ) -> [String: Any] {
        // `executablePath` is the `athena` binary itself; the launchd
        // daemon runs `athena load …` directly. The pre-M42 plist
        // execed the `athenad` shim, which then `execv`-ed `athena`
        // with `argv[0]="athena"` (bare). Under launchd, that argv[0]
        // / actual-binary-path discrepancy made Swift's
        // `Bundle.main.bundleURL` resolve to the wrong directory, so
        // mlx-c's SwiftPM-bundle lookup couldn't find
        // `mlx-swift_Cmlx.bundle` and `MLX.Memory.memoryLimit = …`
        // threw "Failed to load the default metallib library not
        // found …" four times at first call. Foreground `athena load`
        // worked because the shell exec preserved the full path as
        // argv[0]. Skipping `athenad` from the launchd path keeps
        // argv[0] aligned with the kernel's view, so Bundle.main
        // resolves correctly.
        //
        // `--background` (M45.1) tells the daemon it's launchd-spawned
        // so it drops the foreground stdout sink; the macOS unified
        // log is the sole diagnostic surface for the installed
        // daemon. Operators query via `log show / log stream` and
        // tune verbosity at runtime with `sudo log config --mode
        // "level:debug" --subsystem athena`.
        // ADR 037 slice 1 — STATIC plist. The plist no longer freezes ~30 TOML
        // values into `ProgramArguments` (which made every config change a root
        // plist re-render + bootout/bootstrap). It carries only the invariant
        // exec line; the daemon reads the FULL TOML at boot via `ATHENA_CONFIG`
        // (see Load.run — flags still win). A config change is now a TOML edit
        // + restart, never a plist re-render — which is what lets config/restart
        // move off `sudo` (ADR 037). The `config` argument is retained for the
        // few plist-structural keys read below (log dir, service identity).
        let args: [String] = [
            executablePath,
            "load",
            "--background",
        ]

        // Diagnostic surface is the macOS unified log (M45.1). The
        // daemon emits nothing to stdout under launchd, so
        // StandardOutPath is /dev/null. StandardErrorPath stays as a
        // small file under log_dir to catch process-death output
        // (fatalError, NIO precondition failures, MLX/Metal panics)
        // that fires after Logger is torn down — a CRASH DUMP, not a
        // diagnostic log.
        var dict: [String: Any] = [
            "Label": label,
            "ProgramArguments": args,
            "UserName": user,
            "RunAtLoad": true,
            "KeepAlive": true,
            "ProcessType": "Standard",
            "WorkingDirectory": workingDirectory,
            "StandardOutPath": "/dev/null",
            "StandardErrorPath": "\(config.logDir)/athena.err.log",
            "SoftResourceLimits": ["NumberOfFiles": 8192],
        ]
        // NJ2 (M66.4): export the PREFIX-CORRECT installed config path so
        // the daemon's TOML-only re-reads (kv_compression, the egress-proxy
        // keys — none forwarded as `load` args above)
        // resolve to the file this install actually wrote, not the
        // hard-coded `/usr/local`. `ConfigEditor.resolvePath(nil)` reads
        // `ATHENA_CONFIG`.
        if !configPath.isEmpty {
            dict["EnvironmentVariables"] = ["ATHENA_CONFIG": configPath]
        }
        return dict
    }

    public static func xmlData(
        label: String,
        executablePath: String,
        user: String,
        workingDirectory: String,
        config: AthenaConfig,
        configPath: String = ""
    ) throws -> Data {
        let dict = dictionary(
            label: label, executablePath: executablePath, user: user,
            workingDirectory: workingDirectory, config: config,
            configPath: configPath)
        return try PropertyListSerialization.data(
            fromPropertyList: dict, format: .xml, options: 0)
    }
}
