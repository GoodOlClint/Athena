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
        config: AthenaConfig
    ) -> [String: Any] {
        var args: [String] = [
            executablePath, "load",
            "--host", config.listenHost,
            "--port", String(config.listenPort),
        ]
        if let budget = config.budgetBytes {
            args += ["--budget-bytes", String(budget)]
        }
        if let engine = config.engine {
            args += ["--engine", engine]
        }
        if let model = config.model {
            args += ["--model", model]
        }
        if let dataDir = config.dataDir {
            args += ["--data-dir", dataDir]
        }
        if let logLevel = config.logLevel {
            args += ["--log-level", logLevel]
        }

        return [
            "Label": label,
            "ProgramArguments": args,
            "UserName": user,
            "RunAtLoad": true,
            "KeepAlive": true,
            "ProcessType": "Standard",
            "WorkingDirectory": workingDirectory,
            "StandardOutPath": "\(config.logDir)/athena.out.log",
            "StandardErrorPath": "\(config.logDir)/athena.err.log",
            "SoftResourceLimits": ["NumberOfFiles": 8192],
        ]
    }

    public static func xmlData(
        label: String,
        executablePath: String,
        user: String,
        workingDirectory: String,
        config: AthenaConfig
    ) throws -> Data {
        let dict = dictionary(
            label: label, executablePath: executablePath, user: user,
            workingDirectory: workingDirectory, config: config)
        return try PropertyListSerialization.data(
            fromPropertyList: dict, format: .xml, options: 0)
    }
}
