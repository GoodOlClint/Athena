import ArgumentParser
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
    @Argument(help: "Key to set.") var key: String
    @Argument(help: "Value.") var value: String

    func run() async throws {
        let url = ConfigEditor.resolvePath(config)
        ConfigEditor.setScalar(key: key, value: value, in: url)
        print("\(key) = \(value)  (\(url.path))")
    }
}
