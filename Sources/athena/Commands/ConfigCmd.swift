import ArgumentParser
import AthenaDeploy
import Foundation

/// `athena config …` — inspect and edit the daemon's flat TOML config
/// (M9.4b). `get` reads through the canonical `AthenaConfig` parser;
/// `set` rewrites a single scalar in place, preserving comments/layout.
struct Config: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "config",
        abstract: "Inspect and edit the daemon TOML config.",
        subcommands: [
            ConfigPath.self, ConfigShow.self, ConfigGet.self,
            ConfigSet.self,
        ])
}

/// String-valued keys are quoted; these two are bare ints.
private let intKeys: Set<String> = ["listen_port", "budget_bytes"]
private let knownKeys: Set<String> = [
    "listen_host", "listen_port", "budget_bytes", "engine", "model",
    "data_dir", "log_dir",
]

/// Config path: `--config` wins; else the installed file if present;
/// else the in-repo dev copy.
private func resolveConfig(_ override: String?) -> URL {
    if let override {
        return URL(
            fileURLWithPath: (override as NSString).expandingTildeInPath)
    }
    let installed = URL(
        fileURLWithPath: "/usr/local/etc/athena/athena.toml")
    if FileManager.default.fileExists(atPath: installed.path) {
        return installed
    }
    return URL(fileURLWithPath: "deploy/athena.toml")
}

private func readConfig(_ url: URL) -> String {
    guard let s = try? String(contentsOf: url, encoding: .utf8) else {
        FailableExit.die("error: no config at \(url.path)")
    }
    return s
}

/// An active (uncommented) `key = …` assignment on this line?
private func isAssignment(_ line: Substring, _ key: String) -> Bool {
    let t = line.drop(while: { $0 == " " || $0 == "\t" })
    guard !t.hasPrefix("#"), t.hasPrefix(key) else { return false }
    let rest = t.dropFirst(key.count)
        .drop(while: { $0 == " " || $0 == "\t" })
    return rest.first == "="
}

/// A commented `# key = …` line (so `set` can uncomment it in place)?
private func isCommented(_ line: Substring, _ key: String) -> Bool {
    var t = line.drop(while: { $0 == " " || $0 == "\t" })
    guard t.first == "#" else { return false }
    t = t.dropFirst().drop(while: { $0 == " " || $0 == "\t" })
    guard t.hasPrefix(key) else { return false }
    let rest = t.dropFirst(key.count)
        .drop(while: { $0 == " " || $0 == "\t" })
    return rest.first == "="
}

struct ConfigPath: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "path", abstract: "Print the resolved config path.")
    @Option(help: "Config file (overrides auto-resolution).")
    var config: String?
    func run() async throws { print(resolveConfig(config).path) }
}

struct ConfigShow: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "show", abstract: "Print the raw config file.")
    @Option(help: "Config file (overrides auto-resolution).")
    var config: String?
    func run() async throws {
        print(
            readConfig(resolveConfig(config))
                .trimmingCharacters(in: .newlines))
    }
}

struct ConfigGet: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "get",
        abstract: "Print one effective value (parsed).")
    @Option(help: "Config file (overrides auto-resolution).")
    var config: String?
    @Argument(help: "Key: \(knownKeys.sorted().joined(separator: "|")).")
    var key: String

    func run() async throws {
        let url = resolveConfig(config)
        let cfg: AthenaConfig
        do {
            cfg = try AthenaConfig.parse(file: url)
        } catch {
            FailableExit.die("error: \(url.path): \(error)")
        }
        let value: String?
        switch key {
        case "listen_host": value = cfg.listenHost
        case "listen_port": value = String(cfg.listenPort)
        case "budget_bytes": value = cfg.budgetBytes.map(String.init)
        case "engine": value = cfg.engine
        case "model": value = cfg.model
        case "data_dir": value = cfg.dataDir
        case "log_dir": value = cfg.logDir
        default:
            FailableExit.die("error: unknown key '\(key)'")
        }
        guard let value else {
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
        guard knownKeys.contains(key) else {
            FailableExit.die(
                "error: unknown key '\(key)' (allowed: "
                    + knownKeys.sorted().joined(separator: ", ") + ")")
        }
        let formatted: String
        if intKeys.contains(key) {
            guard Int(value) != nil else {
                FailableExit.die(
                    "error: \(key) must be an integer")
            }
            formatted = "\(key) = \(value)"
        } else {
            formatted = "\(key) = \"\(value)\""
        }

        let url = resolveConfig(config)
        let original = readConfig(url)
        var lines = original.split(
            separator: "\n", omittingEmptySubsequences: false)

        if let i = lines.firstIndex(where: { isAssignment($0, key) }) {
            lines[i] = Substring(formatted)
        } else if let i = lines.firstIndex(where: {
            isCommented($0, key)
        }) {
            lines[i] = Substring(formatted)
        } else {
            if lines.last?.isEmpty == true { lines.removeLast() }
            lines.append(Substring(formatted))
        }
        let rewritten = lines.joined(separator: "\n") + "\n"

        do {
            try rewritten.write(
                to: url, atomically: true, encoding: .utf8)
        } catch {
            FailableExit.die("error: cannot write \(url.path): \(error)")
        }
        // Sanity-parse; warn (don't roll back) on failure, since it is
        // usually a pre-existing missing required key, not this edit.
        do {
            _ = try AthenaConfig.parse(file: url)
        } catch {
            FileHandle.standardError.write(
                Data("warning: config now unparseable: \(error)\n".utf8))
        }
        print("\(key) = \(value)  (\(url.path))")
    }
}
