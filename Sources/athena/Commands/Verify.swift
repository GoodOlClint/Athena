import ArgumentParser
import AthenaClient
import AthenaLLM
import Foundation

/// `athena verify [NAME]` — offline structural integrity check of a
/// stored model (config + safetensors headers + tokenizer). `--all`
/// sweeps the whole store. M9.5b.
struct Verify: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "verify",
        abstract: "Check a stored model's integrity (offline).")

    @Argument(help: "Model name (omit with --all).")
    var name: String?

    @Flag(help: "Verify every model in the store.")
    var all = false

    @Option(help: "Model store root. Default: ~/.athena/models.")
    var modelStore: String?

    func run() async throws {
        let root =
            modelStore.map { URL(fileURLWithPath: $0, isDirectory: true) }
            ?? ModelStore.defaultRoot

        let targets: [(name: String, url: URL)]
        if all {
            let subs =
                (try? FileManager.default.contentsOfDirectory(
                    at: root,
                    includingPropertiesForKeys: [
                        .isDirectoryKey
                    ])) ?? []
            targets =
                subs
                .filter {
                    (try? $0.resourceValues(forKeys: [.isDirectoryKey])
                        .isDirectory) == true
                }
                .map { ($0.lastPathComponent, $0) }
                .sorted { $0.name < $1.name }
            if targets.isEmpty {
                print("no models in \(root.path)")
                return
            }
        } else {
            guard let name else {
                FailableExit.die(
                    "error: provide a model name or --all")
            }
            targets = [(name, root.appendingPathComponent(name))]
        }

        var bad = 0
        for t in targets {
            let problems = ModelHealth.check(t.url)
            if problems.isEmpty {
                print("ok    \(t.name)")
            } else {
                bad += 1
                print("BAD   \(t.name)")
                for p in problems { print("        - \(p)") }
            }
        }
        if bad > 0 { throw ExitCode.failure }
    }
}
