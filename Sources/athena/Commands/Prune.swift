import ArgumentParser
import AthenaClient
import AthenaLLM
import Foundation

/// `athena prune` — delete dangling/broken store entries (a dead
/// `pull` symlink whose target vanished, a half-converted dir, a model
/// missing config/safetensors). `--dry-run` previews. Only entries
/// that clearly look like models are ever touched. Logic is shared
/// with the queued `model_prune` op via `ModelStoreOps.prune` (M16.3).
struct Prune: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "prune",
        abstract: "Remove broken/dangling models from the store.")

    @Flag(help: "Show what would be removed; change nothing.")
    var dryRun = false

    @Flag(help: "Stream job progress (remote only).")
    var follow = false

    @Option(help: "Long-poll N seconds for completion (remote only).")
    var wait: Int?

    @Option(help: "Model store root. Default: ~/.athena/models.")
    var modelStore: String?

    @OptionGroup var daemon: DaemonOptions

    func run() async throws {
        if daemon.isRemote {
            try await RemoteModels.job(
                daemon, op: "prune", body: ["dry_run": dryRun],
                follow: follow, wait: wait)
            return
        }
        let root =
            modelStore.map { URL(fileURLWithPath: $0, isDirectory: true) }
            ?? ModelStore.defaultRoot
        let result: ModelStoreOps.PruneResult
        do {
            result = try ModelStoreOps.prune(
                root: root, dryRun: dryRun)
        } catch {
            FailableExit.die("error: no store at \(root.path)")
        }
        if result.victims.isEmpty {
            print("nothing to prune in \(root.path)")
            return
        }
        for v in result.victims {
            print(
                "\(dryRun ? "would remove" : "removing") \(v.name)")
            for p in v.problems { print("    - \(p)") }
        }
        if dryRun {
            print(
                "(dry-run) \(result.victims.count) entr"
                    + (result.victims.count == 1 ? "y" : "ies")
                    + " would go")
            return
        }
        print("pruned \(result.removed)/\(result.victims.count)")
        if result.removed != result.victims.count {
            throw ExitCode.failure
        }
    }
}
