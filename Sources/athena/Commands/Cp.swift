import ArgumentParser
import AthenaClient
import AthenaLLM
import Foundation

/// `athena cp SRC DST` — alias (or copy) a stored model under a new
/// name, e.g. pin a converted checkpoint to a stable name. Defaults to
/// a symlink alias (no multi-GB copy, matching how `pull` lands);
/// `--copy` makes a real deep copy. M9.5d.
struct Cp: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "cp",
        abstract: "Alias or copy a stored model under a new name.")

    @Argument(help: "Source model name (or absolute path).")
    var src: String
    @Argument(help: "Destination name in the store.")
    var dst: String

    @Flag(help: "Deep-copy instead of symlink-aliasing.")
    var copy = false
    @Flag(help: "Overwrite an existing destination.")
    var force = false
    @Option(help: "Model store root. Default: external SSD mlx-models.")
    var modelStore: String?

    func run() async throws {
        let root =
            modelStore.map {
                URL(fileURLWithPath: $0, isDirectory: true)
            } ?? ModelStore.defaultRoot
        let dest: URL
        do {
            dest = try ModelStoreOps.copy(
                root: root, src: src, dst: dst,
                deepCopy: copy, force: force)
        } catch ModelStoreOps.OpError.notFound {
            FailableExit.die("error: no such model: \(src)")
        } catch ModelStoreOps.OpError.invalidName {
            FailableExit.die(
                "error: destination must be a bare store name")
        } catch ModelStoreOps.OpError.exists {
            FailableExit.die("error: \(dst) exists (use --force)")
        } catch {
            FailableExit.die("error: \(error)")
        }
        print(
            "\(copy ? "copied" : "aliased") \(src) → \(dest.path)")
    }
}
