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
        let store = ModelStore(
            rootDirectory: modelStore.map {
                URL(fileURLWithPath: $0, isDirectory: true)
            } ?? ModelStore.defaultRoot)
        let source = store.resolve(src)
        let fm = FileManager.default
        guard fm.fileExists(atPath: source.path) else {
            FailableExit.die("error: no such model: \(source.path)")
        }
        guard !dst.contains("/") else {
            FailableExit.die(
                "error: destination must be a bare store name")
        }
        let dest = store.rootDirectory.appendingPathComponent(
            dst, isDirectory: true)

        if fm.fileExists(atPath: dest.path)
            || (try? dest.checkResourceIsReachable()) == true
        {
            guard force else {
                FailableExit.die(
                    "error: \(dst) exists (use --force)")
            }
            try? fm.removeItem(at: dest)
        } else {
            // Catch a dangling symlink at the destination too.
            try? fm.removeItem(at: dest)
        }
        try fm.createDirectory(
            at: store.rootDirectory, withIntermediateDirectories: true)
        do {
            if copy {
                try fm.copyItem(
                    at: source.resolvingSymlinksInPath(), to: dest)
            } else {
                try fm.createSymbolicLink(
                    at: dest, withDestinationURL: source)
            }
        } catch {
            FailableExit.die("error: cp failed: \(error)")
        }
        print(
            "\(copy ? "copied" : "aliased") \(src) → \(dest.path)")
    }
}
