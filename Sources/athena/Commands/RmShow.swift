import ArgumentParser
import AthenaLLM
import Foundation

/// `athena rm MODEL` — delete a model directory from the store.
struct Rm: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "rm",
        abstract: "Remove a model from the local model store."
    )

    @Argument(help: "Model name (a directory under the store root).")
    var model: String

    @Option(help: "Model store root. Default: external SSD mlx-models.")
    var modelStore: String?

    func run() throws {
        let root =
            modelStore.map { URL(fileURLWithPath: $0, isDirectory: true) }
            ?? ModelStore.defaultRoot
        do {
            try ModelStoreOps.remove(root: root, name: model)
        } catch ModelStoreOps.OpError.invalidName {
            print("error: invalid model name '\(model)'")
            throw ExitCode.failure
        } catch ModelStoreOps.OpError.notFound {
            print("error: no model '\(model)' in \(root.path)")
            throw ExitCode.failure
        } catch {
            print("error: \(error)")
            throw ExitCode.failure
        }
        print("removed \(model)")
    }
}

/// `athena show MODEL` — print a model's config + on-disk size.
struct Show: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "show",
        abstract: "Show a model's config and size."
    )

    @Argument(help: "Model name (a directory under the store root).")
    var model: String

    @Option(help: "Model store root. Default: external SSD mlx-models.")
    var modelStore: String?

    func run() throws {
        let root =
            modelStore.map { URL(fileURLWithPath: $0, isDirectory: true) }
            ?? ModelStore.defaultRoot
        guard
            let d = ModelStoreOps.show(root: root, name: model)
        else {
            print("error: no model '\(model)' in \(root.path)")
            throw ExitCode.failure
        }
        print("model:    \(model)")
        print("path:     \(d.path)")
        print("size:     \(ModelStoreOps.humanBytes(d.bytes))")
        print("config.json:")
        print(
            String(data: d.configJSON, encoding: .utf8)
                ?? "<unreadable>")
    }
}
