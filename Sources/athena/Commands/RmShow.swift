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
        // Only ever a direct child of the store root (no path escape).
        guard !model.contains("/"), model != "..", model != "." else {
            print("error: invalid model name '\(model)'")
            throw ExitCode.failure
        }
        let dir = root.appendingPathComponent(model, isDirectory: true)
        var isDir: ObjCBool = false
        guard
            FileManager.default.fileExists(
                atPath: dir.path, isDirectory: &isDir), isDir.boolValue
        else {
            print("error: no model '\(model)' in \(root.path)")
            throw ExitCode.failure
        }
        try FileManager.default.removeItem(at: dir)
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
        let dir = root.appendingPathComponent(model, isDirectory: true)
        let cfg = dir.appendingPathComponent("config.json")
        guard let data = try? Data(contentsOf: cfg) else {
            print("error: no model '\(model)' in \(root.path)")
            throw ExitCode.failure
        }
        let size = ListModels.safetensorsSize(dir)
        print("model:    \(model)")
        print("path:     \(dir.path)")
        print("size:     \(ListModels.humanBytes(size))")
        print("config.json:")
        print(String(data: data, encoding: .utf8) ?? "<unreadable>")
    }
}
