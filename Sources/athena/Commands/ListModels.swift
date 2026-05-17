import ArgumentParser
import AthenaLLM
import Foundation

/// `athena list` (alias `ls`) — local models in the store, ollama-style.
struct ListModels: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "list",
        abstract: "List models available in the local model store.",
        aliases: ["ls"]
    )

    @Option(help: "Model store root. Default: external SSD mlx-models.")
    var modelStore: String?

    func run() throws {
        let root =
            modelStore.map { URL(fileURLWithPath: $0, isDirectory: true) }
            ?? ModelStore.defaultRoot
        let fm = FileManager.default
        guard
            let entries = try? fm.contentsOfDirectory(
                at: root, includingPropertiesForKeys: [
                    .isDirectoryKey, .contentModificationDateKey,
                ])
        else {
            print("no model store at \(root.path)")
            return
        }

        var rows: [(name: String, size: Int, modified: Date)] = []
        for dir in entries {
            let cfg = dir.appendingPathComponent("config.json")
            guard fm.fileExists(atPath: cfg.path) else { continue }
            let size =
                ((try? fm.contentsOfDirectory(
                    at: dir, includingPropertiesForKeys: [.fileSizeKey]))
                ?? [])
                .filter { $0.pathExtension == "safetensors" }
                .reduce(0) {
                    $0
                        + ((try? $1.resourceValues(forKeys: [.fileSizeKey]))?
                            .fileSize ?? 0)
                }
            let mod =
                (try? dir.resourceValues(forKeys: [
                    .contentModificationDateKey
                ]))?.contentModificationDate ?? .distantPast
            rows.append((dir.lastPathComponent, size, mod))
        }

        guard !rows.isEmpty else {
            print("no models in \(root.path)")
            return
        }
        rows.sort { $0.name < $1.name }
        let df = DateFormatter()
        df.dateFormat = "yyyy-MM-dd HH:mm"
        print(
            "NAME".padding(toLength: 40, withPad: " ", startingAt: 0)
                + "SIZE".padding(toLength: 12, withPad: " ", startingAt: 0)
                + "MODIFIED")
        for r in rows {
            print(
                r.name.padding(
                    toLength: 40, withPad: " ", startingAt: 0)
                    + Self.humanBytes(r.size).padding(
                        toLength: 12, withPad: " ", startingAt: 0)
                    + df.string(from: r.modified))
        }
    }

    static func humanBytes(_ n: Int) -> String {
        let u = ["B", "KB", "MB", "GB", "TB"]
        var v = Double(n), i = 0
        while v >= 1024, i < u.count - 1 { v /= 1024; i += 1 }
        return String(format: i == 0 ? "%.0f %@" : "%.1f %@", v, u[i])
    }
}
