import ArgumentParser
import AthenaClient
import AthenaLLM
import Foundation

/// `athena list` (alias `ls`) — local models in the store, ollama-style.
struct ListModels: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "list",
        abstract: "List models available in the local model store.",
        aliases: ["ls"]
    )

    @Option(help: "Model store root. Default: external SSD mlx-models.")
    var modelStore: String?

    @OptionGroup var daemon: DaemonOptions

    func run() async throws {
        if daemon.isRemote {
            try await RemoteModels.list(daemon)
            return
        }
        let root =
            modelStore.map { URL(fileURLWithPath: $0, isDirectory: true) }
            ?? ModelStore.defaultRoot
        guard
            (try? FileManager.default.contentsOfDirectory(
                atPath: root.path)) != nil
        else {
            print("no model store at \(root.path)")
            return
        }
        let rows = ModelStoreOps.list(root: root)
        guard !rows.isEmpty else {
            print("no models in \(root.path)")
            return
        }
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
                    + ModelStoreOps.humanBytes(r.bytes).padding(
                        toLength: 12, withPad: " ", startingAt: 0)
                    + df.string(from: r.modified))
        }
    }

    /// Forwarders kept so existing callers (`show`) are unchanged; the
    /// canonical math lives in `ModelStoreOps` (M16.2).
    static func safetensorsSize(_ dir: URL) -> Int {
        ModelStoreOps.safetensorsSize(dir)
    }

    static func humanBytes(_ n: Int) -> String {
        ModelStoreOps.humanBytes(n)
    }
}
