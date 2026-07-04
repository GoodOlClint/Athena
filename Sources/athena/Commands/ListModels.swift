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

    @Option(
        help:
            "Filter by TYPE, e.g. llm, draft, vision, embed, asr, diar, speaker, unsupported."
    )
    var type: String?

    @OptionGroup var daemon: DaemonOptions

    func run() async throws {
        if daemon.isRemote {
            try await RemoteModels.list(daemon, type: type)
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
        var rows = ModelStoreOps.list(root: root)
        if let type {
            rows = rows.filter {
                ModelTypeFormat.matches(
                    filter: type, modality: $0.modality, engine: $0.engine,
                    draft: $0.draft, fusedMTP: $0.fusedMTP)
            }
        }
        guard !rows.isEmpty else {
            print(
                type == nil
                    ? "no models in \(root.path)"
                    : "no models of type '\(type!)' in \(root.path)")
            return
        }
        let df = DateFormatter()
        df.dateFormat = "yyyy-MM-dd HH:mm"
        print(
            "NAME".padding(toLength: 40, withPad: " ", startingAt: 0)
                + "TYPE".padding(toLength: 14, withPad: " ", startingAt: 0)
                + "SIZE".padding(toLength: 12, withPad: " ", startingAt: 0)
                + "MODIFIED")
        for r in rows {
            let typeCol = ModelTypeFormat.column(
                modality: r.modality, engine: r.engine,
                draft: r.draft, fusedMTP: r.fusedMTP)
            print(
                r.name.padding(
                    toLength: 40, withPad: " ", startingAt: 0)
                    + typeCol.padding(
                        toLength: 14, withPad: " ", startingAt: 0)
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
