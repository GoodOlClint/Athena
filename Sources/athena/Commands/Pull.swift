import ArgumentParser
import AthenaClient
import AthenaLLM
import Foundation

/// `athena pull MODEL` — download an HF repo into the model store
/// (linked, no multi-GB copy), ollama-style.
struct Pull: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "pull",
        abstract: "Download a model (HF id) into the local store."
    )

    @Argument(help: "HF repo id, e.g. mlx-community/whisper-large-v3-turbo")
    var model: String

    @Option(help: "Git revision / branch / tag (remote only).")
    var revision: String?

    @Flag(help: "Stream job progress (remote only).")
    var follow = false

    @Option(help: "Long-poll N seconds for completion (remote only).")
    var wait: Int?

    @Option(help: "Model store root. Default: ~/.athena/models.")
    var modelStore: String?

    @OptionGroup var daemon: DaemonOptions

    func run() async throws {
        if daemon.isRemote {
            var b: [String: Any] = ["id": model]
            if let revision { b["revision"] = revision }
            try await RemoteModels.job(
                daemon, op: "pull", body: b, follow: follow,
                wait: wait)
            return
        }
        let root =
            modelStore.map { URL(fileURLWithPath: $0, isDirectory: true) }
            ?? ModelStore.defaultRoot
        ProxyEnv.applyConfigAndAuth()  // egress proxy (M13.2)
        HFAuth.exportToEnv()  // gated/private repos (M13)
        print("pulling \(model) …")
        let bar = ProgressBar("  \(model)")
        do {
            let dest = try await ModelPull.pull(
                id: model, into: root,
                progress: { f in bar.update(f) })
            bar.finish()
            print("pulled \(model) → \(dest.path)")
        } catch {
            bar.finish()
            print("error: pull failed — \(error)")
            throw ExitCode.failure
        }
    }
}
