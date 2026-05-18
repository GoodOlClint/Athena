import ArgumentParser
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

    @Option(help: "Model store root. Default: ~/.athena/models.")
    var modelStore: String?

    func run() async throws {
        let root =
            modelStore.map { URL(fileURLWithPath: $0, isDirectory: true) }
            ?? ModelStore.defaultRoot
        ProxyEnv.applyConfigAndAuth()  // egress proxy (M13.2)
        HFAuth.exportToEnv()  // gated/private repos (M13)
        print("pulling \(model) …")
        do {
            let dest = try await ModelPull.pull(id: model, into: root)
            print("pulled \(model) → \(dest.path)")
        } catch {
            print("error: pull failed — \(error)")
            throw ExitCode.failure
        }
    }
}
