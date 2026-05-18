import ArgumentParser
import AthenaClient
import AthenaLLM
import Foundation

/// `athena convert MODEL` — download an HF repo and quantize it into
/// the model store (the `mlx_lm.convert` equivalent, in-process). M9.5a.
struct Convert: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "convert",
        abstract: "Quantize an HF model into the local store."
    )

    @Argument(help: "HF repo id, e.g. Qwen/Qwen3.5-27B")
    var model: String

    @Option(help: "Quantization bits (default 4).")
    var qBits: Int = 4

    @Option(help: "Quantization group size (default 64).")
    var qGroupSize: Int = 64

    @Option(help: "Output name in the store (default <repo>-<bits>bit).")
    var name: String?

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
            var b: [String: Any] = [
                "id": model, "bits": qBits, "group_size": qGroupSize,
            ]
            if let name { b["name"] = name }
            if let revision { b["revision"] = revision }
            try await RemoteModels.job(
                daemon, op: "convert", body: b, follow: follow,
                wait: wait)
            return
        }
        let root =
            modelStore.map { URL(fileURLWithPath: $0, isDirectory: true) }
            ?? ModelStore.defaultRoot
        ProxyEnv.applyConfigAndAuth()  // egress proxy (M13.2)
        HFAuth.exportToEnv()  // gated/private repos (M13)
        print("converting \(model) → \(qBits)-bit (download, "
            + "then quantize) …")
        let bar = ProgressBar("  \(model)")
        do {
            let r = try await ModelConvert.convert(
                id: model, bits: qBits, groupSize: qGroupSize,
                into: root, name: name,
                progress: { f in bar.update(f) })
            bar.finish()
            let mb = Double(r.bytes) / 1_048_576
            print(
                "converted \(model) → \(r.path.path) "
                    + "(\(String(format: "%.0f", mb)) MB)")
        } catch {
            bar.finish()
            print("error: convert failed — \(error)")
            throw ExitCode.failure
        }
    }
}
