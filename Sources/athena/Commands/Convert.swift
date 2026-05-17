import ArgumentParser
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

    @Option(help: "Model store root. Default: ~/.athena/models.")
    var modelStore: String?

    func run() async throws {
        let root =
            modelStore.map { URL(fileURLWithPath: $0, isDirectory: true) }
            ?? ModelStore.defaultRoot
        print("converting \(model) → \(qBits)-bit …")
        do {
            let r = try await ModelConvert.convert(
                id: model, bits: qBits, groupSize: qGroupSize,
                into: root, name: name)
            let mb = Double(r.bytes) / 1_048_576
            print(
                "converted \(model) → \(r.path.path) "
                    + "(\(String(format: "%.0f", mb)) MB)")
        } catch {
            print("error: convert failed — \(error)")
            throw ExitCode.failure
        }
    }
}
