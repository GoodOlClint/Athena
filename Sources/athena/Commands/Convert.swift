import ArgumentParser
import AthenaClient
import AthenaCore
import AthenaLLM
import Foundation

/// `athena convert MODEL` — download an HF repo and convert it into
/// the model store (the `mlx_lm.convert` equivalent, in-process), with
/// optional quantization. M9.5a.
struct Convert: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "convert",
        abstract:
            "Convert an HF model into the local MLX-format store "
            + "(optionally quantize)."
    )

    @Argument(help: "HF repo id, e.g. Qwen/Qwen3.5-27B")
    var model: String

    @Option(
        help: """
            Quantize to N-bit. OMIT to convert without quantizing (the
            model is written in source precision, MLX-native layout).
            Matches `mlx_lm convert -q` / `ollama --quantize`: quantization
            is opt-in.
            """
    )
    var qBits: Int?

    @Option(help: "Quantization group size (default 64; only with --q-bits).")
    var qGroupSize: Int = 64

    @Option(
        help: """
            Output name in the store. Default: <repo>-mlx-<bits>bit when
            --q-bits is set, otherwise <repo>-mlx.
            """
    )
    var name: String?

    @Option(help: "Git revision / branch / tag (remote only).")
    var revision: String?

    @Flag(help: "Print download progress as it streams (remote only).")
    var follow = false

    @Option(help: "Model store root. Default: ~/.athena/models.")
    var modelStore: String?

    @OptionGroup var daemon: DaemonOptions

    func run() async throws {
        if daemon.isRemote {
            var b: [String: Any] = [
                "id": model, "group_size": qGroupSize,
            ]
            if let qBits { b["bits"] = qBits }
            if let name { b["name"] = name }
            if let revision { b["revision"] = revision }
            try await RemoteModels.job(
                daemon, op: "convert", body: b, progress: follow)
            return
        }
        let root =
            modelStore.map { URL(fileURLWithPath: $0, isDirectory: true) }
            ?? ModelStore.defaultRoot
        ProxyEnv.applyConfigAndAuth()  // egress proxy (M13.2)
        HFAuth.exportToEnv()  // gated/private repos (M13)
        let target =
            qBits.map { "\($0)-bit" } ?? "MLX format (no quantization)"
        let tail = qBits == nil ? "save" : "quantize"
        print("converting \(model) → \(target) (download, then \(tail)) …")
        // Multi-phase renderer (audit §3): download → load → quantize i/N →
        // write, so a long convert never sits silent.
        let rr = ModelOpRenderer(label: model)
        do {
            let r = try await ModelConvert.convert(
                id: model, bits: qBits, groupSize: qGroupSize,
                into: root, name: name,
                progress: { p in
                    switch p {
                    case let .download(f, b, t):
                        rr.download(fraction: f, bytes: b, total: t)
                    case let .file(name, i, c, b, t, d):
                        rr.file(
                            name: name, index: i, count: c, bytes: b, total: t,
                            done: d)
                    case let .phase(n): rr.phase(n)
                    case let .quantize(i, n): rr.quantize(index: i, count: n)
                    }
                })
            rr.finish()
            let mb = Double(r.bytes) / 1_048_576
            print(
                "converted \(model) → \(r.path.path) "
                    + "(\(String(format: "%.0f", mb)) MB)")
        } catch let e as AthenaError {
            // Cause-naming convert errors (e.g. an embedding-model redirect or
            // an unsupported architecture, ADR 016) carry an actionable
            // message — surface it instead of a raw substrate dump.
            rr.finish()
            print("error: \(e.message)")
            throw ExitCode.failure
        } catch {
            rr.finish()
            print("error: convert failed — \(error)")
            throw ExitCode.failure
        }
    }
}
