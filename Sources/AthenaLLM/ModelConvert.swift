import AthenaCore
import Foundation
import HuggingFace
import MLX
import MLXHuggingFace
import MLXLMCommon
import MLXNN
import Tokenizers

/// `mlx_lm.convert` for Athena: download an HF repo, load it
/// unquantized through the same vendored path `serve` uses, quantize
/// the weights, and write `model.safetensors` + a `quantization`-
/// stamped `config.json` (plus tokenizer/template aux) into the model
/// store as `storeRoot/<name>`. Done in-process — no Python. M9.5a.
///
/// Memory note: source weights load lazily (mmap), and the quantized
/// weights are materialized incrementally (each source weight is freed
/// before the next), so peak RAM is bounded by roughly the QUANTIZED
/// output size — not the full-precision model. A very large checkpoint
/// still needs headroom for its quantized output plus working set
/// (e.g. a ~250 GB bf16 model → ~60 GB 4-bit output).
public enum ModelConvert {
    public struct Result: Sendable {
        public let path: URL
        public let bytes: Int
    }

    /// Aux files copied verbatim from the source snapshot. The original
    /// `*.safetensors` (and any shard index) are replaced by our single
    /// re-quantized `model.safetensors`; `config.json` is rewritten.
    private static func isAux(_ name: String) -> Bool {
        if name.hasSuffix(".safetensors") { return false }
        if name == "config.json" { return false }
        if name == "model.safetensors.index.json" { return false }
        return !name.hasPrefix(".")
    }

    /// `progress` (0…1) covers the DOWNLOAD phase only; the
    /// quantization tail has no HF progress. Default nil keeps the
    /// daemon/queue caller unchanged.
    ///
    /// `bits` is **opt-in** (matching `mlx_lm convert -q`/`ollama
    /// --quantize`): nil ⇒ no quantization, the model is converted into
    /// the MLX-native on-disk layout in source precision (output name
    /// `<base>-mlx`); set to N ⇒ quantize to N-bit with `groupSize`
    /// (output name `<base>-Nbit`, as before).
    public static func convert(
        id: String, revision: String? = nil,
        bits: Int? = nil, groupSize: Int = 64,
        into storeRoot: URL, name: String? = nil,
        progress: (@Sendable (Double) -> Void)? = nil
    ) async throws -> Result {
        let snapshot = try await #hubDownloader(
            HuggingFace.HubClient(
                session: AthenaProxy.proxiedURLSession())
        ).download(
            id: id, revision: revision,
            matching: [
                "*.json", "*.safetensors", "*.txt", "*.jinja",
                "tokenizer*", "*.model",
            ],
            useLatest: false,
            progressHandler: { p in
                progress?(p.fractionCompleted)
            })

        // Same vendored-model route `serve` uses, so the converted
        // checkpoint loads back through the identical path.
        AthenaModelRegistration.currentModelDirectory = snapshot
        await AthenaModelRegistration.install()
        let container = try await loadModelContainer(
            from: snapshot, using: #huggingFaceTokenizerLoader())

        let base = id.split(separator: "/").last.map(String.init) ?? id
        let outName =
            name ?? (bits.map { "\(base)-\($0)bit" } ?? "\(base)-mlx")
        let dest = storeRoot.appendingPathComponent(
            outName, isDirectory: true)
        try? FileManager.default.removeItem(at: dest)
        try FileManager.default.createDirectory(
            at: dest, withIntermediateDirectories: true)
        let safetensors = dest.appendingPathComponent(
            "model.safetensors")

        // Quantize + save inside `perform`: the non-Sendable model and
        // its MLXArrays never cross the isolation boundary; only the
        // Void result does.
        try await container.perform { (ctx: ModelContext) in
            // `any LanguageModel` is always a `Module` subclass
            // (BaseLanguageModel: Module), so this is an upcast.
            let model = ctx.model as Module
            if let bits {
                quantize(model: model, groupSize: groupSize, bits: bits)
            }
            let weights = Dictionary(
                uniqueKeysWithValues:
                    model.parameters().flattened())

            // Memory: weights load lazily (mmap), so the quantize→write
            // graph is only materialized on eval. Letting `MLX.save`
            // evaluate the whole graph at once can pull many source (bf16)
            // weights into RAM simultaneously and OOM on large
            // checkpoints. Instead: cap MLX's buffer cache so freed bytes
            // return to the OS, then materialize each quantized weight in
            // turn so its source weight is released before the next —
            // bounding peak RAM to roughly the (much smaller) quantized
            // output rather than the full-precision model.
            let priorCacheLimit = MLX.Memory.cacheLimit
            MLX.Memory.cacheLimit = 256 * 1024 * 1024
            defer { MLX.Memory.cacheLimit = priorCacheLimit }
            for (_, value) in weights {
                value.eval()
            }
            MLX.Memory.clearCache()
            try MLX.save(arrays: weights, url: safetensors)
        }

        try writeConfig(
            from: snapshot, to: dest, bits: bits, groupSize: groupSize)
        try copyAux(from: snapshot, to: dest)

        let attrs = try? FileManager.default.attributesOfItem(
            atPath: safetensors.path)
        let bytes = (attrs?[.size] as? Int) ?? 0
        return Result(path: dest, bytes: bytes)
    }

    /// Rewrite config.json with a `quantization` block so the loader
    /// rebuilds quantized layers; all other model fields are preserved.
    /// `bits == nil` ⇒ no-quantize convert: copy the config through
    /// unchanged (no `quantization` block).
    private static func writeConfig(
        from snapshot: URL, to dest: URL, bits: Int?, groupSize: Int
    ) throws {
        let src = snapshot.appendingPathComponent("config.json")
            .resolvingSymlinksInPath()
        let data = try Data(contentsOf: src)
        guard
            var obj = try JSONSerialization.jsonObject(with: data)
                as? [String: Any]
        else {
            throw AthenaError.moduleLoadFailed(
                .llm, reason: "source config.json is not an object")
        }
        if let bits {
            obj["quantization"] = [
                "group_size": groupSize, "bits": bits,
            ]
        }
        let out = try JSONSerialization.data(
            withJSONObject: obj,
            options: [.prettyPrinted, .sortedKeys])
        try out.write(to: dest.appendingPathComponent("config.json"))
    }

    private static func copyAux(from snapshot: URL, to dest: URL) throws
    {
        let fm = FileManager.default
        let entries =
            (try? fm.contentsOfDirectory(
                at: snapshot, includingPropertiesForKeys: nil)) ?? []
        for entry in entries where isAux(entry.lastPathComponent) {
            let to = dest.appendingPathComponent(
                entry.lastPathComponent)
            try? fm.removeItem(at: to)
            try fm.copyItem(
                at: entry.resolvingSymlinksInPath(), to: to)
        }
    }
}
