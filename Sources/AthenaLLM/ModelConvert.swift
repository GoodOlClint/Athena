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
/// Memory note: this loads the full unquantized model into unified
/// memory before quantizing (same as `mlx_lm.convert`); converting a
/// very large checkpoint needs headroom well above the daemon budget.
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

    public static func convert(
        id: String, revision: String? = nil,
        bits: Int = 4, groupSize: Int = 64,
        into storeRoot: URL, name: String? = nil
    ) async throws -> Result {
        let snapshot = try await #hubDownloader().download(
            id: id, revision: revision,
            matching: [
                "*.json", "*.safetensors", "*.txt", "*.jinja",
                "tokenizer*", "*.model",
            ],
            useLatest: false, progressHandler: { _ in })

        // Same vendored-model route `serve` uses, so the converted
        // checkpoint loads back through the identical path.
        AthenaModelRegistration.currentModelDirectory = snapshot
        await AthenaModelRegistration.install()
        let container = try await loadModelContainer(
            from: snapshot, using: #huggingFaceTokenizerLoader())

        let base = id.split(separator: "/").last.map(String.init) ?? id
        let outName = name ?? "\(base)-\(bits)bit"
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
            quantize(model: model, groupSize: groupSize, bits: bits)
            let weights = Dictionary(
                uniqueKeysWithValues:
                    model.parameters().flattened())
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
    private static func writeConfig(
        from snapshot: URL, to dest: URL, bits: Int, groupSize: Int
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
        obj["quantization"] = [
            "group_size": groupSize, "bits": bits,
        ]
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
