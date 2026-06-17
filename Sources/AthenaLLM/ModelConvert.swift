import AthenaCore
import Foundation
import HuggingFace
import MLX
import MLXHuggingFace
import MLXLMCommon
import MLXNN
import MLXVLM
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
        // checkpoint loads back through the identical path. NC1: bind the
        // directory request-scoped (this runs on the queue worker, NOT
        // serialized against a serve cold-load) so the two can't clobber a
        // shared global and mis-decide MTP suppression.
        await AthenaModelRegistration.install()
        // M72 — a vision checkpoint (top-level `vision_config`) MUST load
        // through the VLM path so the image tower is kept: the generic
        // `loadModelContainer` would pick the text LLM factory, whose
        // `sanitize()` strips `vision_tower`, and the converted model would be
        // text-only. Same `hasVisionConfig` detector the serve path (M71.2)
        // uses.
        let isVision =
            ModelConfigInfo.read(modelDirectory: snapshot)?.hasVisionConfig
            ?? false
        let container = try await AthenaModelRegistration
            .$currentModelDirectory.withValue(snapshot) {
                let loader = #huggingFaceTokenizerLoader()
                if isVision {
                    return try await VLMModelFactory.shared.loadContainer(
                        from: snapshot, using: loader)
                }
                return try await loadModelContainer(
                    from: snapshot, using: loader)
            }

        let base = id.split(separator: "/").last.map(String.init) ?? id
        let outName =
            name ?? (bits.map { "\(base)-\($0)bit" } ?? "\(base)-mlx")
        // C6: a caller-supplied `name` like `../evil` would escape the
        // store root through the `removeItem`/`createDirectory` below.
        // Confine the output to a bare child name.
        guard ModelStoreOps.isValidName(outName) else {
            throw ModelStoreOps.OpError.invalidName(outName)
        }
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
        let quantizedModules: [String] = try await container.perform {
            (ctx: ModelContext) in
            // `any LanguageModel` is always a `Module` subclass
            // (BaseLanguageModel: Module), so this is an upcast.
            let model = ctx.model as Module
            if let bits {
                // M72 — quantize via the shared rule: encoder towers stay
                // full-precision (nil ⇒ no `.scales`, the loader leaves them
                // unquantized), the per-layer dense `mlp` + `router.proj` take
                // the 8-bit override, everything else the global bits.
                let rule = Gemma4QuantRule(groupSize: groupSize, bits: bits)
                quantize(
                    model: model,
                    filter: { path, _ in rule.quantization(forPath: path) })
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
            // The quantized modules are exactly those that gained `.scales`
            // (an encoder tower, skipped by the rule, has none). `writeConfig`
            // derives the per-layer override entries from this set, so the
            // saved weights and the emitted config can never disagree.
            return
                weights.keys
                .filter { $0.hasSuffix(".scales") }
                .map { String($0.dropLast(7)) }  // strip ".scales"
        }

        try writeConfig(
            from: snapshot, to: dest, bits: bits, groupSize: groupSize,
            quantizedModules: quantizedModules)
        try copyAux(from: snapshot, to: dest)

        let attrs = try? FileManager.default.attributesOfItem(
            atPath: safetensors.path)
        let bytes = (attrs?[.size] as? Int) ?? 0
        return Result(path: dest, bytes: bytes)
    }

    /// Rewrite config.json with a `quantization` block so the loader
    /// rebuilds quantized layers; all other model fields (incl.
    /// `vision_config` / `audio_config`) are preserved by the whole-object
    /// round-trip. `bits == nil` ⇒ no-quantize convert: copy the config
    /// through unchanged (no `quantization` block).
    ///
    /// M72 — for a quantized convert the block is `{group_size, bits, mode}`
    /// PLUS a per-module override entry for every quantized module whose bits
    /// differ from the global (the dense `mlp` + `router.proj`, at 8-bit).
    /// The override set is derived from `quantizedModules` (the modules that
    /// actually gained `.scales`) via the SAME `Gemma4QuantRule`, so config
    /// and weights agree by construction. The encoder towers are full-
    /// precision (absent from `quantizedModules`) ⇒ no entry ⇒ the
    /// `.scales`-driven loader leaves them unquantized.
    private static func writeConfig(
        from snapshot: URL, to dest: URL, bits: Int?, groupSize: Int,
        quantizedModules: [String]
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
            let rule = Gemma4QuantRule(groupSize: groupSize, bits: bits)
            var quant: [String: Any] = [
                "group_size": groupSize, "bits": bits, "mode": "affine",
            ]
            for o in rule.overrides(forModules: quantizedModules) {
                quant[o.path] = ["group_size": o.groupSize, "bits": o.bits]
            }
            obj["quantization"] = quant
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
