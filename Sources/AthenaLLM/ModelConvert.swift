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

    /// ADR 021 S3 — given a config-only `ModelSupport` verdict, decide whether
    /// `athena convert` can handle this checkpoint. `convert` is a
    /// generative/vision **quantization** pipeline; the other modalities load
    /// in source precision via their own module paths, so they are REDIRECTED
    /// to `pull` with a cause-naming error rather than mis-routed to the
    /// generative factory (the M76 incident, where a Parakeet checkpoint hit
    /// the LLM factory and produced a misleading "bump the substrate" message).
    ///
    /// Returns nil for a convert target: `.llm` / `.vision`, or `.unsupported`
    /// (no `model_type` — proceed so the substrate factory raises the precise
    /// architecture error, which the existing `looksLikeUnsupportedArch`
    /// handler wraps). Returns a 400 `unsupportedConvertClass` for embedding +
    /// the audio modalities. Guidance echoes the operator's own `id` (dynamic
    /// input, not a hard-coded repo) and names no other repo (ADR 021 D5).
    ///
    /// MLX-free decision logic (ADR 008/009): unit-pinned without a network
    /// fetch or model load.
    public static func convertRedirect(
        for support: ModelSupport, id: String
    ) -> AthenaError? {
        switch support.modality {
        case .llm, .vision, .unsupported:
            return nil
        case .embedding:
            return .unsupportedConvertClass(
                model: id, detected: "embedding",
                guidance:
                    "Embedding models load in source precision directly — run "
                    + "`athena pull \(id)`, then select it as the embedding "
                    + "model (`--embedding-model \(id)` / config). `convert` "
                    + "is for generative and vision models.")
        case .transcription, .diarization, .speakerEmbedding:
            let label = support.modality.label
            return .unsupportedConvertClass(
                model: id, detected: label,
                guidance:
                    "\(label.capitalized) models load in source precision via "
                    + "their audio module paths — run `athena pull \(id)` (it "
                    + "becomes selectable for that module automatically; set it "
                    + "as the default with `athena default --module <m> \(id)`). "
                    + "`convert` quantizes only generative and vision models.")
        }
    }

    /// `progress` (0…1) covers the DOWNLOAD phase only; the
    /// quantization tail has no HF progress. Default nil keeps the
    /// daemon/queue caller unchanged.
    ///
    /// `bits` is **opt-in** (matching `mlx_lm convert -q`/`ollama
    /// --quantize`): nil ⇒ no quantization, the model is converted into
    /// the MLX-native on-disk layout in source precision (output name
    /// `<base>-mlx`); set to N ⇒ quantize to N-bit with `groupSize`
    /// (output name `<base>-mlx-Nbit`).
    public static func convert(
        id: String, revision: String? = nil,
        bits: Int? = nil, groupSize: Int = 64,
        into storeRoot: URL, name: String? = nil,
        progress: (@Sendable (Double) -> Void)? = nil
    ) async throws -> Result {
        let downloader = #hubDownloader(
            HuggingFace.HubClient(
                session: AthenaProxy.proxiedURLSession()))

        // ADR 016/021 fail-fast: fetch only the small metadata needed to
        // classify the model's MODALITY before pulling multi-GB weights, and
        // redirect the classes `convert` does not quantize. Embedding,
        // transcription, diarization and speaker-embedding models all load in
        // source precision via their own module paths and need no converted
        // artifact — so convert REDIRECTS them to `pull` (ADR 021 S3) instead
        // of mis-routing them to the generative factory, which for a Parakeet
        // checkpoint produced the misleading "bump the substrate" error (the
        // M76 incident). The decision is the shared `ModelSupport` predicate,
        // so convert and the loaders/preflight can never disagree.
        let meta = try await downloader.download(
            id: id, revision: revision,
            matching: [
                "config.json", "modules.json",
                "config_sentence_transformers.json",
                "sentence_bert_config.json",
            ],
            useLatest: false,
            progressHandler: { _ in })
        if let redirect = convertRedirect(
            for: ModelSupport.detect(in: meta), id: id)
        {
            throw redirect
        }

        let snapshot = try await downloader.download(
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
        let srcInfo = ModelConfigInfo.read(modelDirectory: snapshot)
        let isVision = srcInfo?.hasVisionConfig ?? false
        // ADR 012 amendment — the mixed 8/4 scheme is Gemma-4-specific; any
        // other arch quantizes uniformly at the global bits (the encoder-tower
        // skip still keeps vision/audio towers full-precision). Decided from
        // the SOURCE config's model_type, then threaded to both the quantize
        // closure and the emitted config so they stay in lock-step.
        let mixedPrecision = Gemma4QuantRule.appliesMixedPrecision(
            modelType: srcInfo?.modelType)
        let container: ModelContainer
        do {
            container = try await AthenaModelRegistration
                .$currentModelDirectory.withValue(snapshot) {
                    let loader = #huggingFaceTokenizerLoader()
                    if isVision {
                        return try await VLMModelFactory.shared.loadContainer(
                            from: snapshot, using: loader)
                    }
                    return try await loadModelContainer(
                        from: snapshot, using: loader)
                }
        } catch where AthenaError.looksLikeUnsupportedArch(error) {
            // A generative/vision checkpoint whose `model_type` the substrate
            // has no architecture for (ADR 016) — name the cause instead of
            // leaking the raw substrate `unsupportedModelType`/`keyNotFound`.
            let cls = isVision ? "vision" : "generative"
            throw AthenaError.unsupportedConvertClass(
                model: id,
                detected: srcInfo?.modelType.map { "\($0) (\(cls))" } ?? cls,
                guidance:
                    "Athena loads only architectures the vendored mlx-swift-lm "
                    + "substrate implements. If upstream has since added this "
                    + "model_type, bump the substrate pin and rebuild.")
        }

        let base = id.split(separator: "/").last.map(String.init) ?? id
        // Naming: all converts carry the `-mlx` family marker; a quantized
        // convert appends the bit-width (`<base>-mlx-4bit`), an unquantized one
        // is just `<base>-mlx`. (Purely a label — quant level is read from
        // config.json, never parsed from the name.)
        let outName =
            name ?? (bits.map { "\(base)-mlx-\($0)bit" } ?? "\(base)-mlx")
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
                // unquantized). The per-layer dense `mlp` + `router.proj` take
                // the 8-bit override ONLY for gemma4 (ADR 012 amendment); other
                // arches quantize uniformly at the global bits.
                let rule = Gemma4QuantRule(
                    groupSize: groupSize, bits: bits,
                    mixedPrecision: mixedPrecision)
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
            quantizedModules: quantizedModules, mixedPrecision: mixedPrecision)
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
        quantizedModules: [String], mixedPrecision: Bool
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
            let rule = Gemma4QuantRule(
                groupSize: groupSize, bits: bits,
                mixedPrecision: mixedPrecision)
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
