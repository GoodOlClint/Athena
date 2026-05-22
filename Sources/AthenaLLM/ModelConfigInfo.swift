import Foundation

/// Architecture-agnostic view of a model's `config.json` — the dimensions
/// Athena needs without loading the model or depending on MLX. Used to
/// size the governor prompt-cache cap per-arch (M23 fork C), to build the
/// structured-output vocabulary for any architecture (M23 fork A), and to
/// report a model's type in `athena show` (M23 fork D).
///
/// Fields are read top-level first, then from a nested `text_config`
/// (multimodal / wrapper configs such as Gemma 3 and Qwen3.5 place the
/// transformer dims there). Every field is optional: a model whose config
/// omits a dimension simply yields `nil`, and callers fall back to a
/// conservative constant rather than guessing.
public struct ModelConfigInfo: Sendable, Equatable {
    public let modelType: String?
    public let vocabSize: Int?
    public let numHiddenLayers: Int?
    public let numAttentionHeads: Int?
    public let numKeyValueHeads: Int?
    public let headDim: Int?
    public let hiddenSize: Int?

    public init(
        modelType: String? = nil, vocabSize: Int? = nil,
        numHiddenLayers: Int? = nil, numAttentionHeads: Int? = nil,
        numKeyValueHeads: Int? = nil, headDim: Int? = nil,
        hiddenSize: Int? = nil
    ) {
        self.modelType = modelType
        self.vocabSize = vocabSize
        self.numHiddenLayers = numHiddenLayers
        self.numAttentionHeads = numAttentionHeads
        self.numKeyValueHeads = numKeyValueHeads
        self.headDim = headDim
        self.hiddenSize = hiddenSize
    }

    /// KV heads actually used by attention — `num_key_value_heads` when
    /// present (grouped-query attention), else `num_attention_heads`
    /// (multi-head). nil only when neither is in the config.
    public var effectiveKVHeads: Int? {
        numKeyValueHeads ?? numAttentionHeads
    }

    /// Per-head dimension — explicit `head_dim` when present, else
    /// `hidden_size / num_attention_heads`. nil when it can't be derived.
    public var effectiveHeadDim: Int? {
        if let headDim { return headDim }
        guard let hiddenSize, let heads = numAttentionHeads, heads > 0
        else { return nil }
        return hiddenSize / heads
    }

    /// Conservative per-token KV-cache footprint for the M5 prompt-cache
    /// cap: `2 (K+V) · layers · kv_heads · head_dim · bytesPerElement`.
    /// Returns nil when any dimension is missing, so the caller keeps the
    /// conservative built-in constant rather than under-sizing the cap.
    ///
    /// Over-counts for architectures with recurrent/linear-attention
    /// layers (those store no per-token KV) — the safe direction for an
    /// OOM guard, which must refuse early rather than admit too much.
    public func perTokenKVBytes(bytesPerElement: Int) -> Int? {
        guard let layers = numHiddenLayers,
            let kvHeads = effectiveKVHeads,
            let dim = effectiveHeadDim,
            layers > 0, kvHeads > 0, dim > 0, bytesPerElement > 0
        else { return nil }
        return 2 * layers * kvHeads * dim * bytesPerElement
    }

    /// Parse a `config.json` payload. Unknown/missing fields ⇒ nil. A
    /// non-object or unparseable payload ⇒ an all-nil value.
    public static func parse(configJSON data: Data) -> ModelConfigInfo {
        guard
            let obj = try? JSONSerialization.jsonObject(with: data)
                as? [String: Any]
        else { return ModelConfigInfo() }
        let nested = obj["text_config"] as? [String: Any]

        func int(_ key: String) -> Int? {
            // top-level wins; fall back to text_config. Accept Int or a
            // numeric value bridged as NSNumber (JSONSerialization).
            if let v = (obj[key] as? NSNumber)?.intValue { return v }
            if let v = (nested?[key] as? NSNumber)?.intValue { return v }
            return nil
        }
        func string(_ key: String) -> String? {
            (obj[key] as? String) ?? (nested?[key] as? String)
        }

        return ModelConfigInfo(
            modelType: string("model_type"),
            vocabSize: int("vocab_size"),
            numHiddenLayers: int("num_hidden_layers"),
            numAttentionHeads: int("num_attention_heads"),
            numKeyValueHeads: int("num_key_value_heads"),
            headDim: int("head_dim"),
            hiddenSize: int("hidden_size"))
    }

    /// Read + parse `config.json` from a model directory, resolving a
    /// store-entry symlink first (`pull` lands a model as a symlink). nil
    /// when the file is absent or unreadable.
    public static func read(modelDirectory: URL) -> ModelConfigInfo? {
        let cfg = modelDirectory.resolvingSymlinksInPath()
            .appendingPathComponent("config.json")
        guard let data = try? Data(contentsOf: cfg) else { return nil }
        return parse(configJSON: data)
    }
}
