import AthenaCore
import Foundation
import MLXLMCommon

/// MLX-coupled accessors for the `KVCompression` codec selector. The enum
/// itself + the pure `resolve` precedence logic live in `AthenaCore`
/// (MLX-free, so `ConfigEditor`'s value validation is CI-testable — NB4); the
/// substrate-typed seams that reference `SupportedModels` stay here, in the
/// MLX-linked module, as an extension.
extension KVCompression {
    /// Substrate `GenerateParameters.kvScheme` string for this codec (the
    /// upstream `applyKVScheme` hook swaps each `KVCacheSimple` for a
    /// self-evicting `TriAttentionKVCache`). Non-nil only for `triattention`;
    /// the substrate applies it on the standard attention path only (inert on
    /// MTP/speculative/batch — those pass `parameters: nil` or no scheme).
    public var kvScheme: String? {
        switch self {
        case .none: return nil
        case .triattention: return "triattention"
        }
    }

    /// Whether this codec actually affects the given architecture (M23
    /// fork B). TriAttention eviction attaches only to the vendored
    /// Qwen3.5 model, so it is a no-op for other architectures (the
    /// request still runs, just uncompressed). `none` trivially "serves"
    /// everything.
    ///
    /// A `false` result is NOT fail-closed: per the fork-B decision an
    /// inert-but-valid codec warns + runs uncompressed. Fail-closed is
    /// reserved for an unrecognized VALUE (see `resolve`).
    public func servesArch(modelType: String?) -> Bool {
        switch self {
        case .none: return true
        case .triattention: return SupportedModels.isQwen35(modelType)
        }
    }
}
