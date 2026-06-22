import AthenaCore
import AthenaModels
import Foundation
import MLXLMCommon

/// MLX-coupled accessors for the `KVCompression` codec selector. The enum
/// itself + the pure `resolve` precedence logic live in `AthenaCore`
/// (MLX-free, so `ConfigEditor`'s value validation is CI-testable — NB4); the
/// substrate-typed seams that reference `TriAttentionConfig` and
/// `SupportedModels` stay here, in the MLX-linked module, as an extension.
extension KVCompression {
    /// Token-eviction policy. Non-nil only for `triattention`. Wired into
    /// the vendored model's cache construction for the standard attention
    /// path only (inert on MTP/speculative).
    public var eviction: TriAttentionConfig? {
        switch self {
        case .none: return nil
        case .triattention: return TriAttentionConfig()
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
