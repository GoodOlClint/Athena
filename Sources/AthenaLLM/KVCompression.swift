import AthenaModels
import Foundation
import MLXLMCommon

/// Cross-cutting KV-cache compression selector (the shared `kv_compression`
/// knob). M20 introduces the key + full plumbing with the `none` and
/// `turboquant` cases; M21 adds the `triattention` case.
///
/// `turboquant` *quantizes* KV numerics (the `generation` tuple);
/// `triattention` *evicts* low-importance tokens (the `eviction`
/// accessor — a distinct seam, since eviction is not a quant scheme).
///
/// Precedence: env `ATHENA_KV_COMPRESSION` > TOML `kv_compression` >
/// built-in default `none`. An unrecognized value is a hard error at
/// daemon start (fail-closed — never a silent fallback to `none`).
public enum KVCompression: String, Sendable, CaseIterable, Equatable {
    case none
    case turboquant
    case triattention

    public struct ResolutionError: Error, CustomStringConvertible {
        public let value: String
        public var description: String {
            "unrecognized kv_compression value '\(value)' — expected one "
                + "of: \(KVCompression.allCases.map(\.rawValue).joined(separator: ", "))"
        }
    }

    /// Pure resolver (inject env + TOML values). Precedence env > TOML >
    /// `none`; empty/whitespace is treated as unset; unknown ⇒ throw.
    public static func resolve(env: String?, toml: String?) throws -> KVCompression {
        func clean(_ s: String?) -> String? {
            guard let t = s?.trimmingCharacters(in: .whitespacesAndNewlines),
                !t.isEmpty
            else { return nil }
            return t
        }
        guard let raw = clean(env) ?? clean(toml) else { return .none }
        guard let v = KVCompression(rawValue: raw.lowercased()) else {
            throw ResolutionError(value: raw)
        }
        return v
    }

    /// Process-env wrapper: reads `ATHENA_KV_COMPRESSION`, falling back to
    /// the supplied TOML value, then `none`.
    public static func resolve(config tomlValue: String?) throws -> KVCompression {
        try resolve(
            env: ProcessInfo.processInfo.environment["ATHENA_KV_COMPRESSION"],
            toml: tomlValue)
    }

    /// Substrate KV-quantization scheme + kvBits. `turboquant` defaults
    /// to 4-bit; `none` and `triattention` perform no KV quantization
    /// (`triattention` evicts tokens instead — see `eviction`).
    public var generation: (scheme: KVQuantizationScheme, kvBits: Float?) {
        switch self {
        case .none, .triattention: return (.uniform, nil)
        case .turboquant: return (.turboQuant, 4.0)
        }
    }

    /// Token-eviction policy. Non-nil only for `triattention`; this is
    /// the separate seam (the `generation` quant tuple does not model
    /// eviction). Wired into the vendored model's cache construction for
    /// the standard attention path only (inert on MTP/speculative).
    public var eviction: TriAttentionConfig? {
        switch self {
        case .none, .turboquant: return nil
        case .triattention: return TriAttentionConfig()
        }
    }

    /// Whether this codec actually affects the given architecture (M23
    /// fork B). TurboQuant is a substrate-level KV-quant codec that
    /// applies to any arch; TriAttention eviction attaches only to the
    /// vendored Qwen3.5 model, so it is a no-op for other architectures
    /// (the request still runs, just uncompressed). `none` trivially
    /// "serves" everything.
    ///
    /// A `false` result is NOT fail-closed: per the fork-B decision an
    /// inert-but-valid codec warns + runs uncompressed. Fail-closed is
    /// reserved for an unrecognized VALUE (see `resolve`).
    public func servesArch(modelType: String?) -> Bool {
        switch self {
        case .none, .turboquant: return true
        case .triattention: return SupportedModels.isQwen35(modelType)
        }
    }
}
