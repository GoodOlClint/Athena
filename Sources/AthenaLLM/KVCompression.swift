import Foundation
import MLXLMCommon

/// Cross-cutting KV-cache compression selector (the shared `kv_compression`
/// knob). M20 introduces the key + full plumbing with the `none` and
/// `turboquant` cases; M21 adds only the `triattention` case.
///
/// Precedence: env `ATHENA_KV_COMPRESSION` > TOML `kv_compression` >
/// built-in default `none`. An unrecognized value is a hard error at
/// daemon start (fail-closed — never a silent fallback to `none`).
public enum KVCompression: String, Sendable, CaseIterable, Equatable {
    case none
    case turboquant
    // `triattention` is added by M21 (its codec does not exist yet);
    // until then it resolves as unrecognized → fail-closed.

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

    /// Substrate scheme + the kvBits the codec needs. `turboquant`
    /// defaults to 4-bit (the codec default); `none` disables KV
    /// quantization (nil kvBits).
    public var generation: (scheme: KVQuantizationScheme, kvBits: Float?) {
        switch self {
        case .none: return (.uniform, nil)
        case .turboquant: return (.turboQuant, 4.0)
        }
    }
}
