import Foundation

/// Cross-cutting KV-cache compression selector (the shared `kv_compression`
/// knob). M20 introduced the key + plumbing; M21 added the `triattention`
/// case. (The M20 `turboquant` case was retired when the substrate dropped
/// its bespoke KV-quant codec in favour of upstream's `kvScheme` hook.)
///
/// `triattention` *evicts* low-importance tokens. The substrate-typed
/// accessors that model that (`eviction`, `servesArch`) live in an
/// `extension KVCompression` in `AthenaLLM`, because they reference MLX
/// types (`TriAttentionConfig`, `SupportedModels`).
///
/// NB4 (M70.1b): the ENUM + the pure `resolve` precedence logic live here in
/// the MLX-free `AthenaCore` so `ConfigEditor`'s `kv_compression ∈
/// KVCompression.allCases` validation can move to `AthenaDeploy` and be
/// unit-tested without the MLX graph (ADR 008 follow-on). The MLX-coupled
/// accessors stay in `AthenaLLM` (which imports `AthenaCore`), so the split is
/// transparent to every caller.
///
/// Precedence: env `ATHENA_KV_COMPRESSION` > TOML `kv_compression` >
/// built-in default `none`. An unrecognized value is a hard error at
/// daemon start (fail-closed — never a silent fallback to `none`), and so is a
/// recognized codec the pinned substrate cannot serve (`unavailableReason`).
public enum KVCompression: String, Sendable, CaseIterable, Equatable {
    case none
    case triattention

    /// Why this codec cannot run at the pinned substrate, or nil if it can.
    /// The substrate fails every generation whose `kvScheme` it does not
    /// recognize, so an unavailable codec is refused here, once, by name.
    public var unavailableReason: String? {
        switch self {
        case .none: return nil
        case .triattention:
            return "kv_compression 'triattention' is unavailable: the pinned "
                + "MLX substrate (integration-2026-10-14) does not include "
                + "TriAttention eviction. Set kv_compression = none "
                + "(ADR 028, 2026-10-14 amendment)."
        }
    }

    public struct UnavailableError: Error, CustomStringConvertible {
        public let reason: String
        public var description: String { reason }
    }

    public struct ResolutionError: Error, CustomStringConvertible {
        public let value: String
        public var description: String {
            "unrecognized kv_compression value '\(value)' — expected one "
                + "of: \(KVCompression.allCases.map(\.rawValue).joined(separator: ", "))"
        }
    }

    /// Pure resolver (inject env + TOML values). Precedence env > TOML >
    /// `none`; empty/whitespace is treated as unset; unknown or unavailable ⇒
    /// throw.
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
        if let reason = v.unavailableReason {
            throw UnavailableError(reason: reason)
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
}
