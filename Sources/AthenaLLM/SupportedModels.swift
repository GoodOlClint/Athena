import Foundation

/// Athena's validated architecture set (M23 fork D). The substrate factory
/// can instantiate ~50 architectures from a `model_type`; this is the
/// curated subset Athena has sample-tested for load + generate (+ guided
/// structured output on the substrate path). Anything else still loads
/// best-effort — we just don't claim it.
///
/// Capability tiers:
/// - **Qwen3.5 family** — the vendored model: MTP speculative decoding,
///   TriAttention eviction, and guided structured output.
/// - **validated substrate** — sample-tested: guided structured output;
///   MTP speculative and TriAttention eviction do not apply (they attach
///   to the vendored Qwen3.5 model only).
/// - **best-effort** — loadable via the substrate factory, not in the
///   validated set.
public enum SupportedModels {

    /// `model_type`s routed to the vendored Qwen3.5 model (full features).
    public static let qwen35Family: Set<String> = [
        "qwen3_5", "qwen3_5_moe", "qwen3_5_text",
    ]

    /// True for any Qwen3.5-family `model_type` (case-insensitive).
    public static func isQwen35(_ modelType: String?) -> Bool {
        guard let t = modelType?.lowercased() else { return false }
        return qwen35Family.contains(t)
    }

    /// Sample-tested non-Qwen3.5 architectures Athena claims support for.
    public static let validatedSubstrate: Set<String> = [
        "llama", "mistral", "qwen2", "qwen3", "qwen3_moe",
        "gemma", "gemma2", "gemma3", "gemma3_text",
        "gemma4", "gemma4_text", "gemma4_unified", "gemma4_unified_text",
        "phi", "phi3", "phimoe",
    ]

    public enum Support: String, Equatable, Sendable {
        case qwen35
        case validated
        case bestEffort
    }

    public static func support(for modelType: String?) -> Support {
        if isQwen35(modelType) { return .qwen35 }
        if let t = modelType?.lowercased(), validatedSubstrate.contains(t) {
            return .validated
        }
        return .bestEffort
    }

    /// One-line capability summary for `athena show`.
    public static func describe(modelType: String?) -> String {
        let t = modelType ?? "unknown"
        switch support(for: modelType) {
        case .qwen35:
            return
                "\(t) — Qwen3.5 (vendored): MTP speculative, TriAttention "
                + "eviction, guided structured output"
        case .validated:
            return
                "\(t) — validated substrate arch: guided structured output "
                + "(no MTP speculative / TriAttention eviction)"
        case .bestEffort:
            return
                "\(t) — best-effort: loads via the substrate factory; not "
                + "in Athena's validated set"
        }
    }
}
