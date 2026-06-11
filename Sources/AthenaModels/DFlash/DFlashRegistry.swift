import Foundation

/// Built-in target→drafter mapping for DFlash speculative decoding (M63.3b).
/// Mirrors the reference `runtime.registry.DRAFT_REGISTRY`: a target only
/// engages DFlash if a matching drafter resolves. Keyed by a substring of
/// the resident model's store name so both the bare store-dir name and a
/// full HF id resolve. Drafter weights are operator-pulled from Hugging
/// Face at runtime (passive-oracle carve-out); only the mapping is built in.
public enum DFlashRegistry {
    /// (model-name substring, drafter HF id). First match wins. Order most
    /// specific first so `gemma-4-26b-a4b-it` is not shadowed by a broader
    /// pattern.
    public static let pairs: [(match: String, draftId: String)] = [
        ("gemma-4-31b-it", "z-lab/gemma-4-31B-it-DFlash"),
        ("gemma-4-26b-a4b-it", "z-lab/gemma-4-26B-A4B-it-DFlash"),
    ]

    /// The drafter HF id for a resident model name, or nil if none is
    /// registered (DFlash then does not engage; the request decodes
    /// normally). Case-insensitive substring match.
    public static func draftId(forModel name: String) -> String? {
        let lower = name.lowercased()
        for (match, draftId) in pairs where lower.contains(match) {
            return draftId
        }
        return nil
    }
}
