import Foundation

/// M46.4 — case-insensitive model-id resolution helpers.
///
/// Background: the persisted `model_allowlist` table has a
/// case-sensitive `PRIMARY KEY(module, id)`, and every module's
/// per-request lookup historically used `allowedIds.contains(target)`
/// (case-sensitive). Two foot-guns followed:
///
/// 1. A request asking for `Qwen/Qwen3-Embedding-4b` against an
///    allowlist that stores `Qwen/Qwen3-Embedding-4B` returns a 400
///    `model_not_available`, even though the operator clearly meant
///    the same model. HuggingFace ids are technically case-sensitive
///    (the repo URL is), but in practice operators routinely
///    miscase, and the error message that comes back is unhelpful.
///
/// 2. The two casings can BOTH end up in the allowlist as separate
///    rows (added at different times via different code paths). The
///    400 response then lists both as "Configured models" and the
///    operator has no way to tell them apart.
///
/// M46.4 swaps the lookup to case-insensitive (ASCII `lowercased()`
/// — HuggingFace ids are ASCII in practice, so simple lowercasing is
/// sufficient and avoids Unicode-normalization surprises). The
/// canonical id is whatever's in storage; the request's spelling
/// passes through to the served-model field only when it matches an
/// allowlist row case-insensitively. Historical duplicate rows in
/// the SQLite table aren't deduped here — they're now cosmetic
/// because either casing resolves to the same module load — and
/// the error-response builder deduplicates the displayed
/// "Configured models" list at the wire so an operator sees a clean
/// allowlist.
extension Array where Element == String {
    /// Returns the canonical (stored) id from `self` that matches
    /// `requested` case-insensitively, or nil if none match. Use this
    /// at every model-resolution boundary instead of the raw
    /// `contains(_:)` check — callers should use the RETURNED
    /// canonical id for downstream lookups so storage stays
    /// case-consistent.
    public func canonicalCaseInsensitive(_ requested: String) -> String? {
        first { $0.caseInsensitiveCompare(requested) == .orderedSame }
    }

    /// Returns `self` with case-insensitive duplicates removed,
    /// preserving the FIRST occurrence's case (the canonical row).
    /// Used by `AthenaError.modelNotAvailable`'s display path so a
    /// 400 response doesn't lie about two-cased duplicates when the
    /// SQLite table still carries both rows pending a cleanup pass.
    public func dedupedCaseInsensitive() -> [String] {
        var seen = Set<String>()
        var out: [String] = []
        for s in self {
            let key = s.lowercased()
            if seen.insert(key).inserted { out.append(s) }
        }
        return out
    }
}
