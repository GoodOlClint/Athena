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
extension String {
    /// The store-dir identity of a model id: the basename after the last
    /// `/` (an HF `org/name` id → `name`; an absolute path → its last
    /// component; an already-bare name → itself). This is the directory
    /// name `athena pull` creates under the model store, so it is the
    /// common key by which a model can be named in either its full HF
    /// form or its bare store-dir form.
    public var modelStoreIdentity: String {
        String(split(separator: "/").last ?? Substring(self))
    }
}

extension Array where Element == String {
    /// Match `requested` against `self` by STORE-DIR IDENTITY
    /// (`modelStoreIdentity`, case-insensitive), returning the canonical
    /// stored id. This lets a request name a model by either its full
    /// HuggingFace id (`Qwen/Qwen3-Embedding-4B`) or its bare store-dir
    /// name (`Qwen3-Embedding-4B`, the form `athena pull` creates) and
    /// resolve the same allowlist row — mirroring how the LLM module
    /// already accepts bare store-dir names. The returned canonical id is
    /// the stored spelling (used for served-model echo + local-dir
    /// resolution). Falls back to nothing if no entry shares the identity.
    ///
    /// NOTE collision: two configured ids with the same basename but
    /// different orgs (`a/m` vs `b/m`) collapse to one identity; the model
    /// store already keys by basename (`athena pull`), so the store can
    /// hold only one — callers should guard/declare-time reject such a
    /// clash rather than silently alias.
    public func canonicalByStoreIdentity(_ requested: String) -> String? {
        let key = requested.modelStoreIdentity
        return first {
            $0.modelStoreIdentity.caseInsensitiveCompare(key) == .orderedSame
        }
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
