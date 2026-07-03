import Foundation

/// Model-id identity + case-insensitive resolution helpers.
///
/// (Formerly `ModelAllowlist.swift`. The `model_allowlist` SQLite table it was
/// named for was **retired** in ADR 026 — availability is now "what's pulled into
/// the model store," classified by `ModelSupport`. These helpers survived the
/// retirement because they answer a still-live question: given the set of stored
/// model ids, which stored spelling does a request's model field refer to?)
///
/// Resolution is case-insensitive by **store-dir identity**, so a request can name
/// a model by its full HuggingFace id (`Qwen/Qwen3-Embedding-4B`) or its bare
/// store-dir name (`Qwen3-Embedding-4B`, the form `athena pull` creates), in
/// either casing, and resolve to the same stored spelling — which then drives the
/// served-model echo + local-dir load. HuggingFace ids are ASCII in practice, so
/// ASCII `lowercased()` is sufficient and avoids Unicode-normalization surprises.
///
/// Live helpers: `String.modelStoreIdentity`, `[String].canonicalByStoreIdentity`,
/// `[String].dedupedCaseInsensitive`.
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
