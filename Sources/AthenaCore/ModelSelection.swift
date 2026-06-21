import Foundation

/// ADR 026 — store-backed model selection. With the `model_allowlist` table
/// retired, a module's *selectable set* is no longer an operator-curated list
/// in SQLite; it is **the model store, classified by `ModelSupport` (ADR 021)
/// per modality**. The per-module default lives in config (TOML), and the
/// omit-`model` case resolves by an explicit ambiguity rule instead of a
/// stored `is_default` flag.
///
/// This file holds the two MLX-free pieces every module class shares, so the
/// decision logic is unit-pinned once (ADR 008/009) rather than re-derived in
/// each module:
///   - `StoreModelClass.ids` — enumerate the store dirs of a given modality.
///   - `ModelSelection.resolve` — turn (available set, configured default,
///     requested id) into the canonical id to serve, or a typed failure.

/// The outcome of resolving a request's `model` against the store. `resolved`
/// carries the canonical stored id; the two failures map to the existing 400
/// taxonomy at the module boundary (`model_not_available` / `ambiguous_model`).
public enum ModelSelectionOutcome: Equatable, Sendable {
    case resolved(String)
    /// The requested id is absent / wrong-class, OR `model` was omitted and the
    /// store holds zero models of the class. → 400 `model_not_available`.
    case notAvailable
    /// `model` was omitted, no default is configured, and the store holds more
    /// than one model of the class — there is no safe auto-pick. → 400
    /// `ambiguous_model` (the caller must name a `model`).
    case ambiguous
}

public enum ModelSelection {
    /// Resolve a request's effective model.
    ///
    /// - `available`: the store's canonical ids for this module's modality
    ///   (from `StoreModelClass.ids`), already case-/identity-comparable via
    ///   `canonicalByStoreIdentity`.
    /// - `configuredDefault`: the per-module TOML default (nil/empty ⇒ unset).
    /// - `requested`: the request's `model` field (nil/empty ⇒ omitted).
    ///
    /// Rules (ADR 026 §5): a named `model` resolves against the store by
    /// store-dir identity (a miss is `notAvailable`, never an on-request
    /// download); an omitted `model` uses the configured default when it is
    /// present in the store, else the sole store model, else `ambiguous` (>1)
    /// or `notAvailable` (0). The returned id is the canonical stored spelling
    /// so downstream load/echo stays store-consistent.
    public static func resolve(
        available: [String], configuredDefault: String?, requested: String?
    ) -> ModelSelectionOutcome {
        if let requested, !requested.isEmpty {
            if let canonical = available.canonicalByStoreIdentity(requested) {
                return .resolved(canonical)
            }
            return .notAvailable
        }
        // `model` omitted — resolve a default.
        if let def = configuredDefault, !def.isEmpty,
            let canonical = available.canonicalByStoreIdentity(def)
        {
            return .resolved(canonical)
        }
        switch available.count {
        case 1: return .resolved(available[0])
        case 0: return .notAvailable
        default: return .ambiguous
        }
    }

    /// The best-effort default id for display (`/api/models/resident`,
    /// served-model echo): the configured default when it is in the store, the
    /// sole store model when there is exactly one, else "" (ambiguous / empty —
    /// the request path surfaces the real 400). Never throws.
    public static func displayDefault(
        available: [String], configuredDefault: String?
    ) -> String {
        if case .resolved(let id) = resolve(
            available: available, configuredDefault: configuredDefault,
            requested: nil)
        {
            return id
        }
        return ""
    }
}

/// Enumerate the model store by modality (ADR 026). A model is *available* for
/// a module iff its on-disk `config.json` classifies (via `ModelSupport`,
/// ADR 021) to a modality the module accepts. Foundation-only + MLX-free (it
/// composes `ModelSupport.detect`), so it is unit-pinnable and usable from
/// every module package (each depends on `AthenaCore`).
public enum StoreModelClass {
    /// Bare store-dir names (sorted) under `storeRoot` whose `ModelSupport`
    /// modality satisfies `accept`. A child is considered a model iff it has a
    /// readable `config.json` (the same test `ModelStoreOps.list` uses); a
    /// `pull`-created symlink is followed transparently. Empty when the store
    /// root is nil/absent.
    public static func ids(
        storeRoot: URL?, accept: (ModelModality) -> Bool
    ) -> [String] {
        guard let root = storeRoot else { return [] }
        let fm = FileManager.default
        guard
            let entries = try? fm.contentsOfDirectory(
                at: root, includingPropertiesForKeys: nil)
        else { return [] }
        var out: [String] = []
        for dir in entries {
            let cfg = dir.appendingPathComponent("config.json")
            guard fm.fileExists(atPath: cfg.path) else { continue }
            if accept(ModelSupport.detect(in: dir).modality) {
                out.append(dir.lastPathComponent)
            }
        }
        return out.sorted()
    }
}

extension ModelModality {
    /// Modality acceptors per module class (ADR 026). The LLM slot also serves
    /// vision checkpoints (ADR 010/012 — they load through the LLM module's VLM
    /// path), so `.vision` is accepted there.
    public var isLLMSlot: Bool {
        switch self {
        case .llm, .vision: return true
        default: return false
        }
    }
    public var isEmbeddingSlot: Bool {
        if case .embedding = self { return true }
        return false
    }
    public var isTranscriptionSlot: Bool {
        if case .transcription = self { return true }
        return false
    }
    public var isDiarizationSlot: Bool {
        if case .diarization = self { return true }
        return false
    }
    public var isSpeakerEmbeddingSlot: Bool {
        if case .speakerEmbedding = self { return true }
        return false
    }
}
