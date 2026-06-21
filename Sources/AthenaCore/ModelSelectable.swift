import Foundation

/// Slot-level model selection (M41; ADR 026). Generalizes the M39 embedding
/// pattern across every module class: a single resident slot rebound on demand.
///
/// ADR 026 retired the `model_allowlist` table — the *selectable set* is now
/// **the model store classified by `ModelSupport` (ADR 021)**, so
/// `allowedModelIds()` is a live store scan of the module's modality rather
/// than a pushed-in operator list. A request that names a model absent from the
/// store (or of the wrong class) is a 400 (`modelNotAvailable`), never an
/// on-request download; an omitted `model` resolves by `ModelSelection`'s
/// ambiguity rule (configured default → sole store model → 400 `ambiguousModel`
/// when >1 with no default). One id is resident at a time — rebind unloads the
/// previous — so the governor accounting stays a fixed per-class estimate.
///
/// Concrete modules conform to this in addition to their typed inference
/// protocol; the server walks every module via the `any ModelSelectable`
/// existential to expose `/api/models/{load,unload,resident}`.
public protocol ModelSelectable: Actor {
    /// The module this slot belongs to.
    nonisolated var moduleID: ModuleID { get }
    /// Selectable model ids — the store dirs `ModelSupport` classifies as this
    /// module's modality (ADR 026). Live: reflects `pull`/`rm` without restart.
    func allowedModelIds() -> [String]
    /// Best-effort default id for display (configured default, or the sole
    /// store model, else ""). The request path resolves the real default via
    /// `ModelSelection` so an ambiguous omit-`model` surfaces a 400.
    func defaultModelId() -> String
    /// The id resident in the slot now, or nil when unloaded.
    func residentModelId() -> String?
    /// Rebind the slot to `id` (nil ⇒ resolve the default per ADR 026).
    /// Idempotent if `id` is already resident. Throws
    /// `AthenaError.modelNotAvailable` for an id absent from the store (400),
    /// `AthenaError.ambiguousModel` when `id` is nil and the store holds >1 of
    /// the class with no configured default (400), and
    /// `AthenaError.moduleLoadFailed` on a substrate failure.
    func rebind(to id: String?) async throws
}
