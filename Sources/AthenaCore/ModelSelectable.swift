import Foundation

/// Slot-level model selection (M41). Generalizes the M39 embedding pattern
/// across every module class: a single resident slot rebound on demand
/// from an operator-declared allowlist; a request that names a model
/// outside the set is a 400 (`modelNotAvailable`), never an on-request
/// download. One id is resident at a time — rebind unloads the previous
/// — so the governor accounting stays a fixed per-class estimate.
///
/// Concrete modules conform to this in addition to their typed inference
/// protocol; the server walks every module via the `any ModelSelectable`
/// existential to expose `/api/models/{load,unload,resident}`.
public protocol ModelSelectable: Actor {
    /// The module this slot belongs to.
    nonisolated var moduleID: ModuleID { get }
    /// Selectable model ids (operator-declared, first = default).
    func allowedModelIds() -> [String]
    /// The default id used when a request omits `model`.
    func defaultModelId() -> String
    /// The id resident in the slot now, or nil when unloaded.
    func residentModelId() -> String?
    /// Rebind the slot to `id` (nil ⇒ the default). Idempotent if `id`
    /// is already resident. Throws `AthenaError.modelNotAvailable` for
    /// an id outside `allowedModelIds()` (400) and
    /// `AthenaError.moduleLoadFailed` on a substrate failure.
    func rebind(to id: String?) async throws
}
