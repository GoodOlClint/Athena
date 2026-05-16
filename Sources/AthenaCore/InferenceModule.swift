import Foundation

/// Common contract every in-process inference module conforms to. The
/// governor owns module lifecycle through this protocol; it is deliberately
/// limited to lifecycle + memory accounting.
///
/// `infer` is intentionally NOT part of this protocol: a single signature
/// across chat completion, audio transcription, and embedding would be a
/// forced abstraction with no real shared shape. Concrete modules expose
/// their own typed inference API (e.g. `LLMModule.generate`) and the serve
/// path holds a typed handle. See the open question in the M0 commit body.
///
/// Modules are actors: their internal substrate state is isolated. The
/// governor is the single source of truth for *accounting* — it records the
/// reservation it issued rather than polling the module — so eviction
/// decisions never race a module's internal state.
public protocol InferenceModule: Actor {
    /// Stable identity. Constant for the lifetime of the process.
    nonisolated var id: ModuleID { get }

    /// Bytes this module currently holds resident. 0 when unloaded.
    var residentBytes: Int { get }

    /// Estimated bytes a `load` will need. Called by the governor for
    /// admission *before* any reservation is issued.
    func memoryEstimate() -> Int

    /// Bring the module up against an issued reservation. The governor has
    /// already debited `reservation.bytes` from the global budget; throwing
    /// here returns them.
    func load(reservation: MemoryReservation) async throws

    /// Tear the module down and free its resident bytes. Must be safe to
    /// call when already unloaded.
    func unload() async
}
