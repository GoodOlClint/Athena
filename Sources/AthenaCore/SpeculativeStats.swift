import Foundation

/// M47.2 — per-iteration acceptance signal for the MTP speculative loop.
///
/// Plays the same role for speculative perf observability that
/// `DecodeProgress` plays for token-emission progress: an opt-in
/// `@TaskLocal` the decode loop publishes to and observers (the test
/// tallies) read from. Lives in AthenaCore so the
/// AthenaLLM decode path can publish without acquiring a dependency
/// on any concrete observer.
///
/// Under tight JSON schemas the unmasked-MTP-draft failure mode the
/// M47 root-cause analysis identified produces ~0% accept; the M47.2
/// Guide-masked-draft fix is supposed to restore acceptance into the
/// natural backbone/MTP-agreement-within-the-valid-set range. A test
/// reads this counter to assert the change actually moved the needle.
public protocol SpeculativeAcceptanceObserver: AnyObject, Sendable {
    /// Called once per iteration of the speculative loop with
    /// `accepted = (draft == verifyPred)`. Implementations should be
    /// cheap (an NSLock-isolated counter pair is the canonical shape).
    func recordIteration(accepted: Bool)
}

public enum SpeculativeStats {
    /// Set by an observer (test or metrics surface) for the lifetime of
    /// one generation; read by `SpeculativeGeneration.generate`. nil ⇒
    /// no one is listening, the publish call is a cheap no-op.
    @TaskLocal public static var observer: (any SpeculativeAcceptanceObserver)?
}
