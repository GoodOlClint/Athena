import Foundation

/// The MLX-free decision algebra of the greedy TDT decode loop (ADR 020 S4,
/// ADR 009 stub-tier). The numerics (encoder/joint/argmax) stay in
/// `ParakeetModel`; the *control-flow* — how the encoder frame pointer advances
/// and the anti-stall guard fires — is pure integer logic, so it is unit-pinned
/// under `swift test`.
public enum ParakeetDecodeStep {
    /// Advance the encoder frame pointer after one emission.
    ///
    /// A TDT step predicts a duration (0…4 frames). A non-zero duration moves
    /// the pointer forward by that many frames and clears the consecutive-
    /// zero-duration counter. A zero duration keeps the pointer (the model can
    /// emit several tokens at one frame) — but after `maxSymbols` consecutive
    /// zero-duration emissions the guard forces the pointer forward by one so a
    /// pathological loop can't stall the decode forever.
    ///
    /// - Parameters:
    ///   - duration: frames predicted this step (`durations[decision]`).
    ///   - priorNewSymbols: consecutive zero-duration count *before* this step.
    ///   - maxSymbols: the anti-stall cap.
    /// - Returns: `stepDelta` to add to the frame pointer (duration + any
    ///   anti-stall bump) and the updated consecutive-zero-duration counter.
    public static func advance(
        duration: Int, priorNewSymbols: Int, maxSymbols: Int
    ) -> (stepDelta: Int, newSymbols: Int) {
        if duration != 0 { return (duration, 0) }
        let bumped = priorNewSymbols + 1
        if bumped >= maxSymbols { return (duration + 1, 0) }
        return (duration, bumped)
    }
}
