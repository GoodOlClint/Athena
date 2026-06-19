import Foundation

/// Pure relative-silence window gate for embedding-based speaker diarization
/// (M25.3). Given per-window RMS energies, returns the indices of the windows
/// loud enough to embed — keeping windows at or above `gateFraction` of the
/// loudest. MLX-free + Sendable, so the gate decision is unit-testable on
/// synthetic energies (ND15) without an embedding model.
///
/// Contract (behavior-preserving extraction from `MLXSpeakerEmbeddingModule`):
/// - A window is kept unless the audio carries energy (`maxRMS > 0`) AND the
///   window sits below the gate (`rms[i] < maxRMS * gateFraction`).
/// - Uniformly silent audio (`maxRMS == 0`) keeps every window.
/// - If the gate would drop everything, fall back to keeping all windows —
///   a non-empty input never yields an empty kept set.
/// - Kept indices preserve input (ascending-start) order.
public enum SpeakerWindowGate {
    public static func keptIndices(
        rms: [Float], gateFraction: Float = 0.20
    ) -> [Int] {
        let maxRMS = rms.max() ?? 0
        let gate = maxRMS * gateFraction
        var kept: [Int] = []
        kept.reserveCapacity(rms.count)
        for idx in rms.indices {
            if maxRMS > 0, rms[idx] < gate { continue }
            kept.append(idx)
        }
        // Gate emptied everything (or no windows) → keep all (D-fallback).
        if kept.isEmpty { return Array(rms.indices) }
        return kept
    }
}
