import Foundation
import MLX

/// Norm-only TriAttention importance scoring (the reference's
/// `disable_trig=True` branch): `score(pos) = aggregate_heads ‖k_pos‖`.
/// No positions, no stats, no cross-layer coupling — so it is correct
/// applied per attention layer independently.
enum TriAttentionScorer {

    /// Per-position importance for one layer's cached keys.
    ///
    /// - Parameter keys: `[B, kvHeads, seqLen, headDim]` (post-RoPE, as
    ///   stored in the cache — norm-only scoring is rotation-invariant in
    ///   aggregate so no RoPE inversion is needed).
    /// - Returns: `seqLen` host-side scores (float32 reduction, matching
    ///   the reference `.astype(mx.float32)`), highest = most important.
    static func scoreKeys(
        _ keys: MLXArray, aggregation: TriAttentionConfig.ScoreAggregation
    ) -> [Float] {
        let k = keys.asType(.float32)
        // ‖k‖ per (B, head, pos): sqrt(Σ_d k_d^2 + 1e-8)
        let norms = sqrt((k * k).sum(axis: -1) + 1e-8)  // [B, kvHeads, seqLen]
        let perPos: MLXArray
        switch aggregation {
        case .mean:
            perPos = norms.mean(axis: 1)  // [B, seqLen]
        case .max:
            perPos = norms.max(axis: 1)  // [B, seqLen]
        }
        // Decode batch is 1; reduce any batch dim by mean defensively.
        let reduced = perPos.dim(0) == 1 ? perPos[0] : perPos.mean(axis: 0)
        reduced.eval()
        return reduced.asArray(Float.self)
    }
}
