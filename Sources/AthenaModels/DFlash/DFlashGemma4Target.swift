import Foundation
import MLX
import MLXLLM
import MLXLMCommon

// Gemma4 target adapter for DFlash (M63.2). Wraps the substrate
// `Gemma4TextModel` capture seam (`callReturningHidden`) and the cache
// rollback primitive so the M63.3 decode engine can: (1) run prefill/verify
// forwards that return both the target logits and the per-layer hidden
// states the draft conditions on, (2) build the concatenated context
// feature in the drafter's `target_layer_ids` order, and (3) roll the KV
// caches back by the rejected-tail length after a verify block.
//
// Gemma4 needs only KV-length rollback — no recurrent (GDN) state — so
// rollback is a per-cache trim: full-attention `KVCacheSimple` rewinds its
// offset; sliding `RotatingKVCache` uses the window-wrap-aware `trimRecent`.
// (`engine/target_gemma4.py`: `supports_recurrent_rollback=False`,
// `supports_kv_trim=True`.)
/// Capability both substrate Gemma4 load shapes expose: the bare
/// `Gemma4TextModel` (a `gemma4_text` checkpoint) and the multimodal
/// `Gemma4Model` wrapper (a `gemma4` checkpoint with a vision tower, e.g.
/// gemma-4-31b-it). Both gained `callReturningHidden` in the M63.2 substrate
/// delta; `newCache` is pre-existing.
public protocol DFlashGemma4Backbone: AnyObject {
    func callReturningHidden(
        _ inputs: MLXArray, cache: [KVCache]?, captureLayers: Set<Int>
    ) -> (logits: MLXArray, hidden: [Int: MLXArray])
    func newCache(parameters: GenerateParameters?) -> [any KVCache]
}

extension Gemma4Model: DFlashGemma4Backbone {}
extension Gemma4TextModel: DFlashGemma4Backbone {}

public enum DFlashGemma4Target {

    /// Concatenate the captured hidden states for `layerOrder` along the
    /// feature axis, producing the draft context feature `(B, seq,
    /// layerOrder.count * H)`. Order MUST match the drafter's
    /// `target_layer_ids` (it indexes the draft `fc` projection).
    public static func contextFeature(
        from captured: [Int: MLXArray], layerOrder: [Int]
    ) -> MLXArray {
        precondition(!layerOrder.isEmpty, "DFlash target layer order is empty")
        let slices = layerOrder.map { layer -> MLXArray in
            guard let h = captured[layer] else {
                preconditionFailure(
                    "DFlash capture missing target layer \(layer) — "
                        + "captured: \(captured.keys.sorted())")
            }
            return h
        }
        return slices.count == 1 ? slices[0] : concatenated(slices, axis: -1)
    }

    /// One target forward over `tokens` (B, L) that returns the softcapped
    /// logits (B, L, vocab) and the concatenated context feature (B, L,
    /// k*H) for the draft. Advances `cache` by L positions.
    public static func forward(
        model: DFlashGemma4Backbone,
        tokens: MLXArray,
        cache: [KVCache],
        layerOrder: [Int]
    ) -> (logits: MLXArray, context: MLXArray) {
        let (logits, captured) = model.callReturningHidden(
            tokens, cache: cache, captureLayers: Set(layerOrder))
        return (logits, contextFeature(from: captured, layerOrder: layerOrder))
    }

    /// Roll the target KV caches back by `n` positions (the rejected tail of
    /// a verify block). Sliding `RotatingKVCache` uses the wrap-aware
    /// `trimRecent`; every other trimmable cache uses `trim`.
    public static func rollback(_ caches: [KVCache], by n: Int) {
        guard n > 0 else { return }
        for cache in caches {
            if let rotating = cache as? RotatingKVCache {
                rotating.trimRecent(n)
            } else if cache.isTrimmable {
                _ = cache.trim(n)
            }
        }
    }
}
