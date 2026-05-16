import AthenaModels
import Foundation
import MLX
import MLXLMCommon

/// Greedy MTP speculative decoding (temperature 0 only — the bench-validated
/// regime; temp>0 residual sampling is the deferred named risk, M2.3).
///
/// Correctness invariant: the emitted token sequence MUST be bit-identical
/// to non-speculative greedy decoding of the same model/prompt — MTP only
/// changes speed. The draft/verify/accept structure and the exact
/// per-token mechanics (input shape, `argMax(axis:-1)`, in-place cache
/// mutation) mirror the substrate `TokenIterator` so the invariant holds;
/// the only subtle point is restoring recurrent (GatedDeltaNet/Mamba)
/// cache state on draft rejection — done with an exact deep-copy snapshot.
enum SpeculativeGeneration {

    private static func argmaxLast(_ logits: MLXArray) -> Int {
        // logits: (1, S, vocab) → last position → argmax token id.
        let last = logits[0..., -1, 0...]
        return argMax(last, axis: -1).item(Int.self)
    }

    private static func tokenArray(_ id: Int) -> MLXArray {
        MLXArray([Int32(id)], [1, 1])
    }

    /// Run the loop. `promptTokens` are post-chat-template ids. Returns the
    /// full generated token ids (decoded by the caller) — non-streaming so
    /// the bit-identical comparison is unambiguous.
    static func generate(
        model: AthenaQwen35Model,
        promptTokens: [Int],
        maxTokens: Int,
        eosTokenId: Int?
    ) -> [Int] {
        let backbone = model.newCache(parameters: nil)
        let mtpCacheArr = model.makeMTPCache()
        let mtpCache: [KVCache?] = mtpCacheArr.map { $0 }

        // Mamba (GatedDeltaNet) layers need snapshot/restore on reject;
        // KVCacheSimple layers are trimmable. Record indices once.
        let mambaIdx = backbone.enumerated().compactMap {
            $0.element is MambaCache ? $0.offset : nil
        }

        func snapshotMamba() -> [Int: MambaCache] {
            var s: [Int: MambaCache] = [:]
            for i in mambaIdx {
                s[i] = (backbone[i].copy() as! MambaCache)
            }
            return s
        }
        func restoreMamba(_ snap: [Int: MambaCache]) {
            for (i, m) in snap {
                let live = backbone[i] as! MambaCache
                live.state = m.state.map { $0[.ellipsis] }
                live.offset = m.offset
            }
        }
        func trimKV() {
            for c in backbone where c.isTrimmable { _ = c.trim(1) }
        }

        // --- Prefill --------------------------------------------------
        var hidden: MLXArray
        var logits: MLXArray
        if promptTokens.count > 1 {
            let head = Array(promptTokens.dropLast())
            var i = 0
            while i < head.count {
                let chunk = Array(head[i ..< min(i + 512, head.count)])
                _ = model(
                    MLXArray(chunk.map { Int32($0) }, [1, chunk.count]),
                    cache: backbone)
                asyncEval(backbone)
                i += 512
            }
        }
        (logits, hidden) = model.logitsAndHidden(
            tokenArray(promptTokens.last!), cache: backbone)
        asyncEval(backbone)

        var out: [Int] = []
        var prev = argmaxLast(logits)  // first generated (main) token
        out.append(prev)
        if prev == eosTokenId { return out }

        var draft = argmaxLast(
            model.mtpForward(
                hidden: hidden, nextTokenIds: tokenArray(prev),
                mtpCache: mtpCache) ?? logits)

        // --- Greedy draft/verify/accept ------------------------------
        while out.count < maxTokens {
            // 1. Confirmed forward (the token we already committed).
            let (logitsC, hiddenC) = model.logitsAndHidden(
                tokenArray(prev), cache: backbone)
            let verifyPred = argmaxLast(logitsC)

            // 2. Snapshot post-confirmed recurrent state, then draft.
            let snap = snapshotMamba()
            let (logitsD, hiddenD) = model.logitsAndHidden(
                tokenArray(draft), cache: backbone)
            let bonus = argmaxLast(logitsD)
            asyncEval(backbone)

            if draft == verifyPred {
                // Accept: draft is exactly non-spec's next token.
                out.append(draft)
                if draft == eosTokenId || out.count >= maxTokens { break }
                out.append(bonus)
                if bonus == eosTokenId || out.count >= maxTokens { break }
                prev = bonus
                draft = argmaxLast(
                    model.mtpForward(
                        hidden: hiddenD, nextTokenIds: tokenArray(bonus),
                        mtpCache: mtpCache) ?? logitsD)
            } else {
                // Reject: undo the draft forward, emit the true token.
                trimKV()
                restoreMamba(snap)
                out.append(verifyPred)
                if verifyPred == eosTokenId { break }
                prev = verifyPred
                draft = argmaxLast(
                    model.mtpForward(
                        hidden: hiddenC, nextTokenIds: tokenArray(verifyPred),
                        mtpCache: mtpCache) ?? logitsC)
            }

            if out.count % 256 == 0 { MLX.Memory.clearCache() }
        }
        return out
    }
}
