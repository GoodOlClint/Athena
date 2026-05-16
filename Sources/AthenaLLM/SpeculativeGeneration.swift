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

    private static func argmaxAt(_ logits: MLXArray, _ pos: Int) -> Int {
        // logits: (1, S, vocab) → position `pos` → argmax token id.
        argMax(logits[0..., pos, 0...], axis: -1).item(Int.self)
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

        // Reject undoes the single 2-token verify: KVCacheSimple
        // (attention) layers trim the draft position; GatedDelta/Mamba
        // layers restore the post-confirmed recurrent state the GDN
        // n_confirmed split stashed on AthenaMambaCache.rollbackState.
        let rollback = GDNRollback()
        func trimKV() {
            for c in backbone where c.isTrimmable { _ = c.trim(1) }
        }
        func restoreMambaFromRollback() {
            for c in backbone {
                if let mc = c as? MambaCache,
                    let rb = rollback.states[ObjectIdentifier(mc)]
                {
                    mc.state = rb.map { $0[.ellipsis] }
                }
            }
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
        // Emit a token unless it is EOS (non-speculative greedy stops at
        // EOS WITHOUT surfacing it — to stay bit-identical we must not
        // append the EOS id). Returns true when generation should stop.
        func appendOrStop(_ t: Int) -> Bool {
            if t == eosTokenId { return true }
            out.append(t)
            return out.count >= maxTokens
        }

        let prev0 = argmaxLast(logits)  // first generated (main) token
        if appendOrStop(prev0) { return out }
        var prev = prev0

        var draft = argmaxLast(
            model.mtpForward(
                hidden: hidden, nextTokenIds: tokenArray(prev),
                mtpCache: mtpCache) ?? logits)

        // --- Greedy draft/verify/accept (single-pass verify) ---------
        // ONE backbone forward over [confirmed, draft] with nConfirmed=1:
        // attention/MLP run once over both positions (the speedup); the
        // GDN layers internally split + stash the post-confirmed state.
        // pos0 logits = prediction after the confirmed token
        // (== non-spec's next token); pos1 logits = prediction after the
        // draft (the bonus). Bit-identical to non-spec greedy by the
        // cached-decode state-carry invariant.
        while out.count < maxTokens {
            let inp = MLXArray([Int32(prev), Int32(draft)], [1, 2])
            let (logits2, hidden2) = model.logitsAndHidden(
                inp, cache: backbone, nConfirmed: 1, rollback: rollback)
            let verifyPred = argmaxAt(logits2, 0)
            let bonus = argmaxAt(logits2, 1)
            let hiddenC = hidden2[0..., 0 ..< 1, 0...]
            let hiddenD = hidden2[0..., 1 ..< 2, 0...]
            asyncEval(backbone)

            if draft == verifyPred {
                // Accept: caches already hold [confirmed, draft] (final).
                if appendOrStop(draft) { break }
                if appendOrStop(bonus) { break }
                prev = bonus
                draft = argmaxLast(
                    model.mtpForward(
                        hidden: hiddenD, nextTokenIds: tokenArray(bonus),
                        mtpCache: mtpCache) ?? logits2)
            } else {
                // Reject: drop the draft position — KV trim(1), Mamba
                // restore to the post-confirmed snapshot.
                trimKV()
                restoreMambaFromRollback()
                if appendOrStop(verifyPred) { break }
                prev = verifyPred
                draft = argmaxLast(
                    model.mtpForward(
                        hidden: hiddenC, nextTokenIds: tokenArray(verifyPred),
                        mtpCache: mtpCache) ?? logits2)
            }

            if out.count % 256 == 0 { MLX.Memory.clearCache() }
        }
        return out
    }
}
