import AthenaModels
import AthenaStructured
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

    /// Argmax of `slice` (shape (1, vocab)) under the structured Guide:
    /// add −inf to every token the Guide disallows in its current state,
    /// so the greedy pick is always schema-valid. No guide ⇒ plain argmax.
    static func guidedArgmax(
        _ slice: MLXArray, vocab: Int,
        guide: StructuredGuide?, maskBuf: inout [UInt8]
    ) -> Int {
        guard let guide else {
            return argMax(slice, axis: -1).item(Int.self)
        }
        _ = guide.allowedMask(into: &maskBuf)
        var add = [Float](repeating: -.infinity, count: vocab)
        for i in 0..<vocab where (maskBuf[i >> 3] >> UInt8(i & 7)) & 1 == 1 {
            add[i] = 0
        }
        let masked = slice + MLXArray(add)
        return argMax(masked, axis: -1).item(Int.self)
    }

    static func tokenArray(_ id: Int) -> MLXArray {
        MLXArray([Int32(id)], [1, 1])
    }

    /// Run the loop. `promptTokens` are post-chat-template ids. Returns the
    /// full generated token ids (decoded by the caller) — non-streaming so
    /// the bit-identical comparison is unambiguous.
    static func generate(
        model: AthenaQwen35Model,
        promptTokens: [Int],
        maxTokens: Int,
        eosTokenId: Int?,
        guide: StructuredGuide? = nil
    ) -> [Int] {
        let vocab = model.vocabularySize
        var maskBuf = [UInt8]()
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
        // Commit a token: stop (no append/advance) at EOS — non-spec
        // greedy stops at EOS without surfacing it. The structured Guide
        // only ever advances over COMMITTED tokens, so its state is
        // monotonic and needs NO rollback: mlx-lm's
        // _RollbackingLogitsProcessor exists only because a LogitProcessor
        // can't control commits — this hand-rolled loop can, so the
        // named no-rejection-callback risk is avoided here by construction.
        func commit(_ t: Int) -> Bool {
            if t == eosTokenId { return true }
            out.append(t)
            guide?.advance(UInt32(t))
            return out.count >= maxTokens
        }

        // First (main) token — Guide-masked greedy.
        let prev0 = guidedArgmax(
            logits[0..., -1, 0...], vocab: vocab, guide: guide,
            maskBuf: &maskBuf)
        if commit(prev0) { return out }
        var prev = prev0

        // MTP drafts are NOT masked; a guide-invalid draft simply fails
        // the masked verify and is rejected.
        var draft = argmaxLast(
            model.mtpForward(
                hidden: hidden, nextTokenIds: tokenArray(prev),
                mtpCache: mtpCache) ?? logits)

        // --- Greedy draft/verify/accept (single-pass verify) ---------
        // ONE backbone forward over [confirmed, draft] with nConfirmed=1;
        // pos0 = prediction after the confirmed token, pos1 = after the
        // draft. Each committed pick is Guide-masked, so the emitted
        // sequence is exactly Guide-masked greedy (speculation only
        // accelerates it).
        while out.count < maxTokens {
            let inp = MLXArray([Int32(prev), Int32(draft)], [1, 2])
            let (logits2, hidden2) = model.logitsAndHidden(
                inp, cache: backbone, nConfirmed: 1, rollback: rollback)
            // verifyPred under the Guide state BEFORE this step's commits.
            let verifyPred = guidedArgmax(
                logits2[0..., 0, 0...], vocab: vocab, guide: guide,
                maskBuf: &maskBuf)
            let hiddenC = hidden2[0..., 0 ..< 1, 0...]
            let hiddenD = hidden2[0..., 1 ..< 2, 0...]
            asyncEval(backbone)

            if draft == verifyPred {
                if commit(draft) { break }  // advances the Guide
                // bonus under the Guide state AFTER committing draft.
                let bonus = guidedArgmax(
                    logits2[0..., 1, 0...], vocab: vocab, guide: guide,
                    maskBuf: &maskBuf)
                if commit(bonus) { break }
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
                if commit(verifyPred) { break }
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
