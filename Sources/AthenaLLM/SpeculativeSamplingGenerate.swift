import AthenaModels
import Foundation
import MLX
import MLXLMCommon

// MLX-aware speculative-SAMPLING decode loop (M40.2). Same MTP
// draft / single-pass verify / accept structure as the greedy
// `SpeculativeGeneration.generate`, but per-position acceptance is
// the Leviathan/Chen probabilistic test instead of an argmax equality
// check, and a rejection resamples from `normalize(max(0, p - q))`
// rather than collapsing to the target's argmax.
//
// Correctness invariant: at a fixed seed + same prompt + same
// temperature/top_p, the emitted ids are IDENTICAL to non-speculative
// sampling under the same params — speculation is a speedup, not a
// quality knob. (At temperature == 0 the path is unused; the greedy
// `SpeculativeGeneration` keeps that contract bit-identically.)

extension SpeculativeSampling {

    /// Sampling-mode speculative decoding over the MTP draft head and
    /// the backbone target. Mirrors `SpeculativeGeneration.generate`
    /// shape for shape — same prefill, same single-pass verify with
    /// `nConfirmed=1`, same GDN/Mamba rollback on rejection — but the
    /// acceptance gate and the post-reject token sampling are the
    /// Leviathan/Chen probabilistic versions. Returns the full id
    /// sequence; the caller decodes.
    ///
    /// Pre: `temperature > 0` (use `SpeculativeGeneration.generate`
    /// for the greedy path; the bit-identical contract lives there).
    public static func generate(
        model: AthenaQwen35Model,
        promptTokens: [Int],
        maxTokens: Int,
        eosTokenId: Int?,
        temperature: Float,
        topP: Float?,
        seed: Int?
    ) -> [Int] {
        precondition(
            temperature > 0,
            "speculative SAMPLING requires temperature > 0; the greedy "
                + "path is SpeculativeGeneration.generate")
        let vocab = model.vocabularySize
        let backbone = model.newCache(parameters: nil)
        let mtpCacheArr = model.makeMTPCache()
        let mtpCache: [KVCache?] = mtpCacheArr.map { $0 }

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

        var rng = SamplingRNG(seed: seed)

        // --- Prefill ---------------------------------------------------
        // Same chunked prefill as the greedy speculative path so the
        // hidden state at the last prompt token is identical to that
        // path's — only the per-step sampling differs.
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
            SpeculativeGeneration.tokenArray(promptTokens.last!),
            cache: backbone)
        asyncEval(backbone)

        var out: [Int] = []
        func commit(_ t: Int) -> Bool {
            if t == eosTokenId { return true }
            out.append(t)
            return out.count >= maxTokens
        }

        // First emitted token: sample from the target at the last prompt
        // position. Acceptance has nothing to compare against yet, so
        // this is just a plain target draw — identical to the first
        // step of non-speculative sampling.
        let p0 = SpeculativeSampling.distribution(
            logits: lastPositionFloat(logits, vocab: vocab),
            temperature: temperature, topP: topP)
        let prev0 = SpeculativeSampling.sampleFromDistribution(p0, rng: &rng)
        if commit(prev0) { return out }
        var prev = prev0

        // First draft from the MTP head: predict token t+2 from the
        // backbone PRE-norm hidden at t and the just-sampled token t+1.
        // Falls back to the target's last-position logits if the model
        // has no MTP head (the caller should have ruled this out).
        var draftLogits =
            model.mtpForward(
                hidden: hidden,
                nextTokenIds: SpeculativeGeneration.tokenArray(prev),
                mtpCache: mtpCache)
            ?? logits
        var q = SpeculativeSampling.distribution(
            logits: lastPositionFloat(draftLogits, vocab: vocab),
            temperature: temperature, topP: topP)
        var draft = SpeculativeSampling.sampleFromDistribution(q, rng: &rng)

        // --- Draft / single-pass verify / accept-or-reject ------------
        while out.count < maxTokens {
            let inp = MLXArray([Int32(prev), Int32(draft)], [1, 2])
            let (logits2, hidden2) = model.logitsAndHidden(
                inp, cache: backbone, nConfirmed: 1, rollback: rollback)
            let verifySlice = logits2[0..., 0, 0...]
            let bonusSlice = logits2[0..., 1, 0...]
            let hiddenC = hidden2[0..., 0 ..< 1, 0...]
            let hiddenD = hidden2[0..., 1 ..< 2, 0...]
            asyncEval(backbone)

            // p is the target's truncated distribution at the same
            // position the draft sampled from (post-confirmed `prev`,
            // pre-commit of `draft`). It MUST share the same temp+top_p
            // truncation as q — asymmetric truncation breaks the
            // no-quality-loss guarantee.
            let p = SpeculativeSampling.distribution(
                logits: sliceToFloat(verifySlice, vocab: vocab),
                temperature: temperature, topP: topP)
            let accepted = SpeculativeSampling.acceptOrReject(
                token: draft, pDistribution: p, qDistribution: q,
                rng: &rng)

            if accepted {
                if commit(draft) { break }
                // Bonus token: a plain target draw at position +1
                // (no draft, no acceptance test — the verify-pass
                // already paid for it).
                let pBonus = SpeculativeSampling.distribution(
                    logits: sliceToFloat(bonusSlice, vocab: vocab),
                    temperature: temperature, topP: topP)
                let bonus = SpeculativeSampling.sampleFromDistribution(
                    pBonus, rng: &rng)
                if commit(bonus) { break }
                prev = bonus
                let next =
                    model.mtpForward(
                        hidden: hiddenD,
                        nextTokenIds: SpeculativeGeneration.tokenArray(
                            bonus), mtpCache: mtpCache) ?? bonusSlice
                q = SpeculativeSampling.distribution(
                    logits: lastPositionFloat(next, vocab: vocab),
                    temperature: temperature, topP: topP)
                draft = SpeculativeSampling.sampleFromDistribution(
                    q, rng: &rng)
            } else {
                // Reject: drop the draft position from the KV trail,
                // restore the GDN/Mamba recurrent state to the
                // post-confirmed snapshot, then sample the corrected
                // token from the residual `normalize(max(0, p - q))`.
                trimKV()
                restoreMambaFromRollback()
                let r = SpeculativeSampling.residual(p: p, q: q)
                let resid = SpeculativeSampling.sampleFromDistribution(
                    r, rng: &rng)
                if commit(resid) { break }
                prev = resid
                let next =
                    model.mtpForward(
                        hidden: hiddenC,
                        nextTokenIds: SpeculativeGeneration.tokenArray(
                            resid), mtpCache: mtpCache) ?? verifySlice
                q = SpeculativeSampling.distribution(
                    logits: lastPositionFloat(next, vocab: vocab),
                    temperature: temperature, topP: topP)
                draft = SpeculativeSampling.sampleFromDistribution(
                    q, rng: &rng)
            }

            if out.count % 256 == 0 { MLX.Memory.clearCache() }
        }
        return out
    }

    /// Copy the last position of a `(1, S, vocab)` (or `(1, vocab)`
    /// when S==1) MLX array into a fp32 Swift array — the sampling
    /// math runs on plain `[Float]` so it stays MLX-free and testable.
    private static func lastPositionFloat(
        _ array: MLXArray, vocab: Int
    ) -> [Float] {
        let slice =
            array.ndim == 3 ? array[0..., -1, 0...] : array
        return sliceToFloat(slice, vocab: vocab)
    }

    /// Copy a `(1, vocab)`-shaped MLX slice into a fp32 Swift array.
    private static func sliceToFloat(
        _ slice: MLXArray, vocab: Int
    ) -> [Float] {
        let flat = slice.asType(.float32).reshaped(vocab)
        flat.eval()
        return flat.asArray(Float.self)
    }
}
