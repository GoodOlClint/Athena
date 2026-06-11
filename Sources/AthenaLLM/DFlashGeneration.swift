import AthenaCore
import AthenaModels
import Foundation
import MLX
import MLXLMCommon

/// DFlash lossless speculative decoding for attention-only targets
/// (Gemma4-first). A small block-diffusion drafter, conditioned on the
/// target's captured per-layer hidden states, proposes a block of tokens in
/// one pass; the target verifies the block in one forward and accepts the
/// longest prefix that matches its own argmax. Every emitted token equals
/// the target's argmax under the verify (block) forward, so the output is the
/// target's block-forward greedy sequence — lossless. It matches Athena's
/// single-token greedy except at rare SDPA-kernel near-ties (MLX dispatches a
/// different kernel for n=1 vs n>=2, which round differently); see ADR 001 §3
/// and DFlashEngineParityTests. Ported
/// from bstnxbt/dflash-mlx (`engine/spec_epoch.py` core cycle +
/// `draft_backend.py`); see `Sources/AthenaModels/DFlash/NOTICE`.
///
/// M63.3 scope: the **unguided** greedy cycle. It uses the no-cache draft
/// forward (the draft recomputes the full projected context each cycle —
/// correctness-first; the windowed draft KV cache and the fast Metal kernel
/// are deferred perf follow-ups). Guide-masked structured output is M63.4.
enum DFlashGeneration {

    private static func argmaxIds(_ logits: MLXArray) -> [Int] {
        // logits (1, L, vocab) → (1, L) → [L]
        argMax(logits, axis: -1).reshaped(-1).asArray(Int32.self).map { Int($0) }
    }

    /// Greedy DFlash decode. `promptTokens` are post-chat-template ids.
    /// Returns the generated token ids (caller decodes). Non-streaming so the
    /// bit-identical A/B comparison is unambiguous.
    static func generate(
        target: DFlashGemma4Backbone,
        draft: DFlashDraftModel,
        promptTokens: [Int],
        maxTokens: Int,
        stopTokens: Set<Int>
    ) -> [Int] {
        precondition(!promptTokens.isEmpty, "DFlash requires a non-empty prompt")
        let layerOrder = draft.targetLayerIds
        let captureSet = Set(layerOrder)
        let K = max(2, draft.blockSize)
        let maskId = Int32(draft.maskTokenId)
        draft.bindTargetModel(embedScale: target.textEmbedScale)

        let cache = target.newCache(parameters: nil)

        // --- Prefill (chunked) -------------------------------------------
        var contextFeat: MLXArray? = nil
        var lastLogits: MLXArray? = nil
        let chunk = 512
        var i = 0
        while i < promptTokens.count {
            let end = min(i + chunk, promptTokens.count)
            let arr = MLXArray(
                promptTokens[i ..< end].map { Int32($0) }, [1, end - i])
            let (logits, captured) = target.callReturningHidden(
                arr, cache: cache, captureLayers: captureSet)
            let feat = DFlashGemma4Target.contextFeature(
                from: captured, layerOrder: layerOrder)
            contextFeat =
                contextFeat == nil ? feat : concatenated([contextFeat!, feat], axis: 1)
            lastLogits = logits
            asyncEval(cache)
            i = end
        }
        var ctx = contextFeat!
        var staged = argmaxIds(lastLogits![0..., (lastLogits!.dim(1) - 1)..., 0...])[0]

        var out: [Int] = []
        /// Append `t` unless it is a stop token; return true to stop (any
        /// stop token — matching the substrate's full EOS set, including
        /// e.g. Gemma's <end_of_turn> — or the token cap).
        func emit(_ t: Int) -> Bool {
            if stopTokens.contains(t) { return true }
            out.append(t)
            DecodeProgress.counter?.incrementToken()
            return out.count >= maxTokens
        }

        // --- Block draft / verify / accept -------------------------------
        let maskTail = Array(repeating: maskId, count: K - 1)
        while out.count < maxTokens {
            if DecodeProgress.counter?.isCancelled == true { break }

            // Draft: noise block [staged, mask×(K-1)] → target embed → draft
            // net → target head on the tail → greedy tail tokens.
            let blockIds = MLXArray([Int32(staged)] + maskTail, [1, K])
            let noiseEmb = target.embedTokens(blockIds)
            let draftHidden = draft(noiseEmbedding: noiseEmb, targetHidden: ctx)
            let tailLogits = target.logitsFromHidden(draftHidden[0..., 1..., 0...])
            let draftedTail = argmaxIds(tailLogits)  // K-1 ids

            // Verify the full candidate [staged, draftedTail] in one forward.
            let candidate = [staged] + draftedTail
            let candArr = MLXArray(candidate.map { Int32($0) }, [1, K])
            let (verifyLogits, verifyCaptured) = target.callReturningHidden(
                candArr, cache: cache, captureLayers: captureSet)
            let posterior = argmaxIds(verifyLogits)  // K ids

            // Longest prefix where the draft tail matches the target argmax.
            var acceptLen = 0
            while acceptLen < K - 1, draftedTail[acceptLen] == posterior[acceptLen] {
                acceptLen += 1
            }

            // Acceptance observability (reuses the M47 SpeculativeStats
            // TaskLocal): record each drafted tail position as accepted up to
            // the first reject. All-accept records K-1 accepts and no reject.
            if let obs = SpeculativeStats.observer {
                for j in 0 ..< (K - 1) {
                    let accepted = j < acceptLen
                    obs.recordIteration(accepted: accepted)
                    if !accepted { break }
                }
            }

            // Commit the staged token + the accepted tail.
            var stop = emit(staged)
            if !stop {
                for j in 0 ..< acceptLen where !stop { stop = emit(draftedTail[j]) }
            }
            if stop {
                // Roll the cache back to just the committed positions before
                // returning, so a reused target cache is left consistent.
                DFlashGemma4Target.rollback(cache, by: K - 1 - acceptLen)
                break
            }

            // Drop the rejected tail from the target KV cache, keeping the
            // 1 + acceptLen committed positions.
            DFlashGemma4Target.rollback(cache, by: K - 1 - acceptLen)
            // Next staged = the target argmax after the last accepted token.
            staged = posterior[acceptLen]
            // Extend the draft context with the committed positions' hiddens.
            let committed = verifyCaptured.mapValues {
                $0[0..., 0 ..< (acceptLen + 1), 0...]
            }
            let committedFeat = DFlashGemma4Target.contextFeature(
                from: committed, layerOrder: layerOrder)
            ctx = concatenated([ctx, committedFeat], axis: 1)
            asyncEval(cache)
            if out.count % 256 == 0 { MLX.Memory.clearCache() }
        }
        return out
    }
}
