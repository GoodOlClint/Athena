import AthenaCore
import AthenaModels
import AthenaStructured
import Dispatch
import Foundation
import Logging
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
        // M53 diag — speculative acceptance + throughput summary.
        let genStart = Date()
        var specIters = 0
        var specAccepts = 0
        // PERF TRACE (env-gated, ATHENA_PERF_TRACE=1) — attribute per-token
        // wall time to stages. Forces eval() at stage boundaries so MLX's
        // lazy graph doesn't fold the backbone-forward cost into the
        // sampling .item() sync. Adds sync overhead, so it perturbs the
        // absolute rate slightly; the STAGE RATIOS are the signal.
        let trace = ProcessInfo.processInfo.environment["ATHENA_PERF_TRACE"] == "1"
        ForwardProfile.reset()
        var tBackbone = 0.0, tVerifySample = 0.0, tMTP = 0.0
        var tDraftPick = 0.0, tRollback = 0.0, tPrefill = 0.0
        @inline(__always) func now() -> UInt64 { DispatchTime.now().uptimeNanoseconds }
        @inline(__always) func dt(_ t0: UInt64) -> Double { Double(now() - t0) / 1e9 }
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
        // M48.4 — publish per-chunk prefill progress so the heartbeat
        // can distinguish "stuck in prefill" from "decoding slowly."
        let tPrefill0 = trace ? now() : 0
        if promptTokens.count > 1 {
            let head = Array(promptTokens.dropLast())
            let chunkSize = 512
            let totalChunks =
                (head.count + chunkSize - 1) / chunkSize
            var i = 0
            var done = 0
            while i < head.count {
                let chunk = Array(
                    head[i ..< min(i + chunkSize, head.count)])
                _ = model(
                    MLXArray(chunk.map { Int32($0) }, [1, chunk.count]),
                    cache: backbone)
                asyncEval(backbone)
                i += chunkSize
                done += 1
                DecodeProgress.counter?.recordPrefillChunk(
                    completed: done, total: totalChunks)
            }
        }
        (logits, hidden) = model.logitsAndHidden(
            tokenArray(promptTokens.last!), cache: backbone)
        asyncEval(backbone)
        if trace { eval(logits, hidden); tPrefill = dt(tPrefill0) }

        var out: [Int] = []
        // Deferred-enforcement phase machine (IDLE→ENFORCING). The Guide
        // only ever advances over COMMITTED tokens and never un-commits,
        // so its state is monotonic — NO rollback (mlx-lm's
        // _RollbackingLogitsProcessor exists only because a
        // LogitProcessor can't control commits; this loop can).
        var decoder = GuidedDecoder(
            guide: guide, vocab: vocab,
            idleBudget: max(8, maxTokens / 2))
        // Index in `out` where the enforced JSON span begins (the
        // unconstrained <think>/tool prefix before it is dropped from
        // the structured response). nil ⇒ never started.
        var jsonStart: Int?
        func commit(_ t: Int) -> Bool {
            if t == eosTokenId {
                // Model tried to stop in IDLE: force enforcement instead
                // of returning empty (don't append/stop on the EOS).
                if guide != nil, !decoder.enforcing {
                    decoder.forceEnforce()
                    return false
                }
                return true
            }
            out.append(t)
            // M46.8 — per-token progress for the heartbeat. Speculative
            // can commit 0, 1, or 2 tokens per iteration (reject /
            // accept-no-bonus / accept-with-bonus); incrementing inside
            // `commit()` captures every committed token exactly once,
            // matching the heartbeat's "tokens emitted" semantics.
            DecodeProgress.counter?.incrementToken()
            if case .jsonStart = decoder.commit(t) {
                jsonStart = out.count - 1
            }
            return out.count >= maxTokens
        }
        func result() -> [Int] {
            guide == nil ? out : Array(out[(jsonStart ?? out.count)...])
        }

        // First token — IDLE plain argmax until JSON starts.
        let prev0 = decoder.pick(logits[0..., -1, 0...])
        if commit(prev0) { return result() }
        var prev = prev0

        // M47.2 — MTP drafts are Guide-masked at the SAME Guide state the
        // upcoming verify will use, so the draft is drawn from the
        // schema-allowed set instead of the full vocabulary. Under tight
        // schemas (~150k vocab, ≤10 valid tokens per position) the
        // unmasked draft almost never matched the masked verify ⇒ ~100%
        // reject, every iteration paid an MTP forward + KV trim + Mamba
        // rollback for nothing. Bit-identical-greedy contract is
        // unchanged: the verify gate (`commit(draft)` only runs when
        // `draft == verifyPred`, AND `verifyPred` is the masked argmax
        // of the backbone) still decides what gets committed; the draft
        // mask only changes WHICH token is proposed for verification.
        // `decoder.pick(_:)` is non-state-advancing (only `commit(_:)`
        // advances the Guide), so calling it for draft and verify at
        // the same logical position is safe.
        var draft = decoder.pick(
            (model.mtpForward(
                hidden: hidden, nextTokenIds: tokenArray(prev),
                mtpCache: mtpCache) ?? logits)[0..., -1, 0...])

        // --- Greedy draft/verify/accept (single-pass verify) ---------
        // ONE backbone forward over [confirmed, draft] with nConfirmed=1;
        // pos0 = prediction after the confirmed token, pos1 = after the
        // draft. Each committed pick is Guide-masked, so the emitted
        // sequence is exactly Guide-masked greedy (speculation only
        // accelerates it).
        while out.count < maxTokens {
            let inp = MLXArray([Int32(prev), Int32(draft)], [1, 2])
            var t0 = trace ? now() : 0
            let (logits2, hidden2) = model.logitsAndHidden(
                inp, cache: backbone, nConfirmed: 1, rollback: rollback)
            if trace { eval(logits2, hidden2); tBackbone += dt(t0); t0 = now() }
            // verifyPred under the Guide state BEFORE this step's commits.
            let verifyPred = decoder.pick(logits2[0..., 0, 0...])
            let hiddenC = hidden2[0..., 0 ..< 1, 0...]
            let hiddenD = hidden2[0..., 1 ..< 2, 0...]
            if trace { tVerifySample += dt(t0) }
            asyncEval(backbone)

            // M47.2 — publish accept/reject so the test (and any future
            // perf-observability surface) can read the acceptance rate
            // via the SpeculativeStats TaskLocal.
            SpeculativeStats.observer?
                .recordIteration(accepted: draft == verifyPred)
            specIters += 1
            if draft == verifyPred { specAccepts += 1 }

            if draft == verifyPred {
                if commit(draft) { break }  // advances the Guide
                // bonus under the Guide state AFTER committing draft.
                let bonus = decoder.pick(logits2[0..., 1, 0...])
                if commit(bonus) { break }
                prev = bonus
                // M47.2 — draft masked under the post-bonus Guide state
                // (same state the next iteration's verify will use).
                t0 = trace ? now() : 0
                let mtpL = model.mtpForward(
                    hidden: hiddenD, nextTokenIds: tokenArray(bonus),
                    mtpCache: mtpCache) ?? logits2
                if trace { eval(mtpL); tMTP += dt(t0); t0 = now() }
                draft = decoder.pick(mtpL[0..., -1, 0...])
                if trace { tDraftPick += dt(t0) }
            } else {
                // Reject: drop the draft position — KV trim(1), Mamba
                // restore to the post-confirmed snapshot.
                t0 = trace ? now() : 0
                trimKV()
                restoreMambaFromRollback()
                if trace { eval(backbone); tRollback += dt(t0) }
                if commit(verifyPred) { break }
                prev = verifyPred
                // M47.2 — draft masked under the post-verifyPred Guide
                // state (same state the next iteration's verify will use).
                t0 = trace ? now() : 0
                let mtpL = model.mtpForward(
                    hidden: hiddenC, nextTokenIds: tokenArray(verifyPred),
                    mtpCache: mtpCache) ?? logits2
                if trace { eval(mtpL); tMTP += dt(t0); t0 = now() }
                draft = decoder.pick(mtpL[0..., -1, 0...])
                if trace { tDraftPick += dt(t0) }
            }

            if out.count % 256 == 0 { MLX.Memory.clearCache() }
        }
        let secs = Date().timeIntervalSince(genStart)
        let rate = specIters > 0 ? Double(specAccepts) / Double(specIters) : 0
        MLXLLMModule.log.notice(
            """
            speculative summary: tokens=\(out.count) \
            tok/s=\(String(format: "%.1f", Double(out.count) / max(secs, 0.001))) \
            spec_iters=\(specIters) accepted=\(specAccepts) \
            accept_rate=\(String(format: "%.2f", rate)) \
            guided=\(guide != nil)
            """,
            metadata: ["function": "SpeculativeGeneration.generate"])
        if trace {
            let decodeSecs = max(secs - tPrefill, 0.001)
            let f = { (x: Double) in String(format: "%.3f", x) }
            let pct = { (x: Double) in String(format: "%.1f%%", x / decodeSecs * 100) }
            MLXLLMModule.log.notice(
                """
                perf trace: total=\(f(secs))s prefill=\(f(tPrefill))s \
                decode=\(f(decodeSecs))s | backbone=\(f(tBackbone))s(\(pct(tBackbone))) \
                verifySample=\(f(tVerifySample))s(\(pct(tVerifySample))) \
                mtp=\(f(tMTP))s(\(pct(tMTP))) draftPick=\(f(tDraftPick))s(\(pct(tDraftPick))) \
                rollback=\(f(tRollback))s(\(pct(tRollback))) \
                per_iter_ms=\(f(decodeSecs / Double(max(specIters,1)) * 1000)) \
                iters=\(specIters)
                """,
                metadata: ["function": "SpeculativeGeneration.generate"])
        }
        if ForwardProfile.enabled {
            MLXLLMModule.log.notice(
                "\(ForwardProfile.summary())",
                metadata: ["function": "SpeculativeGeneration.generate"])
        }
        return result()
    }
}
