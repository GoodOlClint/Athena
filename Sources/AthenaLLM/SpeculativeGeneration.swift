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
        // M70.3 (L5): shared MLX-free seam builds the additive mask; the
        // argMax stays here on the MLXArray. `GuidedMask.maskedArgmax` is the
        // CI-testable pure equivalent of `argMax(slice + add)`.
        let add = GuidedMask.additiveMask(allowed: maskBuf, vocab: vocab)
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
        guide: StructuredGuide? = nil,
        prefixCache: PrefixKVCache? = nil,
        cacheScope: String? = nil
    ) -> (ids: [Int], cachedTokens: Int) {
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
        // M59.1 — try to resume from a cached shared prefix. `prefixHit` is
        // non-nil only when caching is enabled AND a prior request in this
        // scope shares a prefix spanning ≥1 full 512-chunk. The returned
        // caches are cloned and already restored to boundary `B`; the entry
        // is refcounted until `release` (the `defer` below). On a miss with
        // caching enabled, `recorder` collects recurrent checkpoints during
        // prefill so the entry can be stored at the prefill→decode seam.
        let backbone: [KVCache]
        let prefillStart: Int
        var prefixHit: PrefixKVCache.Hit?
        var recorder: PrefixKVCache.Recorder?
        if let prefixCache, let cacheScope,
            let hit = prefixCache.acquire(
                scope: cacheScope, promptTokens: promptTokens, model: model)
        {
            backbone = hit.caches
            prefillStart = hit.startOffset
            prefixHit = hit
            MLXLLMModule.log.notice(
                """
                prefix-cache HIT prompt=\(promptTokens.count) \
                L=\(hit.commonPrefix) B=\(hit.startOffset) \
                suffix=\(promptTokens.count - hit.startOffset)
                """,
                metadata: ["function": "SpeculativeGeneration.generate"])
        } else {
            backbone = model.newCache(parameters: nil)
            prefillStart = 0
            if let prefixCache, cacheScope != nil {
                recorder = prefixCache.makeRecorder()
                MLXLLMModule.log.notice(
                    "prefix-cache MISS prompt=\(promptTokens.count)",
                    metadata: ["function": "SpeculativeGeneration.generate"])
            }
        }
        defer { if let prefixHit { prefixCache?.release(prefixHit) } }
        // M59.3 — tokens served from the reused prefix = the boundary we
        // resumed from (B). 0 on a cold prefill. Surfaced as OpenAI
        // `cached_tokens`.
        let cachedTokens = prefixHit?.startOffset ?? 0
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
            let chunkSize = PrefixKVCache.chunkSize
            // M59.1 — resume the chunk loop at `prefillStart` (the restored
            // 512-boundary B on a warm hit, else 0). Starting at a 512-multiple
            // keeps the absolute chunk grid identical to a cold prefill, which
            // is what makes prefix reuse bit-identical.
            let remaining = max(0, head.count - prefillStart)
            let totalChunks = (remaining + chunkSize - 1) / chunkSize
            var i = prefillStart
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
                // Snapshot recurrent state at this absolute boundary (only
                // 512-multiples are kept; the snapshot guards that). nil on a
                // warm hit (we don't re-store).
                recorder?.snapshot(
                    offset: min(i, head.count), backbone: backbone)
                DecodeProgress.counter?.recordPrefillChunk(
                    completed: done, total: totalChunks)
            }
        }
        (logits, hidden) = model.logitsAndHidden(
            tokenArray(promptTokens.last!), cache: backbone)
        asyncEval(backbone)
        if trace { eval(logits, hidden); tPrefill = dt(tPrefill0) }

        // M59.1 — store the post-prefill backbone NOW, before the decode loop
        // below trims/appends to it. The attention caches are at full prompt
        // length here; `store` clones them so this request's decode never
        // corrupts the cached entry.
        if let recorder, let prefixCache, let cacheScope {
            prefixCache.store(
                scope: cacheScope, promptTokens: promptTokens,
                backbone: backbone, recorder: recorder)
        }

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
        if commit(prev0) { return (result(), cachedTokens) }
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
            // M60.5 — stop early if the request was cancelled (client gone or
            // the M33 deadline) so we free the GPU instead of decoding all the
            // way to maxTokens for a request no one is waiting on.
            if DecodeLoopControl.isCancelled() { break }
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
            // L1 (M70.3): the greedy-parity accept decision — its sequencing
            // is pinned MLX-free in SpeculativeAcceptanceTests.
            let accepted = SpeculativeAcceptance.accepts(
                draft: draft, verifyPred: verifyPred)
            SpeculativeStats.observer?.recordIteration(accepted: accepted)
            specIters += 1
            if accepted { specAccepts += 1 }

            if accepted {
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

            // ADR 023 G1: skip the legacy periodic flush when the serve path
            // already bounds the MLX cache (it only churns the buffer pool);
            // keep it only under the unbounded operator opt-out.
            if !GovernorMemory.serveCacheBounded, out.count % 256 == 0 {
                MLX.Memory.clearCache()
            }
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
        return (result(), cachedTokens)
    }
}
