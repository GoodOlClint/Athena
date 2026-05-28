import AthenaCore
import AthenaModels
import AthenaStructured
import Foundation
import MLX
import MLXLMCommon

/// Plain Guide-masked greedy decoding — no MTP, no speculation. Used for
/// structured output when the speculative path is unavailable (model has
/// no MTP head, or `--speculative` is off). With a Guide it emits exactly
/// the schema-constrained greedy sequence; without one it is ordinary
/// greedy (so it also serves as a non-MTP fallback). Reuses
/// `SpeculativeGeneration`'s mask/argmax helpers.
enum GuidedGreedy {
    static func generate(
        model: AthenaQwen35Model,
        promptTokens: [Int],
        maxTokens: Int,
        eosTokenId: Int?,
        guide: StructuredGuide?
    ) -> [Int] {
        let vocab = model.vocabularySize
        let backbone = model.newCache(parameters: nil)

        // Prefill all but the last prompt token. M48.4 publishes
        // per-chunk progress to the heartbeat so an operator can tell
        // "stuck in prefill at chunk N" apart from "decoding slowly."
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
        var (logits, _) = model.logitsAndHidden(
            SpeculativeGeneration.tokenArray(promptTokens.last!),
            cache: backbone)
        asyncEval(backbone)

        var out: [Int] = []
        var decoder = GuidedDecoder(
            guide: guide, vocab: vocab,
            idleBudget: max(8, maxTokens / 2))
        var jsonStart: Int?
        func commit(_ t: Int) -> Bool {
            if t == eosTokenId {
                if guide != nil, !decoder.enforcing {
                    decoder.forceEnforce()
                    return false
                }
                return true
            }
            out.append(t)
            if case .jsonStart = decoder.commit(t) {
                jsonStart = out.count - 1
            }
            return out.count >= maxTokens
        }
        func result() -> [Int] {
            guide == nil ? out : Array(out[(jsonStart ?? out.count)...])
        }

        while out.count < maxTokens {
            let t = decoder.pick(logits[0..., -1, 0...])
            if commit(t) { break }
            // M46.8 — per-iteration progress for the heartbeat. The
            // Guide path is fully synchronous and only emits one `.text`
            // event at completion, so without this increment the
            // heartbeat sees `tokens=0` and reports `tokens_per_sec=0.0`
            // for the entire decode. `DecodeProgress.counter` is set by
            // the serve path's collectMetered via TaskLocal and is nil
            // outside that context (so the call is a cheap no-op for
            // direct unit-test invocation).
            DecodeProgress.counter?.incrementToken()
            (logits, _) = model.logitsAndHidden(
                SpeculativeGeneration.tokenArray(t), cache: backbone)
            asyncEval(backbone)
            if out.count % 256 == 0 { MLX.Memory.clearCache() }
        }
        return result()
    }
}
