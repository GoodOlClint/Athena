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

        // Prefill all but the last prompt token.
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
            (logits, _) = model.logitsAndHidden(
                SpeculativeGeneration.tokenArray(t), cache: backbone)
            asyncEval(backbone)
            if out.count % 256 == 0 { MLX.Memory.clearCache() }
        }
        return result()
    }
}
