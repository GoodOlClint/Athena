import AthenaCore
import AthenaStructured
import Foundation
import MLX
import MLXLMCommon

/// Schema-guided decoding on the **substrate** generation path, for any
/// architecture the substrate factory loads (Llama, Gemma, Mistral, Phi,
/// …). M23 fork A: structured output / tool calls used to silently drop
/// the schema for non-Qwen models (the `AthenaQwen35Model` vocab/guide
/// casts returned nil → unconstrained text). This makes them honor the
/// schema by masking logits to the outlines-core Guide.
///
/// Enforcement is from token 0 (matches the "structured ⇒ NO-THINK"
/// operator decision): the mask forces a schema-valid opener (`{`/`[`)
/// immediately, so output is guaranteed non-empty + schema-valid and the
/// Qwen-specific opener-realignment / deferred-thinking machinery is not
/// needed here. The vendored Qwen3.5 path is untouched (faster MTP +
/// optional thinking prefix), so this is purely additive for other arches.
enum GuidedSubstrate {

    /// A `LogitProcessor` that constrains every step to the Guide's
    /// currently-allowed token set and advances the Guide on each
    /// committed token. Greedy (paired with `ArgMaxSampler`): the forced
    /// token is always schema-valid, so `advance` never fails and the
    /// Guide state stays monotonic (no rollback).
    struct GuidedLogitProcessor: LogitProcessor {
        let guide: StructuredGuide
        let vocab: Int

        func prompt(_ prompt: MLXArray) {}

        func process(logits: MLXArray) -> MLXArray {
            var mask: [UInt8] = []
            _ = guide.allowedMask(into: &mask)
            var add = [Float](repeating: -.infinity, count: vocab)
            for i in 0..<vocab
            where (mask[i >> 3] >> UInt8(i & 7)) & 1 == 1 {
                add[i] = 0
            }
            // `logits` is the last-position slice, shape (1, vocab);
            // the (vocab,) additive mask broadcasts over it.
            return logits + MLXArray(add)
        }

        func didSample(token: MLXArray) {
            _ = guide.advance(UInt32(token.item(Int.self)))
        }
    }

    /// Drive the substrate `TokenIterator` with a guided processor.
    /// `promptTokens` are post-chat-template ids (rebuilt into an `LMInput`
    /// here so the caller never captures a non-Sendable `LMInput` across
    /// the actor boundary). Returns the generated token ids (the JSON span
    /// — enforcement is from token 0, so there is no prefix to drop).
    /// Stops at the Guide's EOS (forced once the schema is satisfied) or
    /// `maxTokens`.
    static func generate(
        model: any LanguageModel,
        promptTokens: [Int],
        vocab: Int,
        maxTokens: Int,
        eosTokenId: Int?,
        guide: StructuredGuide
    ) throws -> [Int] {
        let input = LMInput(tokens: MLXArray(promptTokens.map { Int32($0) }))
        let processor = GuidedLogitProcessor(guide: guide, vocab: vocab)
        var iterator = try TokenIterator(
            input: input, model: model, cache: nil,
            processor: processor, sampler: ArgMaxSampler(),
            prefillStepSize: 512, maxTokens: maxTokens)

        var out: [Int] = []
        while let token = iterator.next() {
            if let eosTokenId, token == eosTokenId { break }
            out.append(token)
            // M46.8 — per-iteration progress for the heartbeat. See the
            // matching note in GuidedGreedy.generate: the substrate
            // path here is also fully synchronous and only surfaces
            // one `.text` event at completion.
            DecodeProgress.counter?.incrementToken()
            if out.count % 256 == 0 { MLX.Memory.clearCache() }
        }
        return out
    }
}
