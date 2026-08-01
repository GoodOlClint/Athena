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
/// schema by masking logits to the structured-output Guide.
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
        let guide: StructuredGuide?
        let vocab: Int
        /// C2 — optional per-token logprob capture (ADR 013 §4). When `guide`
        /// is nil this processor does NO masking and exists only to capture
        /// (the non-Qwen unguided-greedy logprobs path).
        let sink: LogprobSink?

        func prompt(_ prompt: MLXArray) {}

        func process(logits: MLXArray) -> MLXArray {
            // C2: stash the UNMASKED last-position slice for the chosen token's
            // logprob (finalized in didSample once the token is known).
            sink?.stash(slice: logits)
            guard let guide else { return logits }
            var mask: [UInt8] = []
            _ = guide.allowedMask(into: &mask)
            // M70.3 (L5): the MLX-free seam (`GuidedMask`) for the
            // bit→additive-mask unpack. It was extracted to stop two copies
            // drifting; both siblings are now gone (publication S0, then #49),
            // so this is its one production caller — see the seam's own doc
            // for why it still lives out here rather than inlined.
            let add = GuidedMask.additiveMask(allowed: mask, vocab: vocab)
            // `logits` is the last-position slice, shape (1, vocab);
            // the (vocab,) additive mask broadcasts over it.
            return logits + MLXArray(add)
        }

        func didSample(token: MLXArray) {
            let t = token.item(Int.self)
            sink?.finalizeStashed(chosen: t)
            if let guide { _ = guide.advance(UInt32(t)) }
        }
    }

    /// Drive the substrate `TokenIterator` with a guided (and/or capturing)
    /// processor. `promptTokens` are post-chat-template ids (rebuilt into an
    /// `LMInput` here so the caller never captures a non-Sendable `LMInput`
    /// across the actor boundary). Returns the generated token ids.
    ///
    /// `guide` non-nil ⇒ schema-enforced from token 0 (the JSON span, no prefix
    /// to drop). `guide` nil + `sink` non-nil ⇒ plain greedy capture for the
    /// non-Qwen unguided logprobs path (C2). Stops at the Guide's EOS (forced
    /// once the schema is satisfied), `eosTokenId`, or `maxTokens`.
    static func generate(
        model: any LanguageModel,
        promptTokens: [Int],
        vocab: Int,
        maxTokens: Int,
        eosTokenId: Int?,
        guide: StructuredGuide?,
        sink: LogprobSink? = nil
    ) throws -> [Int] {
        let input = LMInput(tokens: MLXArray(promptTokens.map { Int32($0) }))
        let processor = GuidedLogitProcessor(
            guide: guide, vocab: vocab, sink: sink)
        var iterator = try TokenIterator(
            input: input, model: model, cache: nil,
            processor: processor, sampler: ArgMaxSampler(),
            prefillStepSize: 512, maxTokens: maxTokens)

        var out: [Int] = []
        while let token = iterator.next() {
            // M60.5 — abort early on request cancellation (disconnect/deadline).
            if DecodeLoopControl.isCancelled() { break }
            if let eosTokenId, token == eosTokenId { break }
            out.append(token)
            // C2: the token was emitted ⇒ promote its pending logprob capture.
            sink?.keep()
            // M46.8 — per-iteration progress for the heartbeat (see
            // `DecodeProgress`): this path is fully synchronous and
            // only surfaces one `.text` event at completion.
            DecodeProgress.counter?.incrementToken()
            // ADR 023 G1: skip the legacy flush when the serve path bounds the
            // MLX cache; keep it only under the unbounded operator opt-out.
            if !GovernorMemory.serveCacheBounded, out.count % 256 == 0 {
                MLX.Memory.clearCache()
            }
        }
        return out
    }
}
