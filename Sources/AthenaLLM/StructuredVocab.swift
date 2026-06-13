import AthenaStructured
import Foundation
import MLXLMCommon

/// Builds the structured-output `VocabToken` set from the *model's own*
/// tokenizer so the byte mapping matches exactly (a mismatch would make
/// the guide mask the wrong tokens). Per-token bytes = the UTF-8 of
/// decoding that single id (special tokens kept), mirroring the Python
/// reference's `convert_tokens_to_string([token])`. The shim builds the
/// llguidance token trie from these (M53).
enum StructuredVocab {
    static func tokens(
        tokenizer: any Tokenizer, vocabSize: Int
    ) -> (tokens: [VocabToken], eos: UInt32) {
        // C12: when the tokenizer has no eos, DON'T fabricate `vocabSize-1`
        // — that is a real token, and skipping it below would drop it from
        // the guide's allowed set (it could then never be generated under a
        // schema). Use a sentinel one past the real range: the loop never
        // hits it (so no real token is skipped), and the shim's build_words
        // sizes the trie to `max_id+1`, adding a single never-emitted
        // control slot. The decode loop's stop token is the tokenizer's own
        // eos (separate), so a phantom eos here can't affect stopping.
        let eos = tokenizer.eosTokenId ?? vocabSize
        var out: [VocabToken] = []
        out.reserveCapacity(vocabSize)
        for id in 0..<vocabSize {
            if id == eos { continue }
            let s = tokenizer.decode(tokenIds: [id], skipSpecialTokens: false)
            out.append(VocabToken(id: UInt32(id), bytes: Array(s.utf8)))
        }
        return (out, UInt32(eos))
    }
}
