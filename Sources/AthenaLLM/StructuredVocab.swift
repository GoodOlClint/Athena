import AthenaStructured
import Foundation
import MLXLMCommon

/// Builds the outlines-core `VocabToken` set from the *model's own*
/// tokenizer so the byte mapping matches exactly (a mismatch makes
/// `Index::new` fail with `IncompatibleVocabulary`). Per-token bytes =
/// the UTF-8 of decoding that single id (special tokens kept), mirroring
/// the Python reference's `convert_tokens_to_string([token])`.
enum StructuredVocab {
    static func tokens(
        tokenizer: any Tokenizer, vocabSize: Int
    ) -> (tokens: [VocabToken], eos: UInt32) {
        let eos = tokenizer.eosTokenId ?? (vocabSize - 1)
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
