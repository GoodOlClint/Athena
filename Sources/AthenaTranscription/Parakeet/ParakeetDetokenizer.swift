import Foundation

/// SentencePiece detokenization for Parakeet's TDT vocabulary. MLX-free, so it
/// is unit-testable under `swift test` (ADR 008/009) — the decode loop emits a
/// list of token ids, and turning those into text is pure string work.
///
/// The Parakeet-TDT-0.6B-v3 vocabulary (`joint.vocabulary`, 8192 pieces) is a
/// SentencePiece unigram model whose pieces use the `▁` (U+2581) word-boundary
/// marker, exactly like the Python reference's `decode()` (join pieces, `▁`→
/// space). Two production refinements over the spike's naive join:
///
///   1. **Special-token stripping.** v3's vocabulary opens with 264 control
///      pieces (`<unk>`, `<pad>`, `<|startoftranscript|>`, 183 `<|xx|>`
///      language tags, emotion/timestamp/diarization tags, `<|spltoken*|>` …).
///      Greedy TDT decode can emit these; rendered literally they corrupt the
///      transcript. Any piece of the form `<…>` is dropped.
///   2. **Leading-space trim.** The first real piece carries a leading `▁`,
///      which would otherwise produce a transcript with a leading space.
///
/// Out-of-range ids are skipped defensively (a decode bug can't crash the
/// detokenizer); the blank id is never appended to the hypothesis upstream.
public enum ParakeetDetokenizer {
    /// True when `piece` is a SentencePiece control token (`<unk>`, `<pad>`,
    /// `<|…|>`, etc.) that must not appear in the rendered transcript.
    public static func isSpecial(_ piece: String) -> Bool {
        piece.count >= 2 && piece.hasPrefix("<") && piece.hasSuffix(">")
    }

    /// Detokenize `ids` against `vocabulary`: drop specials + out-of-range ids,
    /// join the rest, map `▁`→space, and trim a single leading space.
    public static func detokenize(_ ids: [Int], vocabulary: [String]) -> String {
        var s = ""
        for id in ids where id >= 0 && id < vocabulary.count {
            let piece = vocabulary[id]
            if isSpecial(piece) { continue }
            s += piece.replacingOccurrences(of: "\u{2581}", with: " ")
        }
        // SentencePiece renders the leading word-boundary marker as a space;
        // drop exactly that one leading space (not all whitespace — interior
        // spacing is meaningful).
        if s.hasPrefix(" ") { s.removeFirst() }
        return s
    }
}
