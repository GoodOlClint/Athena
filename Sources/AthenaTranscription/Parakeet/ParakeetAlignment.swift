import Foundation

/// Turns the flat list of TDT-aligned tokens the greedy decode emits into the
/// timed segments + words the `/v1/audio/transcriptions` `verbose_json`/SRT/VTT
/// responses need (ADR 020 S3). MLX-free — the decode loop produces token
/// `(start, duration)` from the TDT durations (`time_ratio` = 0.08 s/encoder
/// frame), and grouping those into words/sentences is pure string + arithmetic
/// work — so it is unit-pinned under `swift test` (ADR 008/009).
///
/// Ported from the Python reference (`alignment.py` `tokens_to_sentences` with
/// its default punctuation-only `SentenceConfig`, and the `" " in token.text`
/// word-boundary rule). SentencePiece pieces carry the `▁` word-boundary marker
/// rendered as a leading space; a token whose text begins with a space starts a
/// new word.
public enum ParakeetAlignment {
    /// One decoded token with its TDT-derived timing (seconds from clip start).
    /// `text` is the single-piece detokenization (`▁`→space, special tokens
    /// already excluded by the caller); `confidence` is the predicted token's
    /// softmax probability (0, 1].
    public struct Token: Sendable, Equatable {
        public let id: Int
        public let text: String
        public let start: Double
        public let duration: Double
        public let confidence: Double
        public var end: Double { start + duration }
        public init(
            id: Int, text: String, start: Double, duration: Double,
            confidence: Double = 1.0
        ) {
            self.id = id
            self.text = text
            self.start = start
            self.duration = duration
            self.confidence = confidence
        }
    }

    /// Group tokens into words at `▁` boundaries (a token whose text starts with
    /// a space opens a new word). Word timing spans its tokens; `probability` is
    /// the mean token confidence.
    public static func words(from tokens: [Token]) -> [WordTiming] {
        var out: [WordTiming] = []
        var cur: [Token] = []
        func flush() {
            guard !cur.isEmpty else { return }
            let text = cur.map(\.text).joined()
                .trimmingCharacters(in: .whitespaces)
            guard !text.isEmpty else { cur = []; return }
            let prob = cur.map(\.confidence).reduce(0, +) / Double(cur.count)
            out.append(
                WordTiming(
                    word: text, start: cur.first!.start, end: cur.last!.end,
                    probability: prob))
            cur = []
        }
        for t in tokens {
            // A leading space marks a SentencePiece word boundary. Start a new
            // word only once the current one has content (the first token also
            // carries a leading ▁ but must not flush an empty word).
            if t.text.hasPrefix(" ") && !cur.isEmpty { flush() }
            cur.append(t)
        }
        flush()
        return out
    }

    /// True when `text` ends a sentence: terminal punctuation, or a period
    /// followed by a word boundary (so abbreviations mid-word don't split).
    /// `nextStartsWord` is whether the following token opens a new word; nil at
    /// the end of the stream (treated as a boundary for the `.` rule).
    static func isSentenceEnd(_ text: String, nextStartsWord: Bool?) -> Bool {
        if text.contains("!") || text.contains("?") || text.contains("。")
            || text.contains("？") || text.contains("！")
        {
            return true
        }
        if text.contains(".") {
            return nextStartsWord ?? true
        }
        return false
    }

    /// Group tokens into sentence segments (reference `tokens_to_sentences`,
    /// punctuation-only). When `attachWords`, each segment carries the words
    /// that fall within it. `avgLogprob` is the mean per-token log-probability
    /// (ln of the softmax confidence) over the segment's tokens — a genuine
    /// average log-prob, matching the OpenAI field's semantics.
    public static func segments(
        from tokens: [Token], attachWords: Bool
    ) -> [TranscriptionSegment] {
        guard !tokens.isEmpty else { return [] }
        let allWords = attachWords ? words(from: tokens) : []
        var out: [TranscriptionSegment] = []
        var cur: [Token] = []

        func flush() {
            guard !cur.isEmpty else { return }
            let text = cur.map(\.text).joined()
                .trimmingCharacters(in: .whitespaces)
            let start = cur.first!.start
            let end = cur.last!.end
            let meanLogProb =
                cur.map { Foundation.log(max($0.confidence, 1e-10)) }
                .reduce(0, +) / Double(cur.count)
            let segWords =
                attachWords
                ? allWords.filter { $0.start >= start && $0.start <= end }
                : nil
            out.append(
                TranscriptionSegment(
                    start: start, end: end, text: text,
                    avgLogprob: meanLogProb, words: segWords))
            cur = []
        }

        for (idx, t) in tokens.enumerated() {
            cur.append(t)
            let nextStartsWord: Bool? =
                idx + 1 < tokens.count
                ? tokens[idx + 1].text.hasPrefix(" ") : nil
            if isSentenceEnd(t.text, nextStartsWord: nextStartsWord) { flush() }
        }
        flush()
        return out
    }
}
