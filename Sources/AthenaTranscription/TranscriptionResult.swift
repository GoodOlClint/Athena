import Foundation

/// A single word with cross-attention DTW-aligned timing (M26.2). Times
/// are seconds from the start of the whole audio.
public struct WordTiming: Sendable, Codable {
    public let word: String
    public let start: Double
    public let end: Double
    /// Mean token probability over the word's tokens (0...1).
    public let probability: Double
    public init(
        word: String, start: Double, end: Double, probability: Double
    ) {
        self.word = word
        self.start = start
        self.end = end
        self.probability = probability
    }
}

/// A timed transcription span. Times are seconds from the start of the
/// whole audio (window offsets already applied).
public struct TranscriptionSegment: Sendable, Codable {
    public let start: Double
    public let end: Double
    public let text: String
    /// Mean per-token log-probability of the tokens decoded into this
    /// span (M26.1). nil when not tracked (e.g. the stub). Surfaced only
    /// in the `verbose_json` response; other formats ignore it.
    public let avgLogprob: Double?
    /// Words whose timing falls within this span (M26.2). nil when word
    /// timestamps were not requested; surfaced only in `verbose_json`.
    public let words: [WordTiming]?
    public init(
        start: Double, end: Double, text: String,
        avgLogprob: Double? = nil, words: [WordTiming]? = nil
    ) {
        self.start = start
        self.end = end
        self.text = text
        self.avgLogprob = avgLogprob
        self.words = words
    }
}

/// Full transcription result — text plus the detail needed for the
/// OpenAI `verbose_json`/`srt`/`vtt` response formats.
public struct TranscriptionResult: Sendable {
    public let text: String
    public let language: String
    public let duration: Double
    public let segments: [TranscriptionSegment]
    /// All aligned words across the audio (M26.2), empty unless word
    /// timestamps were requested. Surfaced only in `verbose_json`.
    public let words: [WordTiming]
    public init(
        text: String, language: String, duration: Double,
        segments: [TranscriptionSegment], words: [WordTiming] = []
    ) {
        self.text = text
        self.language = language
        self.duration = duration
        self.segments = segments
        self.words = words
    }
}

/// Pure formatters for the subtitle response formats. Model-free.
public enum TranscriptionFormat {
    private static func stamp(_ t: Double, comma: Bool) -> String {
        let ms = Int((t * 1000).rounded())
        let h = ms / 3_600_000
        let m = (ms % 3_600_000) / 60_000
        let s = (ms % 60_000) / 1000
        let f = ms % 1000
        return String(
            format: "%02d:%02d:%02d%@%03d", h, m, s,
            comma ? "," : ".", f)
    }

    public static func srt(_ segs: [TranscriptionSegment]) -> String {
        segs.enumerated().map { i, s in
            // D12: never emit a cue whose end precedes its start — an
            // inverted span (start > end from a malformed timestamp stream)
            // produces a backwards cue most players reject. Clamp end ≥ start.
            "\(i + 1)\n"
                + "\(stamp(s.start, comma: true)) --> "
                + "\(stamp(max(s.start, s.end), comma: true))\n"
                + "\(s.text.trimmingCharacters(in: .whitespaces))\n"
        }.joined(separator: "\n")
    }

    public static func vtt(_ segs: [TranscriptionSegment]) -> String {
        "WEBVTT\n\n"
            + segs.map { s in
                // D12: clamp end ≥ start (see srt).
                "\(stamp(s.start, comma: false)) --> "
                    + "\(stamp(max(s.start, s.end), comma: false))\n"
                    + "\(s.text.trimmingCharacters(in: .whitespaces))\n"
            }.joined(separator: "\n")
    }
}
