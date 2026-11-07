import Foundation

/// One word with DTW-aligned timing (M26.2). OpenAI `word` shape.
public struct WordTimestamp: Codable {
    public let word: String
    public let start: Double
    public let end: Double
    public let probability: Double
}

public struct VerboseSegment: Codable {
    public let id: Int
    public let start: Double
    public let end: Double
    public let text: String
    /// Mean per-token log-probability for the span (M26.1).
    public let avg_logprob: Double?
    /// `<|nospeech|>` probability at the window's first decode step.
    public let no_speech_prob: Double?
    /// UTF-8 byte length / zlib-compressed byte length of `text`.
    public let compression_ratio: Double?
    /// Speaker id (M4.3c) — present only when diarization was requested.
    public let speaker: Int?
    /// Words within this span (M26.2) — present only when requested.
    public let words: [WordTimestamp]?
}

/// OpenAI `response_format: "verbose_json"` shape. Optional fields are
/// omitted from the encoding when nil.
public struct VerboseTranscriptionResponse: Codable {
    public let task: String
    public let language: String
    public let duration: Double
    public let text: String
    /// Resident model id that produced the transcript.
    public let model: String?
    public let segments: [VerboseSegment]
    public let words: [WordTimestamp]?

    public static func make(
        _ result: TranscriptionResult, wantWords: Bool,
        speaker: (TranscriptionSegment) -> Int?
    ) -> VerboseTranscriptionResponse {
        func words(_ ws: [WordTiming]?) -> [WordTimestamp]? {
            ws.map {
                $0.map {
                    WordTimestamp(
                        word: $0.word, start: $0.start, end: $0.end,
                        probability: $0.probability)
                }
            }
        }
        return VerboseTranscriptionResponse(
            task: "transcribe", language: result.language,
            duration: result.duration, text: result.text,
            model: result.model,
            segments: result.segments.enumerated().map { i, s in
                VerboseSegment(
                    id: i, start: s.start, end: s.end, text: s.text,
                    avg_logprob: s.avgLogprob,
                    no_speech_prob: s.noSpeechProb,
                    compression_ratio: s.compressionRatio,
                    speaker: speaker(s), words: words(s.words))
            },
            words: wantWords ? words(result.words) : nil)
    }
}
