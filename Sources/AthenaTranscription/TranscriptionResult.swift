import Foundation

/// A timed transcription span. Times are seconds from the start of the
/// whole audio (window offsets already applied).
public struct TranscriptionSegment: Sendable, Codable {
    public let start: Double
    public let end: Double
    public let text: String
    public init(start: Double, end: Double, text: String) {
        self.start = start
        self.end = end
        self.text = text
    }
}

/// Full transcription result — text plus the detail needed for the
/// OpenAI `verbose_json`/`srt`/`vtt` response formats.
public struct TranscriptionResult: Sendable {
    public let text: String
    public let language: String
    public let duration: Double
    public let segments: [TranscriptionSegment]
    public init(
        text: String, language: String, duration: Double,
        segments: [TranscriptionSegment]
    ) {
        self.text = text
        self.language = language
        self.duration = duration
        self.segments = segments
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
            "\(i + 1)\n"
                + "\(stamp(s.start, comma: true)) --> "
                + "\(stamp(s.end, comma: true))\n"
                + "\(s.text.trimmingCharacters(in: .whitespaces))\n"
        }.joined(separator: "\n")
    }

    public static func vtt(_ segs: [TranscriptionSegment]) -> String {
        "WEBVTT\n\n"
            + segs.map { s in
                "\(stamp(s.start, comma: false)) --> "
                    + "\(stamp(s.end, comma: false))\n"
                    + "\(s.text.trimmingCharacters(in: .whitespaces))\n"
            }.joined(separator: "\n")
    }
}
