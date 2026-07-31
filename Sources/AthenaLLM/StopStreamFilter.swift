/// Streaming stop-sequence truncation (M31.3 — OpenAI `stop`). Feed
/// generated text in (whole or in chunks) and it returns only the text
/// safe to surface, holding back a short tail that could still become the
/// start of a stop sequence so a match spanning a chunk boundary is not
/// missed. Once any stop sequence appears, `stopped` latches and all
/// further input is suppressed (the caller keeps draining the generator
/// for usage). Pure value type — no MLX — so it is unit-testable.
public struct StopStreamFilter: Sendable {
    public let stops: [String]
    private let maxLen: Int
    private var buffer = ""
    public private(set) var stopped = false
    /// WP7 — the stop sequence that actually latched `stopped` (the
    /// earliest-position match), so a caller reporting *which* sequence hit
    /// (Anthropic `stop_sequence`) names the real one instead of guessing
    /// `stops.first`. nil until a stop matches.
    public private(set) var matchedStop: String?

    public init(stops: [String]) {
        self.stops = stops.filter { !$0.isEmpty }
        // C13: measure the hold-back in UNICODE SCALARS, not graphemes.
        // `.count` (Character/grapheme count) under-counts a stop that
        // contains a multi-scalar grapheme (e.g. a ZWJ emoji, a combining
        // sequence), so the grapheme-sized hold-back could surface part of
        // it and let the stop split across a chunk boundary slip through.
        self.maxLen = self.stops.map { $0.unicodeScalars.count }.max() ?? 0
    }

    /// Whether any stop sequences are active.
    public var isActive: Bool { !stops.isEmpty }

    /// Push more generated text; returns the portion safe to surface now.
    public mutating func push(_ piece: String) -> String {
        if stopped || !isActive { return stopped ? "" : piece }
        buffer += piece
        if let r = earliestStop(in: buffer) {
            let out = String(buffer[buffer.startIndex ..< r.range.lowerBound])
            stopped = true
            matchedStop = r.stop
            buffer = ""
            return out
        }
        // Hold back the last (maxLen-1) UNICODE SCALARS: they could be the
        // prefix of a stop sequence completed by the next chunk. Counting
        // in scalars (not graphemes) guarantees a multi-scalar stop is never
        // partially surfaced before it can be matched (C13).
        let keep = maxLen - 1
        let scalars = buffer.unicodeScalars
        guard scalars.count > keep else { return "" }
        let cut = scalars.index(scalars.endIndex, offsetBy: -keep)
        let out = String(buffer.unicodeScalars[scalars.startIndex ..< cut])
        buffer = String(buffer.unicodeScalars[cut...])
        return out
    }

    /// Drain whatever remains once the generator ends (no stop matched).
    public mutating func flush() -> String {
        if stopped { return "" }
        let out = buffer
        buffer = ""
        return out
    }

    /// Earliest range of any stop sequence within `text`, and which stop it was.
    private func earliestStop(in text: String)
        -> (range: Range<String.Index>, stop: String)?
    {
        var best: (range: Range<String.Index>, stop: String)?
        for s in stops {
            if let r = text.range(of: s) {
                if best == nil || r.lowerBound < best!.range.lowerBound {
                    best = (r, s)
                }
            }
        }
        return best
    }

    /// One-shot truncation for a complete string (the sync path): the text
    /// up to the first stop sequence, whether a stop was found, and WHICH
    /// stop matched (earliest position — same attribution the streaming
    /// `matchedStop` reports, so both paths name the real sequence).
    public static func truncate(_ text: String, stops: [String])
        -> (text: String, stopped: Bool, matched: String?)
    {
        let active = stops.filter { !$0.isEmpty }
        var best: (range: Range<String.Index>, stop: String)?
        for s in active {
            if let r = text.range(of: s),
                best == nil || r.lowerBound < best!.range.lowerBound
            {
                best = (r, s)
            }
        }
        if let best {
            return (
                String(text[text.startIndex ..< best.range.lowerBound]),
                true, best.stop
            )
        }
        return (text, false, nil)
    }
}
