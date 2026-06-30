import Foundation

/// Channel-delimited reasoning markers (ADR 035). gemma-4-26b-a4b-it and family
/// emit `<|channel>thought\n…reasoning…\n<channel|>…content…` as LITERAL text
/// (not special tokens), with the thought channel forced on whenever tools are
/// present. These delimiters don't occur in normal content, so the filter is a
/// safe no-op for models that never emit them.
public let reasoningChannelStart = "<|channel>"
public let reasoningChannelEnd = "<channel|>"

/// Streaming splitter that routes generated text to `content` vs `reasoning` by
/// pulling out `<|channel>…<channel|>` blocks across chunk boundaries (ADR 035).
/// Mirrors `StopStreamFilter`'s partial-marker holdback so a marker split over
/// two chunks is never surfaced. Pure value type — no MLX — unit-testable.
///
/// State machine: `.content` → (sees start) → `.header` (drops the `thought\n`
/// channel-name line) → `.reasoning` → (sees end) → `.content`.
public struct ReasoningChannelFilter: Sendable {
    private enum State { case content, header, reasoning }
    private var state: State = .content
    private var buffer = ""
    // Holdback = longest marker minus one scalar: never surface a tail that
    // could complete into a start/end marker on the next chunk.
    private static let holdback =
        max(reasoningChannelStart.unicodeScalars.count,
            reasoningChannelEnd.unicodeScalars.count) - 1

    public init() {}

    /// Push more text; returns the content + reasoning safe to surface now.
    public mutating func push(
        _ piece: String
    ) -> (content: String, reasoning: String) {
        buffer += piece
        return drain(flush: false)
    }

    /// Drain everything once generation ends (emit any held tail).
    public mutating func flush() -> (content: String, reasoning: String) {
        drain(flush: true)
    }

    private mutating func drain(
        flush: Bool
    ) -> (content: String, reasoning: String) {
        var content = ""
        var reasoning = ""
        loop: while true {
            switch state {
            case .content:
                if let r = buffer.range(of: reasoningChannelStart) {
                    content += String(buffer[..<r.lowerBound])
                    buffer = String(buffer[r.upperBound...])
                    state = .header
                    continue loop
                }
                content += emitSafe(flush: flush)
                break loop
            case .header:
                // Drop the channel-name line (`thought\n`) after the start
                // marker. Once the newline arrives we're into reasoning text.
                if let nl = buffer.firstIndex(of: "\n") {
                    buffer = String(buffer[buffer.index(after: nl)...])
                    state = .reasoning
                    continue loop
                }
                // No newline yet: on flush, a header with no body/newline ⇒
                // nothing to surface (drop the dangling header).
                if flush { buffer = "" }
                break loop
            case .reasoning:
                if let r = buffer.range(of: reasoningChannelEnd) {
                    reasoning += String(buffer[..<r.lowerBound])
                    buffer = String(buffer[r.upperBound...])
                    state = .content
                    continue loop
                }
                reasoning += emitSafe(flush: flush)
                break loop
            }
        }
        return (content, reasoning)
    }

    /// The portion of `buffer` safe to surface now (holding back a possible
    /// partial-marker tail), advancing `buffer`. On flush, surface everything.
    private mutating func emitSafe(flush: Bool) -> String {
        if flush {
            let out = buffer
            buffer = ""
            return out
        }
        let scalars = buffer.unicodeScalars
        guard scalars.count > Self.holdback else { return "" }
        let cut = scalars.index(scalars.endIndex, offsetBy: -Self.holdback)
        let out = String(buffer.unicodeScalars[scalars.startIndex..<cut])
        buffer = String(buffer.unicodeScalars[cut...])
        return out
    }
}

/// One-shot split of a complete generation into `(content, reasoning)` for the
/// non-streaming path (ADR 035). Reuses the streaming filter so both paths
/// agree byte-for-byte.
public func splitReasoningChannel(
    _ text: String
) -> (content: String, reasoning: String) {
    var f = ReasoningChannelFilter()
    let a = f.push(text)
    let b = f.flush()
    return (a.content + b.content, a.reasoning + b.reasoning)
}
