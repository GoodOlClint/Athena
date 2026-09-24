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
        max(
            reasoningChannelStart.unicodeScalars.count,
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
        let out = String(buffer.unicodeScalars[scalars.startIndex ..< cut])
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

// MARK: - Qwen3.5 `<think>…</think>` (#198, ADR 035 amendment)

/// Qwen3.5's reasoning markers (#198). Unlike Gemma's `<|channel>`/`<channel|>`
/// (ADR 035), these are hard-coded rather than read from `tokenizer_config`:
/// Qwen3.5's `tokenizer_config.json` carries no dedicated soc/eoc-style field
/// for them (checked directly against the mlx-community/Qwen3.5-27B-4bit
/// snapshot) — the tags are literals baked into the Jinja chat template, so
/// there is nothing to read.
public let qwenThinkStart = "<think>"
public let qwenThinkEnd = "</think>"

/// Model-gated, unlike `ReasoningChannelFilter` (ADR 035 amendment — see the
/// ADR file). Qwen3.5's chat template pre-inserts the OPENING `<think>\n`
/// into the *prompt* whenever thinking is active (`enable_thinking` unset or
/// `true`), so the model's own completion carries only the CLOSE marker —
/// reasoning text starts at generation's first byte with no delimiter to key
/// on. A marker-only filter (Gemma's design) can't tell that apart from
/// ordinary content for a model that never emits `</think>` at all, so this
/// filter's initial state has to be told, per request, whether the caller
/// already opened the block: `qwenExpectsThinking` resolves that from the
/// same inputs the template itself uses (the model + `enable_thinking`),
/// BEFORE generation starts, so streaming never has to buffer output while
/// it waits to find out.
///
/// The close-tag-only form is the observed default (`enable_thinking` unset
/// or `true`, 2026-09-23, `Qwen3.5-27B-4bit`, greedy): raw completion bytes
/// were `Thinking Process:\n\n1.  **Analyze the Request:**\n…\n5.  **Construct
/// Final Response:** 4.cw\n</think>\n\n4` — no opening tag anywhere in the
/// completion. A defensive literal `<think>` match is still honored in
/// `.content` state (paired form) in case a differently-configured template
/// ever emits the open tag itself.
public struct QwenThinkFilter: Sendable {
    private enum State { case content, reasoning }
    private var state: State
    private var buffer = ""
    private static let holdback =
        max(
            qwenThinkStart.unicodeScalars.count,
            qwenThinkEnd.unicodeScalars.count) - 1

    /// `expectThinking`: the block is already open (no start marker will
    /// arrive) — start directly in `.reasoning`. Otherwise start in
    /// `.content`, still watching for a literal `<think>` (paired form).
    public init(expectThinking: Bool) {
        state = expectThinking ? .reasoning : .content
    }

    public mutating func push(
        _ piece: String
    ) -> (content: String, reasoning: String) {
        buffer += piece
        return drain(flush: false)
    }

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
                if let r = buffer.range(of: qwenThinkStart) {
                    content += String(buffer[..<r.lowerBound])
                    buffer = String(buffer[r.upperBound...])
                    state = .reasoning
                    continue loop
                }
                content += emitSafe(flush: flush)
                break loop
            case .reasoning:
                if let r = buffer.range(of: qwenThinkEnd) {
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

    private mutating func emitSafe(flush: Bool) -> String {
        if flush {
            let out = buffer
            buffer = ""
            return out
        }
        let scalars = buffer.unicodeScalars
        guard scalars.count > Self.holdback else { return "" }
        let cut = scalars.index(scalars.endIndex, offsetBy: -Self.holdback)
        let out = String(buffer.unicodeScalars[scalars.startIndex ..< cut])
        buffer = String(buffer.unicodeScalars[cut...])
        return out
    }
}

/// One-shot split for the non-streaming path (mirrors `splitReasoningChannel`).
public func splitQwenThink(
    _ text: String, expectThinking: Bool
) -> (content: String, reasoning: String) {
    var f = QwenThinkFilter(expectThinking: expectThinking)
    let a = f.push(text)
    let b = f.flush()
    return (a.content + b.content, a.reasoning + b.reasoning)
}

/// Resolve, from request-time inputs alone (before any text has generated),
/// whether Qwen3.5's chat template will have opened the `<think>` block in
/// the prompt. `modelName` is the resolved canonical store id — a
/// name-substring heuristic (no `model_type` config read: this stays
/// MLX-free and AthenaServerKit has no AthenaLLM/ModelSupport dependency to
/// classify architecture properly). `chatTemplateKwargs["enable_thinking"]`
/// mirrors the template's own default: thinking is ON unless explicitly set
/// to `false`.
public func qwenExpectsThinking(
    modelName: String, chatTemplateKwargs: [String: any Sendable]?
) -> Bool {
    guard modelName.lowercased().contains("qwen3.5") else { return false }
    if let flag = chatTemplateKwargs?["enable_thinking"] as? Bool {
        return flag
    }
    return true
}
