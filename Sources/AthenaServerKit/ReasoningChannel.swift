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

/// **Not model-gated** (operator ruling, PR #213 round 3 — reverses an
/// earlier revision of this same PR that WAS model-gated): whether reasoning
/// starts already open is derived from the actual rendered PROMPT
/// (`ReasoningPromptTail.startsInOpenBlock`, computed in `AthenaLLM` where
/// the tokenizer lives, passed in as `startsInReasoning`), never from a
/// model-name guess. This keeps ADR 035's "universal, not model-gated"
/// property true for real: any model whose chat template pre-opens a
/// `<think>` block it never closes gets `.reasoningOpen`; every other model
/// gets `.awaitingOpenTag`, the same marker-driven no-op-in-practice design
/// Gemma's filter already relies on (the markers are chosen because they
/// don't occur in ordinary content — the same assumption ADR 035 makes for
/// `<|channel>`, not a new one).
///
/// The close-tag-only form is the observed default (`enable_thinking` unset
/// or `true`, 2026-09-23, `Qwen3.5-27B-4bit`, greedy): raw completion bytes
/// were `Thinking Process:\n\n1.  **Analyze the Request:**\n…\n5.  **Construct
/// Final Response:** 4.cw\n</think>\n\n4` — no opening tag anywhere in the
/// completion, because the template already put `<think>\n` at the end of
/// the prompt.
public struct QwenThinkFilter: Sendable {
    /// - `.awaitingOpenTag`: the default. Watches for a literal `<think>`
    ///   (paired form) before entering `.reasoning` — a no-op unless that
    ///   marker actually appears, same design as `ReasoningChannelFilter`.
    /// - `.reasoningOpen`: the prompt was determined (from its own rendered
    ///   text, see `ReasoningPromptTail`) to already be inside an open
    ///   `<think>` block — the close-tag-only form; reasoning starts at
    ///   byte 0.
    /// - `.passthrough`: schema-guided or forced-tool-call output (Codex
    ///   adversarial review, PR #213 round 4). The completion is structured
    ///   data, not prose — a legitimate schema value or tool argument can
    ///   contain the literal text `<think>…</think>` (e.g. a user asking the
    ///   model to echo it), and `.awaitingOpenTag` would silently strip that
    ///   substring out of valid JSON. Scans nothing; every byte is content.
    public enum Mode: Sendable, Equatable {
        case awaitingOpenTag
        case reasoningOpen
        case passthrough
    }

    private enum State { case content, reasoning, passthrough }
    private var state: State
    private var buffer = ""
    private static let holdback =
        max(
            qwenThinkStart.unicodeScalars.count,
            qwenThinkEnd.unicodeScalars.count) - 1

    public init(mode: Mode) {
        switch mode {
        case .awaitingOpenTag: state = .content
        case .reasoningOpen: state = .reasoning
        case .passthrough: state = .passthrough
        }
    }

    public mutating func push(
        _ piece: String
    ) -> (content: String, reasoning: String) {
        if case .passthrough = state { return (piece, "") }
        buffer += piece
        return drain(flush: false)
    }

    public mutating func flush() -> (content: String, reasoning: String) {
        if case .passthrough = state { return ("", "") }
        return drain(flush: true)
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
            case .passthrough:
                content += buffer
                buffer = ""
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
    _ text: String, mode: QwenThinkFilter.Mode
) -> (content: String, reasoning: String) {
    var f = QwenThinkFilter(mode: mode)
    let a = f.push(text)
    let b = f.flush()
    return (a.content + b.content, a.reasoning + b.reasoning)
}

/// Resolve the `QwenThinkFilter.Mode` for this request. `startsInReasoning`
/// comes from `ReasoningPromptTail.startsInOpenBlock` on the actual rendered
/// prompt (computed in `AthenaLLM`, where the tokenizer lives — not a
/// model-name guess). `isStructured` (schema-guided decoding OR a forced
/// tool call) overrides it to `.passthrough` regardless: the completion is
/// structured data, not prose, so it must never be scanned for `<think>` at
/// all — not `.reasoningOpen` (Codex adversarial review, PR #213 round 1:
/// that would swallow the whole structured response/tool call into
/// `reasoning_content`), and not `.awaitingOpenTag` either (Codex
/// adversarial review, PR #213 round 4: that still scans for literal
/// `<think>…</think>` pairs, silently corrupting a schema value or tool
/// argument that legitimately contains that text).
public func qwenReasoningMode(
    startsInReasoning: Bool, isStructured: Bool
) -> QwenThinkFilter.Mode {
    if isStructured { return .passthrough }
    return startsInReasoning ? .reasoningOpen : .awaitingOpenTag
}
