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
/// filter's mode has to be told, per request, whether the caller already
/// opened the block: `qwenReasoningMode` resolves that from the same inputs
/// the template itself uses (the model + `enable_thinking` + whether the
/// response is schema-guided), BEFORE generation starts, so streaming never
/// has to buffer output while it waits to find out.
///
/// The close-tag-only form is the observed default (`enable_thinking` unset
/// or `true`, 2026-09-23, `Qwen3.5-27B-4bit`, greedy): raw completion bytes
/// were `Thinking Process:\n\n1.  **Analyze the Request:**\n…\n5.  **Construct
/// Final Response:** 4.cw\n</think>\n\n4` — no opening tag anywhere in the
/// completion.
///
/// `.disabled` is a TRUE no-op — it never scans for either marker. This
/// matters beyond Qwen3.5: without it, a `<think>` literal appearing in
/// ordinary content from ANY model (e.g. a prompt asking to explain XML/HTML
/// tags) would be silently stripped as a false-positive paired-form match,
/// breaking ADR 035's "safe no-op for models that don't emit the markers"
/// guarantee (Codex adversarial review, PR #213). Every non-Qwen3.5 request
/// gets `.disabled`.
public struct QwenThinkFilter: Sendable {
    /// - `.disabled`: never scans for either marker — a true no-op (non-Qwen
    ///   models, or Qwen3.5 requests this daemon didn't recognize as such).
    /// - `.awaitingOpenTag`: Qwen3.5, but the prompt did NOT pre-open the
    ///   block (`enable_thinking: false`, or a schema-guided/forced-tool-call
    ///   response — the Guide masks from token 0 and suppresses `<think>`,
    ///   so there is no reasoning to extract). Still watches defensively for
    ///   a literal `<think>` (paired form), scoped to Qwen3.5 only.
    /// - `.reasoningOpen`: Qwen3.5, prompt pre-opened the block — the
    ///   close-tag-only form; reasoning starts at byte 0.
    public enum Mode: Sendable, Equatable {
        case disabled
        case awaitingOpenTag
        case reasoningOpen
    }

    private enum State { case disabled, content, reasoning }
    private var state: State
    private var buffer = ""
    private static let holdback =
        max(
            qwenThinkStart.unicodeScalars.count,
            qwenThinkEnd.unicodeScalars.count) - 1

    public init(mode: Mode) {
        switch mode {
        case .disabled: state = .disabled
        case .awaitingOpenTag: state = .content
        case .reasoningOpen: state = .reasoning
        }
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
            case .disabled:
                content += buffer
                buffer = ""
                break loop
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
    _ text: String, mode: QwenThinkFilter.Mode
) -> (content: String, reasoning: String) {
    var f = QwenThinkFilter(mode: mode)
    let a = f.push(text)
    let b = f.flush()
    return (a.content + b.content, a.reasoning + b.reasoning)
}

/// Resolve, from request-time inputs alone (before any text has generated),
/// the `QwenThinkFilter.Mode` for this request. `modelName` is the resolved
/// canonical store id — a name-substring heuristic (no `model_type` config
/// read: this stays MLX-free and `AthenaServerKit` has no `AthenaLLM`/
/// `ModelSupport` dependency to classify architecture properly). Non-Qwen3.5
/// models always get `.disabled` — a true no-op, never scanning for either
/// marker (Codex adversarial review, PR #213: an earlier revision searched
/// for the markers unconditionally, which could strip a literal `<think>`
/// out of ordinary content from an unrelated model).
///
/// `isStructured` (schema-guided decoding OR a forced tool call) forces
/// `.awaitingOpenTag` even when `enable_thinking` would otherwise be on:
/// Athena's Guide masks the vocabulary from token 0 for both, which
/// suppresses the model's own `<think>` emission entirely (same review: an
/// earlier revision defaulted straight to `.reasoningOpen` here, which
/// swallowed the entire structured response — and any forced tool call —
/// into `reasoning_content`, leaving `content`/the parsed tool call empty).
/// `chatTemplateKwargs["enable_thinking"]` otherwise mirrors the template's
/// own default: thinking is ON unless explicitly set to `false`.
public func qwenReasoningMode(
    modelName: String, chatTemplateKwargs: [String: any Sendable]?,
    isStructured: Bool
) -> QwenThinkFilter.Mode {
    guard modelName.lowercased().contains("qwen3.5") else { return .disabled }
    if isStructured { return .awaitingOpenTag }
    if let flag = chatTemplateKwargs?["enable_thinking"] as? Bool, !flag {
        return .awaitingOpenTag
    }
    return .reasoningOpen
}
