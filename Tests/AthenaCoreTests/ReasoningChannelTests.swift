import Foundation
import XCTest

@testable import AthenaServerKit

/// ADR 035 — pins the channel-delimited reasoning splitter: `<|channel>thought…
/// <channel|>` blocks route to `reasoning`, the rest to `content`, across chunk
/// boundaries. MLX-free, fast tier (ADR 009).
final class ReasoningChannelTests: XCTestCase {
    func testNoMarkersIsPassthrough() {
        let r = splitReasoningChannel("Hello, how are you?")
        XCTAssertEqual(r.content, "Hello, how are you?")
        XCTAssertEqual(r.reasoning, "")
    }

    /// The exact leaked shape: empty thought channel then content.
    func testEmptyThoughtChannelStripped() {
        let r = splitReasoningChannel(
            "<|channel>thought\n<channel|>Based on the results, it is sunny.")
        XCTAssertEqual(r.content, "Based on the results, it is sunny.")
        XCTAssertEqual(r.reasoning, "")
    }

    func testReasoningExtracted() {
        let r = splitReasoningChannel(
            "<|channel>thought\nlet me think\n<channel|>The answer is 42.")
        XCTAssertEqual(r.content, "The answer is 42.")
        XCTAssertEqual(r.reasoning, "let me think\n")
    }

    /// Streaming, byte-by-byte, must equal the one-shot split — including when a
    /// marker is split across chunk boundaries.
    func testStreamingEqualsOneShot() {
        let full =
            "pre<|channel>thought\nreasoning here\n<channel|>post-answer text"
        let oneShot = splitReasoningChannel(full)
        var f = ReasoningChannelFilter()
        var content = "", reasoning = ""
        for ch in full {  // one character at a time — worst case for holdback
            let s = f.push(String(ch))
            content += s.content
            reasoning += s.reasoning
        }
        let tail = f.flush()
        content += tail.content
        reasoning += tail.reasoning
        XCTAssertEqual(content, oneShot.content)
        XCTAssertEqual(reasoning, oneShot.reasoning)
        XCTAssertEqual(content, "prepost-answer text")
        XCTAssertEqual(reasoning, "reasoning here\n")
    }

    /// A literal "<" in ordinary content must not be eaten (it's held back then
    /// flushed, never dropped).
    func testAngleBracketInContentSurvives() {
        var f = ReasoningChannelFilter()
        let a = f.push("a < b and c")
        let b = f.flush()
        XCTAssertEqual(a.content + b.content, "a < b and c")
        XCTAssertEqual(a.reasoning + b.reasoning, "")
    }
}

/// #198 — Qwen3.5's `<think>…</think>`. Unlike ADR 035's Gemma filter, this
/// one is model-gated (`qwenReasoningMode`): the close-tag-only form has no
/// start marker to key on, so the filter must be told up front whether the
/// prompt already opened the block — and non-Qwen3.5 requests must get a
/// TRUE no-op (`.disabled`), never scanning for either marker (Codex
/// adversarial review, PR #213: an earlier revision searched for `<think>`
/// unconditionally, which could strip a literal occurrence of the tag out of
/// ordinary content from an unrelated model). MLX-free, fast tier (ADR 009).
final class QwenThinkFilterTests: XCTestCase {
    /// The exact observed shape (2026-09-23, `Qwen3.5-27B-4bit`, greedy,
    /// default `enable_thinking`): the completion carries ONLY the close
    /// marker — the prompt already opened the block.
    func testCloseTagOnlyForm() {
        let raw =
            "Thinking Process:\n\n1.  Analyze.\n\n5.  Construct: 4.\n</think>\n\n4"
        let r = splitQwenThink(raw, mode: .reasoningOpen)
        XCTAssertEqual(r.content, "\n\n4")
        XCTAssertEqual(r.reasoning, "Thinking Process:\n\n1.  Analyze.\n\n5.  Construct: 4.\n")
    }

    /// Paired form is honored in `.awaitingOpenTag` (Qwen3.5, but the prompt
    /// did NOT pre-open the block — e.g. a differently-configured template
    /// that emits the open tag itself).
    func testPairedTagFormAwaitingOpenTag() {
        let raw = "<think>reasoning</think>content"
        let r = splitQwenThink(raw, mode: .awaitingOpenTag)
        XCTAssertEqual(r.content, "content")
        XCTAssertEqual(r.reasoning, "reasoning")
    }

    /// `.disabled` is a TRUE no-op: it must not touch a literal `<think>` in
    /// ordinary content — the exact failure Codex's review found in an
    /// earlier revision (a non-Qwen model, or a Qwen3.5 request this daemon
    /// didn't recognize as such, discussing the tag itself).
    func testDisabledDoesNotStripLiteralTags() {
        let raw = "Explain the <think>...</think> HTML-like tag convention."
        let r = splitQwenThink(raw, mode: .disabled)
        XCTAssertEqual(r.content, raw)
        XCTAssertEqual(r.reasoning, "")
    }

    /// `enable_thinking:false` (or any non-Qwen model): no markers at all —
    /// the filter must be a pure no-op, exactly the observed disabled-form
    /// bytes (2026-09-23, same checkpoint, `enable_thinking:false`): raw
    /// completion was the literal string "4", nothing else.
    func testNoThinkingIsPassthrough() {
        let r = splitQwenThink("4", mode: .awaitingOpenTag)
        XCTAssertEqual(r.content, "4")
        XCTAssertEqual(r.reasoning, "")
    }

    /// Streaming, byte-by-byte, must equal the one-shot split for all modes.
    func testStreamingEqualsOneShotCloseTagOnly() {
        let full = "reasoning text here\n</think>\n\nfinal answer"
        let oneShot = splitQwenThink(full, mode: .reasoningOpen)
        var f = QwenThinkFilter(mode: .reasoningOpen)
        var content = "", reasoning = ""
        for ch in full {
            let s = f.push(String(ch))
            content += s.content
            reasoning += s.reasoning
        }
        let tail = f.flush()
        content += tail.content
        reasoning += tail.reasoning
        XCTAssertEqual(content, oneShot.content)
        XCTAssertEqual(reasoning, oneShot.reasoning)
    }

    func testStreamingEqualsOneShotPaired() {
        let full = "pre<think>\nreasoning here\n</think>post"
        let oneShot = splitQwenThink(full, mode: .awaitingOpenTag)
        var f = QwenThinkFilter(mode: .awaitingOpenTag)
        var content = "", reasoning = ""
        for ch in full {
            let s = f.push(String(ch))
            content += s.content
            reasoning += s.reasoning
        }
        let tail = f.flush()
        content += tail.content
        reasoning += tail.reasoning
        XCTAssertEqual(content, oneShot.content)
        XCTAssertEqual(reasoning, oneShot.reasoning)
        XCTAssertEqual(content, "prepost")
        XCTAssertEqual(reasoning, "\nreasoning here\n")
    }

    func testStreamingEqualsOneShotDisabled() {
        let full = "pre<think>literal</think>post"
        let oneShot = splitQwenThink(full, mode: .disabled)
        var f = QwenThinkFilter(mode: .disabled)
        var content = "", reasoning = ""
        for ch in full {
            let s = f.push(String(ch))
            content += s.content
            reasoning += s.reasoning
        }
        let tail = f.flush()
        content += tail.content
        reasoning += tail.reasoning
        XCTAssertEqual(content, oneShot.content)
        XCTAssertEqual(reasoning, oneShot.reasoning)
        XCTAssertEqual(content, full)
        XCTAssertEqual(reasoning, "")
    }

    // MARK: qwenReasoningMode

    func testModeDefaultReasoningOpenForQwen35() {
        XCTAssertEqual(
            qwenReasoningMode(
                modelName: "Qwen3.5-27B-4bit", chatTemplateKwargs: nil,
                isStructured: false),
            .reasoningOpen)
    }

    func testModeExplicitEnableThinkingFalse() {
        XCTAssertEqual(
            qwenReasoningMode(
                modelName: "Qwen3.5-27B-4bit",
                chatTemplateKwargs: ["enable_thinking": false],
                isStructured: false),
            .awaitingOpenTag)
    }

    func testModeExplicitEnableThinkingTrue() {
        XCTAssertEqual(
            qwenReasoningMode(
                modelName: "Qwen3.5-27B-4bit",
                chatTemplateKwargs: ["enable_thinking": true],
                isStructured: false),
            .reasoningOpen)
    }

    /// Non-Qwen models always get `.disabled` — a true no-op — even with
    /// `enable_thinking:true` set (an opaque kwarg to that template).
    func testModeDisabledForOtherModels() {
        XCTAssertEqual(
            qwenReasoningMode(
                modelName: "gemma-4-26b-a4b-it-4bit",
                chatTemplateKwargs: ["enable_thinking": true],
                isStructured: false),
            .disabled)
    }

    /// #198 Codex follow-up (round 2): a schema-guided or forced-tool-call
    /// response is Guide-masked from token 0, which suppresses `<think>`
    /// entirely — `isStructured` must force `.awaitingOpenTag` even when
    /// `enable_thinking` would otherwise resolve true, or the whole
    /// structured response (or tool call) gets swallowed into
    /// `reasoning_content` and `content` comes back empty.
    func testModeStructuredForcesAwaitingOpenTagEvenWithThinkingOn() {
        XCTAssertEqual(
            qwenReasoningMode(
                modelName: "Qwen3.5-27B-4bit", chatTemplateKwargs: nil,
                isStructured: true),
            .awaitingOpenTag)
        XCTAssertEqual(
            qwenReasoningMode(
                modelName: "Qwen3.5-27B-4bit",
                chatTemplateKwargs: ["enable_thinking": true],
                isStructured: true),
            .awaitingOpenTag)
    }
}
