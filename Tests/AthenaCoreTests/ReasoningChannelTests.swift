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
/// one is model-gated (`qwenExpectsThinking`): the close-tag-only form has no
/// start marker to key on, so the filter must be told up front whether the
/// prompt already opened the block. MLX-free, fast tier (ADR 009).
final class QwenThinkFilterTests: XCTestCase {
    /// The exact observed shape (2026-09-23, `Qwen3.5-27B-4bit`, greedy,
    /// default `enable_thinking`): the completion carries ONLY the close
    /// marker — the prompt already opened the block.
    func testCloseTagOnlyForm() {
        let raw =
            "Thinking Process:\n\n1.  Analyze.\n\n5.  Construct: 4.\n</think>\n\n4"
        let r = splitQwenThink(raw, expectThinking: true)
        XCTAssertEqual(r.content, "\n\n4")
        XCTAssertEqual(r.reasoning, "Thinking Process:\n\n1.  Analyze.\n\n5.  Construct: 4.\n")
    }

    /// Paired form is honored even when `expectThinking` is false (a
    /// differently-configured template that emits the open tag itself).
    func testPairedTagFormWithoutExpectThinking() {
        let raw = "<think>reasoning</think>content"
        let r = splitQwenThink(raw, expectThinking: false)
        XCTAssertEqual(r.content, "content")
        XCTAssertEqual(r.reasoning, "reasoning")
    }

    /// `enable_thinking:false` (or any non-Qwen model): no markers at all —
    /// the filter must be a pure no-op, exactly the observed disabled-form
    /// bytes (2026-09-23, same checkpoint, `enable_thinking:false`): raw
    /// completion was the literal string "4", nothing else.
    func testNoThinkingIsPassthrough() {
        let r = splitQwenThink("4", expectThinking: false)
        XCTAssertEqual(r.content, "4")
        XCTAssertEqual(r.reasoning, "")
    }

    /// Streaming, byte-by-byte, must equal the one-shot split for both forms.
    func testStreamingEqualsOneShotCloseTagOnly() {
        let full = "reasoning text here\n</think>\n\nfinal answer"
        let oneShot = splitQwenThink(full, expectThinking: true)
        var f = QwenThinkFilter(expectThinking: true)
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
        let oneShot = splitQwenThink(full, expectThinking: false)
        var f = QwenThinkFilter(expectThinking: false)
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

    // MARK: qwenExpectsThinking

    func testExpectsThinkingDefaultTrueForQwen35() {
        XCTAssertTrue(
            qwenExpectsThinking(
                modelName: "Qwen3.5-27B-4bit", chatTemplateKwargs: nil))
    }

    func testExpectsThinkingExplicitFalse() {
        XCTAssertFalse(
            qwenExpectsThinking(
                modelName: "Qwen3.5-27B-4bit",
                chatTemplateKwargs: ["enable_thinking": false]))
    }

    func testExpectsThinkingExplicitTrue() {
        XCTAssertTrue(
            qwenExpectsThinking(
                modelName: "Qwen3.5-27B-4bit",
                chatTemplateKwargs: ["enable_thinking": true]))
    }

    /// Non-Qwen models never enter the reasoning-implicit initial state, even
    /// with `enable_thinking:true` set (an opaque kwarg to that template) —
    /// the model-gate is what keeps this filter a safe no-op elsewhere.
    func testExpectsThinkingFalseForOtherModels() {
        XCTAssertFalse(
            qwenExpectsThinking(
                modelName: "gemma-4-26b-a4b-it-4bit",
                chatTemplateKwargs: ["enable_thinking": true]))
    }
}
