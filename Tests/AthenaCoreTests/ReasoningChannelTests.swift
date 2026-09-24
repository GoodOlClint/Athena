import AthenaCore
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

/// #198 — the close-tag-only reasoning form (Qwen3.5's `<think>…</think>`,
/// and any other model whose chat template pre-opens a reasoning block it
/// never closes). Unlike model-gating (rejected — see ADR 035's amendment),
/// `qwenReasoningMode` is **prompt-derived**: the caller passes
/// `startsInReasoning`, computed from the actual rendered prompt
/// (`ReasoningPromptTail`, `AthenaCoreTests`), never a model-name guess.
/// `.awaitingOpenTag` (the default) is a marker-driven no-op exactly like
/// `ReasoningChannelFilter` — it only touches text when a literal `<think>`
/// actually appears, universal across every model, same as ADR 035's
/// original design. MLX-free, fast tier (ADR 009).
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

    /// Paired form is honored in `.awaitingOpenTag` (the prompt did NOT
    /// open the block — e.g. `enable_thinking: false`, or a differently
    /// configured template that emits the open tag itself).
    func testPairedTagFormAwaitingOpenTag() {
        let raw = "<think>reasoning</think>content"
        let r = splitQwenThink(raw, mode: .awaitingOpenTag)
        XCTAssertEqual(r.content, "content")
        XCTAssertEqual(r.reasoning, "reasoning")
    }

    /// `.awaitingOpenTag` must not touch ordinary content that never
    /// contains the marker at all — the no-op case every non-Qwen response,
    /// and every Qwen response with thinking off, falls into.
    func testNoMarkerIsPassthrough() {
        let raw = "Explain the concept of a state machine."
        let r = splitQwenThink(raw, mode: .awaitingOpenTag)
        XCTAssertEqual(r.content, raw)
        XCTAssertEqual(r.reasoning, "")
    }

    /// `enable_thinking:false`: no markers at all — the filter must be a
    /// pure no-op, exactly the observed disabled-form bytes (2026-09-23,
    /// same checkpoint, `enable_thinking:false`): raw completion was the
    /// literal string "4", nothing else.
    func testNoThinkingIsPassthrough() {
        let r = splitQwenThink("4", mode: .awaitingOpenTag)
        XCTAssertEqual(r.content, "4")
        XCTAssertEqual(r.reasoning, "")
    }

    /// Streaming, byte-by-byte, must equal the one-shot split for both modes.
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

    /// Codex adversarial review, PR #213 round 4: `.passthrough` must not
    /// scan for markers at all — a literal paired `<think>…</think>` inside
    /// a schema value or tool argument must survive byte-for-byte, not be
    /// read as reasoning and stripped out of the structured data.
    func testPassthroughLeavesLiteralTagsInContentUntouched() {
        let raw = #"{"value":"<think>x</think>"}"#
        let r = splitQwenThink(raw, mode: .passthrough)
        XCTAssertEqual(r.content, raw)
        XCTAssertEqual(r.reasoning, "")
    }

    func testStreamingEqualsOneShotPassthrough() {
        let full = #"{"value":"<think>x</think>","n":1}"#
        let oneShot = splitQwenThink(full, mode: .passthrough)
        var f = QwenThinkFilter(mode: .passthrough)
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

    func testModeReasoningOpenWhenPromptStartsInReasoning() {
        XCTAssertEqual(
            qwenReasoningMode(startsInReasoning: true, isStructured: false),
            .reasoningOpen)
    }

    func testModeAwaitingOpenTagWhenPromptDoesNotStartInReasoning() {
        XCTAssertEqual(
            qwenReasoningMode(startsInReasoning: false, isStructured: false),
            .awaitingOpenTag)
    }

    /// #198 Codex follow-up (round 4): `.awaitingOpenTag` still scans for
    /// literal `<think>…</think>` pairs, which would corrupt a schema value
    /// or tool argument containing that text — `isStructured` must force the
    /// true no-op `.passthrough`, not `.awaitingOpenTag`.
    func testModeStructuredForcesPassthroughEvenWhenPromptOpened() {
        XCTAssertEqual(
            qwenReasoningMode(startsInReasoning: true, isStructured: true),
            .passthrough)
    }

    func testModeStructuredForcesPassthroughWhenPromptDoesNotOpen() {
        XCTAssertEqual(
            qwenReasoningMode(startsInReasoning: false, isStructured: true),
            .passthrough)
    }
}

/// #198 (ADR 035 amendment) — the prompt-derived signal `qwenReasoningMode`
/// consumes. Pure String scanning, MLX-free (`AthenaCore`, not
/// `AthenaServerKit` — see `ReasoningPromptTail`'s doc comment for why).
final class ReasoningPromptTailTests: XCTestCase {
    /// The exact suffix Qwen3.5's chat template renders when thinking is
    /// active (`enable_thinking` unset or `true`).
    func testDetectsUnclosedOpenTag() {
        XCTAssertTrue(
            ReasoningPromptTail.startsInOpenBlock(
                "<|im_start|>assistant\n<think>\n"))
    }

    /// The exact suffix when `enable_thinking: false` — both tags present,
    /// already closed, nothing open.
    func testClosedBlockIsNotOpen() {
        XCTAssertFalse(
            ReasoningPromptTail.startsInOpenBlock(
                "<|im_start|>assistant\n<think>\n\n</think>\n\n"))
    }

    func testNoMarkerAtAllIsNotOpen() {
        XCTAssertFalse(
            ReasoningPromptTail.startsInOpenBlock("<|im_start|>assistant\n"))
    }

    /// A prior turn's reconstructed reasoning (the template's own history
    /// re-serialization emits `<think>…</think>` pairs for earlier assistant
    /// messages) must not be mistaken for an open block at the tail.
    func testEarlierClosedPairThenOpenGenerationPromptIsOpen() {
        XCTAssertTrue(
            ReasoningPromptTail.startsInOpenBlock(
                "<think>\nprior reasoning\n</think>\n\nprior answer<|im_end|>\n<|im_start|>assistant\n<think>\n"
            ))
    }

    func testEarlierClosedPairWithNoNewOpenIsNotOpen() {
        XCTAssertFalse(
            ReasoningPromptTail.startsInOpenBlock(
                "<think>\nprior reasoning\n</think>\n\nprior answer<|im_end|>\n"
            ))
    }

    /// Codex adversarial review, PR #213 round 3: a USER message ending in
    /// the literal text `<think>` is followed by real template boilerplate
    /// (the generation-prompt suffix), not whitespace — must read as closed.
    func testUserContentEndingInLiteralOpenTagIsNotForgeable() {
        XCTAssertFalse(
            ReasoningPromptTail.startsInOpenBlock(
                "Please explain the <think> tag convention<|im_end|>\n<|im_start|>assistant\n"
            ))
    }

    /// An open marker with trailing whitespace only (Qwen3.5's exact shape,
    /// `<think>\n`) is still open — the whitespace-only requirement must not
    /// be stricter than the real template output.
    func testOpenMarkerFollowedByOnlyWhitespaceIsOpen() {
        XCTAssertTrue(
            ReasoningPromptTail.startsInOpenBlock("<think>\n  \n"))
    }
}
