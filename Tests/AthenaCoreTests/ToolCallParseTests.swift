import Foundation
import XCTest

@testable import AthenaCore

/// Pins the tool-call parse that both the streaming (`pumpTokens`) and
/// non-streaming (`chatChoice`) chat paths use to surface OpenAI `tool_calls`.
/// The streaming-mode bug (2026-06-30): tool JSON leaked through as
/// `delta.content` because the stream path didn't parse it. MLX-free, so it
/// runs in the fast `swift test` tier (ADR 009).
final class ToolCallParseTests: XCTestCase {
    func testNameFirst() throws {
        let call = try XCTUnwrap(
            parseToolCall(#"{"name":"web_search","arguments":{"query":"x"}}"#))
        XCTAssertEqual(call.name, "web_search")
        XCTAssertEqual(call.argsJSON, #"{"query":"x"}"#)
    }

    /// The literal field-incident: Gemma 4 emits `arguments` BEFORE `name`.
    /// JSON key order must not matter.
    func testArgumentsFirst() throws {
        let call = try XCTUnwrap(
            parseToolCall(
                "{\n  \"arguments\": {\n    \"query\": \"x\"\n  },\n"
                    + "  \"name\": \"web_search\"\n}"))
        XCTAssertEqual(call.name, "web_search")
        XCTAssertEqual(call.argsJSON, #"{"query":"x"}"#)
    }

    /// Arguments are re-serialized with sorted keys → stable string regardless
    /// of the model's emission order (the OpenAI `arguments` is a string).
    func testArgsKeysSorted() throws {
        let call = try XCTUnwrap(
            parseToolCall(#"{"name":"f","arguments":{"b":2,"a":1}}"#))
        XCTAssertEqual(call.argsJSON, #"{"a":1,"b":2}"#)
    }

    func testMissingArgumentsDefaultsEmptyObject() throws {
        let call = try XCTUnwrap(parseToolCall(#"{"name":"ping"}"#))
        XCTAssertEqual(call.name, "ping")
        XCTAssertEqual(call.argsJSON, "{}")
    }

    /// Plain assistant text / malformed JSON ⇒ nil ⇒ caller streams it as
    /// `content`. Guards against treating ordinary chat as a tool call.
    func testNonToolCallReturnsNil() {
        XCTAssertNil(parseToolCall("hello there"))
        XCTAssertNil(parseToolCall(#"{"arguments":{"query":"x"}}"#))  // no name
        XCTAssertNil(parseToolCall(#"{"name":"#))  // truncated JSON
    }

    // MARK: - toolArgumentsJSON (ADR 034 — shared by parse + substrate path)

    func testToolArgumentsJSONSortsKeys() {
        XCTAssertEqual(
            toolArgumentsJSON(["b": 2, "a": 1] as [String: Any]),
            #"{"a":1,"b":2}"#)
    }

    func testToolArgumentsJSONNilAndNonObjectDefaultEmpty() {
        XCTAssertEqual(toolArgumentsJSON(nil), "{}")
        XCTAssertEqual(toolArgumentsJSON("not an object"), "{}")
    }

    func testToolArgumentsJSONEmptyObject() {
        XCTAssertEqual(toolArgumentsJSON([String: Any]()), "{}")
    }

    // MARK: - resolveToolCallOutcome (WP7 — the one precedence algebra all four
    // OpenAI/Anthropic × streaming/non-streaming encode sites switch on)

    /// A substrate-detected free call (ADR 034) wins over everything and the
    /// assistant text is kept as content — even when the Guide is engaged.
    func testDetectedWins() {
        XCTAssertEqual(
            resolveToolCallOutcome(
                detected: (name: "f", argsJSON: #"{"a":1}"#),
                text: "some reasoning", isToolCall: true),
            .detected(name: "f", argsJSON: #"{"a":1}"#))
    }

    /// Guide-forced (required/named) with no free call: the text IS the call.
    func testForcedParsesText() {
        XCTAssertEqual(
            resolveToolCallOutcome(
                detected: nil,
                text: #"{"name":"web_search","arguments":{"q":"x"}}"#,
                isToolCall: true),
            .forced(name: "web_search", argsJSON: #"{"q":"x"}"#))
    }

    /// Guide-forced but the buffer didn't parse (e.g. truncated) ⇒ `.none` so
    /// the caller surfaces the raw text rather than silently dropping it.
    func testForcedUnparseableFallsThroughToNone() {
        XCTAssertEqual(
            resolveToolCallOutcome(
                detected: nil, text: #"{"name":"#, isToolCall: true),
            .none)
    }

    /// Plain assistant turn (no forcing, no free call) ⇒ `.none`.
    func testPlainTextIsNone() {
        XCTAssertEqual(
            resolveToolCallOutcome(
                detected: nil, text: "hello there", isToolCall: false),
            .none)
    }

    /// Well-formed tool JSON but NOT a forced turn and NOT detected ⇒ still
    /// `.none` (auto mode surfaces a free call via the substrate event, not by
    /// re-parsing ordinary content).
    func testUnforcedNeverParsesText() {
        XCTAssertEqual(
            resolveToolCallOutcome(
                detected: nil, text: #"{"name":"f","arguments":{}}"#,
                isToolCall: false),
            .none)
    }
}
