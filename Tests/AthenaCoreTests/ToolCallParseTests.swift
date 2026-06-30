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
}
