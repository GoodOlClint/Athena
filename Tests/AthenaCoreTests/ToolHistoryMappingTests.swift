import Foundation
import MLXLMCommon
import XCTest

@testable import AthenaLLM

/// ADR 034 — pins the tool-history args round-trip: an assistant tool call's
/// stringified `arguments` (carried on a `ChatTurn`) must parse back into the
/// substrate `[String: JSONValue]` so the chat template renders a coherent
/// call→result history. Pure mapping (no MLX kernels) — fast tier (ADR 009).
final class ToolHistoryMappingTests: XCTestCase {
    /// `toolArgsObject(argsJSON)` re-encodes to the same sorted JSON — i.e. the
    /// arguments survive the string→object→template round-trip without loss.
    func testArgsRoundTrip() throws {
        let argsJSON = #"{"a":1,"query":"weather"}"#
        let obj = MLXLLMModule.toolArgsObject(argsJSON)
        XCTAssertEqual(Set(obj.keys), ["a", "query"])
        let reencoded = try JSONEncoder.sortedKeys().encode(obj)
        XCTAssertEqual(
            String(decoding: reencoded, as: UTF8.self), argsJSON)
    }

    func testMalformedArgsBecomeEmpty() {
        XCTAssertTrue(MLXLLMModule.toolArgsObject("not json").isEmpty)
        XCTAssertTrue(MLXLLMModule.toolArgsObject("[1,2,3]").isEmpty)  // array
        XCTAssertTrue(MLXLLMModule.toolArgsObject("").isEmpty)
    }
}

extension JSONEncoder {
    fileprivate static func sortedKeys() -> JSONEncoder {
        let e = JSONEncoder()
        e.outputFormatting = [.sortedKeys]
        return e
    }
}
