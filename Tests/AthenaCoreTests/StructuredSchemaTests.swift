import Foundation
import XCTest

@testable import AthenaStructured

final class StructuredSchemaTests: XCTestCase {

    func testJSONValueRoundTripsSchemaObject() throws {
        let src = #"{"properties":{"n":{"type":"integer"}},"type":"object"}"#
        let v = try JSONDecoder().decode(JSONValue.self, from: Data(src.utf8))
        // sortedKeys ⇒ deterministic, equals the (already-sorted) source.
        XCTAssertEqual(v.jsonString(), src)
    }

    func testRouting() {
        let schema = JSONValue.object(["type": .string("object")])
        XCTAssertEqual(
            StructuredSchema.schemaJSON(
                responseFormatType: "json_schema", jsonSchema: schema),
            #"{"type":"object"}"#)
        XCTAssertEqual(
            StructuredSchema.schemaJSON(
                responseFormatType: "json_object", jsonSchema: nil),
            #"{"type":"object"}"#)
        XCTAssertNil(
            StructuredSchema.schemaJSON(
                responseFormatType: "text", jsonSchema: nil))
        XCTAssertNil(
            StructuredSchema.schemaJSON(
                responseFormatType: nil, jsonSchema: schema))
    }

    func testToolCallSchema() throws {
        let s = try XCTUnwrap(
            StructuredSchema.toolCallSchema(
                functionName: "get_weather",
                parameters: .object([
                    "type": .string("object"),
                    "properties": .object([
                        "city": .object(["type": .string("string")])
                    ]),
                ])))
        let v = try JSONDecoder().decode(JSONValue.self, from: Data(s.utf8))
        guard case .object(let o) = v,
            case .object(let props)? = o["properties"],
            case .object(let nameC)? = props["name"],
            case .string(let cnst)? = nameC["const"]
        else { return XCTFail("unexpected shape: \(s)") }
        XCTAssertEqual(cnst, "get_weather")
        XCTAssertNotNil(props["arguments"])
    }

    func testToolCallUnionSchema() throws {
        XCTAssertNil(StructuredSchema.toolCallUnionSchema(tools: []))
        let s = try XCTUnwrap(
            StructuredSchema.toolCallUnionSchema(tools: [
                (
                    "get_weather",
                    .object([
                        "type": .string("object"),
                        "properties": .object([
                            "city": .object(["type": .string("string")])
                        ]),
                    ])
                ),
                ("get_time", nil),
            ]))
        let v = try JSONDecoder().decode(JSONValue.self, from: Data(s.utf8))
        guard case .object(let o) = v,
            case .array(let branches)? = o["oneOf"]
        else { return XCTFail("expected oneOf array: \(s)") }
        XCTAssertEqual(branches.count, 2)
        var names: [String] = []
        for b in branches {
            guard case .object(let bo) = b,
                case .object(let props)? = bo["properties"],
                case .object(let nameC)? = props["name"],
                case .string(let cnst)? = nameC["const"]
            else { return XCTFail("bad branch shape: \(b)") }
            XCTAssertNotNil(props["arguments"])
            XCTAssertEqual(bo["additionalProperties"], .bool(false))
            names.append(cnst)
        }
        XCTAssertEqual(names.sorted(), ["get_time", "get_weather"])
    }

    func testFoundationValueLowersToNativeContainers() throws {
        let v: JSONValue = .object([
            "type": .string("object"),
            "properties": .object([
                "n": .object(["type": .string("integer")])
            ]),
            "flag": .bool(true),
            "items": .array([.number(1), .string("x")]),
        ])
        guard let o = v.foundationValue() as? [String: any Sendable]
        else { return XCTFail("root not a dict") }
        XCTAssertEqual(o["type"] as? String, "object")
        XCTAssertEqual(o["flag"] as? Bool, true)
        let props = o["properties"] as? [String: any Sendable]
        let n = props?["n"] as? [String: any Sendable]
        XCTAssertEqual(n?["type"] as? String, "integer")
        let items = o["items"] as? [any Sendable]
        XCTAssertEqual(items?.count, 2)
        XCTAssertEqual(items?[1] as? String, "x")
    }

    func testCompiledSchemaGuideWalks() throws {
        // Pure end-to-end: routed schema string → Index → Guide.
        let tokens = (0 ..< 10).map {
            VocabToken(id: UInt32($0), bytes: [UInt8(0x30 + $0)])
        }
        let schema = StructuredSchema.schemaJSON(
            responseFormatType: "json_schema",
            jsonSchema: .object(["type": .string("integer")]))
        let guide = try StructuredGuide(
            index: StructuredIndex(
                jsonSchema: try XCTUnwrap(schema),
                vocabulary: StructuredVocabulary(
                    tokens: tokens, eosTokenId: 10)))
        XCTAssertGreaterThan(guide.maskLength, 0)
    }

    /// G7: a schema integer constant above 2^53 must round-trip exactly.
    /// Decoded as a Double (2^53 + 1) it would collapse to its even
    /// neighbor; the `.integer(Int64)` case preserves it.
    func testIntegerConstantAbove2to53RoundTripsExactly() throws {
        let src = #"{"const":9007199254740993}"#  // 2^53 + 1
        let v = try JSONDecoder().decode(JSONValue.self, from: Data(src.utf8))
        guard case .object(let o) = v,
            case .integer(let i)? = o["const"]
        else { return XCTFail("expected .object with an .integer const") }
        XCTAssertEqual(i, 9_007_199_254_740_993)
        // Re-serialized exactly — no Double rounding to 9007199254740992.
        XCTAssertEqual(v.jsonString(), src)
    }
}
