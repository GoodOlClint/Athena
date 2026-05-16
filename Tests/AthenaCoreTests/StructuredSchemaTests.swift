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

    func testCompiledSchemaGuideWalks() throws {
        // Pure end-to-end: routed schema string → Index → Guide.
        let tokens = (0..<10).map {
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
}
