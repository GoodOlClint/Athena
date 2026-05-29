import Foundation
import XCTest

@testable import AthenaStructured

/// M49.5 — CI-safe unit coverage for the structured-output schema
/// complexity analyzer that gates outlines-core's DFA compile against
/// pathological shapes. The analyzer is pure JSON inspection — no MLX,
/// no model, no Rust shim — so it runs on every build.
final class SchemaComplexityTests: XCTestCase {

    // MARK: - Happy path: flat / shallow schemas pass without violation.

    func testFlatObjectHasNoArrayCounts() throws {
        let s = #"""
            {"type":"object","properties":{
              "name":{"type":"string"},
              "age":{"type":"integer"}
            }}
            """#
        let a = try XCTUnwrap(SchemaComplexity.analyze(s))
        XCTAssertEqual(a.boundedArrays, 0)
        XCTAssertEqual(a.unboundedInnerArrays, 0)
        XCTAssertEqual(a.unboundedTopLevelArrays, 0)
    }

    func testTopLevelUnboundedArrayIsNotPathological() throws {
        // A bare unbounded array (no outer bound) doesn't trigger the
        // pathology — there's no per-position state to multiply.
        let s = #"""
            {"type":"array","items":{"type":"string"}}
            """#
        let a = try XCTUnwrap(SchemaComplexity.analyze(s))
        XCTAssertEqual(a.boundedArrays, 0)
        XCTAssertEqual(a.unboundedInnerArrays, 0)
        XCTAssertEqual(a.unboundedTopLevelArrays, 1)
    }

    func testBoundedOuterWithBoundedInnerIsSafe() throws {
        // The compile pattern that works: outer + inner BOTH bounded.
        let s = #"""
            {"type":"array","maxItems":10,
             "items":{"type":"array","maxItems":5,
               "items":{"type":"string"}}}
            """#
        let a = try XCTUnwrap(SchemaComplexity.analyze(s))
        XCTAssertEqual(a.boundedArrays, 2)
        XCTAssertEqual(a.unboundedInnerArrays, 0)
    }

    // MARK: - The pathology: bounded outer + unbounded inner.

    func testBoundedOuterWithSingleUnboundedInnerIsCounted() throws {
        let s = #"""
            {"type":"array","maxItems":30,
             "items":{"type":"object","properties":{
               "tags":{"type":"array","items":{"type":"string"}}
             }}}
            """#
        let a = try XCTUnwrap(SchemaComplexity.analyze(s))
        XCTAssertEqual(a.boundedArrays, 1)
        XCTAssertEqual(a.unboundedInnerArrays, 1)
        XCTAssertEqual(a.unboundedTopLevelArrays, 0)
    }

    func testUnboundedNestedTwoDeepBothCount() throws {
        // Once we cross a bounded outer, ALL nested unbounded arrays
        // count — each one multiplies the per-position state.
        let s = #"""
            {"type":"array","maxItems":10,
             "items":{"type":"object","properties":{
               "outer":{"type":"array","items":{
                 "type":"object","properties":{
                   "inner":{"type":"array","items":{"type":"string"}}
                 }}}
             }}}
            """#
        let a = try XCTUnwrap(SchemaComplexity.analyze(s))
        XCTAssertEqual(a.boundedArrays, 1)
        XCTAssertEqual(a.unboundedInnerArrays, 2)
    }

    // MARK: - $ref / $defs resolution.

    func testDefsRefIsFollowed() throws {
        let s = #"""
            {"$defs":{
              "Item":{"type":"object","properties":{
                "tags":{"type":"array","items":{"type":"string"}}
              }}
            },
            "type":"array","maxItems":30,
            "items":{"$ref":"#/$defs/Item"}}
            """#
        let a = try XCTUnwrap(SchemaComplexity.analyze(s))
        XCTAssertEqual(a.boundedArrays, 1)
        XCTAssertEqual(a.unboundedInnerArrays, 1,
            "the unbounded array under the $ref must be counted")
    }

    func testCircularRefDoesNotInfiniteLoop() throws {
        // $defs/A references itself via $defs/B which references A.
        // The visiting-set must prevent infinite recursion.
        let s = #"""
            {"$defs":{
              "A":{"type":"object","properties":{
                "child":{"$ref":"#/$defs/B"}
              }},
              "B":{"type":"object","properties":{
                "back":{"$ref":"#/$defs/A"}
              }}
            },
            "$ref":"#/$defs/A"}
            """#
        let a = try XCTUnwrap(SchemaComplexity.analyze(s))
        XCTAssertEqual(a.boundedArrays, 0)
        XCTAssertEqual(a.unboundedInnerArrays, 0)
    }

    // MARK: - Composition keywords.

    func testAnyOfBranchesRecurseUnderTheSameBoundedFlag() throws {
        // Pydantic-style Optional → anyOf [{type:string}, {type:null}].
        // The array-with-anyOf-string-items pattern is the most common
        // shape from FastAPI/Pydantic; both branches must be walked.
        let s = #"""
            {"type":"array","maxItems":5,
             "items":{"anyOf":[
               {"type":"object","properties":{
                 "tags":{"type":"array","items":{"type":"string"}}
               }},
               {"type":"null"}
             ]}}
            """#
        let a = try XCTUnwrap(SchemaComplexity.analyze(s))
        XCTAssertEqual(a.boundedArrays, 1)
        XCTAssertEqual(a.unboundedInnerArrays, 1)
    }

    // MARK: - the consuming application's known pathological shape.

    func testthe consuming applicationShapeIsDetected() throws {
        // Faithful reduction of repro-request.json: events.maxItems=30
        // outer with multiple unbounded inner arrays per EventExtraction.
        let s = #"""
            {"$defs":{
              "Event":{"type":"object","properties":{
                "participants":{"type":"array","items":{"type":"string"}},
                "quotes":{"type":"array","items":{"type":"string"}},
                "tags":{"type":"array","items":{"type":"string"}},
                "refs":{"type":"array","items":{"type":"string"}},
                "ids":{"type":"array","items":{"type":"integer"}},
                "aliases":{"type":"array","items":{"type":"string"}}
              }}
            },
            "type":"object","properties":{
              "events":{"type":"array","maxItems":30,
                "items":{"$ref":"#/$defs/Event"}}
            }}
            """#
        let a = try XCTUnwrap(SchemaComplexity.analyze(s))
        XCTAssertEqual(a.boundedArrays, 1)
        XCTAssertEqual(a.unboundedInnerArrays, 6)
        XCTAssertGreaterThan(
            a.unboundedInnerArrayPaths.count, 0,
            "paths must be reported so the operator can fix them")
    }

    // MARK: - Reason-message construction is actionable.

    func testTooComplexReasonNamesTheCountsAndPaths() {
        var a = SchemaComplexity.Analysis(
            boundedArrays: 1, unboundedInnerArrays: 13,
            unboundedInnerArrayPaths: ["#/items/properties/quotes"])
        a.schemaBytes = 17273
        let r = SchemaComplexity.tooComplexReason(a, max: 5)
        XCTAssertTrue(r.contains("13"), "must name the actual count")
        XCTAssertTrue(r.contains("5"), "must name the limit")
        XCTAssertTrue(r.contains("maxItems"), "must guide the fix")
        XCTAssertTrue(
            r.contains("#/items/properties/quotes"),
            "must list a violating path so the caller knows where to edit")
        XCTAssertTrue(
            r.contains("structured_max_unbounded_subarrays"),
            "must document the operator escape hatch")
    }

    // MARK: - Robustness.

    func testMalformedJSONReturnsNilWithoutCrashing() {
        XCTAssertNil(SchemaComplexity.analyze("{not json"))
        XCTAssertNil(SchemaComplexity.analyze(""))
        // Non-object top level — outlines-core itself rejects this;
        // analyzer declines to opine.
        XCTAssertNil(SchemaComplexity.analyze("[]"))
        XCTAssertNil(SchemaComplexity.analyze("42"))
    }
}
