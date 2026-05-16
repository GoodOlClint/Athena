import XCTest

@testable import AthenaStructured

/// Model-free (no MLX/SSD): mirrors the Rust `cargo test` digit-vocab
/// walk through the safe Swift wrapper, exercising the full FFI
/// lifecycle (vocab → index → guide → mask/advance/rollback/free).
final class StructuredGuideTests: XCTestCase {

    private func digitVocab() throws -> StructuredVocabulary {
        // single-char byte tokens '0'..'9' → ids 0..9, eos id 10
        let tokens = (0..<10).map {
            VocabToken(id: UInt32($0), bytes: [UInt8(0x30 + $0)])
        }
        return try StructuredVocabulary(tokens: tokens, eosTokenId: 10)
    }

    func testRegexGuideWalkMaskAndRollback() throws {
        let guide = try StructuredGuide(
            index: StructuredIndex(
                regex: "[0-9][0-9]", vocabulary: digitVocab()))

        XCTAssertFalse(guide.isFinal)
        var mask = [UInt8]()
        XCTAssertTrue(guide.allowedMask(into: &mask))
        XCTAssertTrue(mask.contains { $0 != 0 }, "start allows digits")
        XCTAssertEqual(guide.allowedRollback, 0)

        XCTAssertTrue(guide.advance(3))
        XCTAssertFalse(guide.isFinal)        // need two digits
        XCTAssertEqual(guide.allowedRollback, 1)

        XCTAssertTrue(guide.advance(7))
        XCTAssertTrue(guide.isFinal)         // "37" complete
        XCTAssertEqual(guide.allowedRollback, 2)

        XCTAssertTrue(guide.rollback(2))
        XCTAssertFalse(guide.isFinal)
        XCTAssertEqual(guide.allowedRollback, 0)
        XCTAssertFalse(guide.rollback(1), "past recorded history")
    }

    func testInvalidTokenNotAdvanced() throws {
        let guide = try StructuredGuide(
            index: StructuredIndex(
                regex: "[0-9]", vocabulary: digitVocab()))
        // eos id (10) has no transition from the start state.
        XCTAssertFalse(guide.advance(10))
        XCTAssertEqual(guide.allowedRollback, 0)
    }

    func testJSONSchemaCompilesToGuide() throws {
        let guide = try StructuredGuide(
            index: StructuredIndex(
                jsonSchema: #"{"type":"integer"}"#,
                vocabulary: digitVocab()))
        XCTAssertGreaterThan(guide.maskLength, 0)
    }
}
