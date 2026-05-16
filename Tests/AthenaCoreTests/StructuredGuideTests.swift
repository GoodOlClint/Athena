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

    // MARK: - Issue #2 opener realignment

    func testOpenerAliasesMapsWhitespacePrefixedOpeners() {
        let toks = [
            VocabToken(id: 0, bytes: [0x7B]),  // {
            VocabToken(id: 1, bytes: [0x7B, 0x22]),  // {"
            VocabToken(id: 2, bytes: [0x5B]),  // [
            VocabToken(id: 3, bytes: [0x20, 0x7B]),  // " {"
            VocabToken(id: 4, bytes: [0x20, 0x7B, 0x22]),  // " {\""
            VocabToken(id: 5, bytes: [0x0A, 0x5B]),  // "\n["
            VocabToken(id: 6, bytes: [0x20, 0x66]),  // " f" (not opener)
            VocabToken(id: 7, bytes: [0x20, 0x7D]),  // " }" (not opener)
            // " {{" — no bare token with bytes "{{" ⇒ not aliased.
            VocabToken(id: 8, bytes: [0x20, 0x7B, 0x7B]),
        ]
        let a = StructuredVocabulary.openerAliases(tokens: toks)
        XCTAssertEqual(a[3], 0)  // " {" → {
        XCTAssertEqual(a[4], 1)  // " {\"" → {"
        XCTAssertEqual(a[5], 2)  // "\n[" → [
        XCTAssertNil(a[6])  // non-opener ws token
        XCTAssertNil(a[7])  // " }" is not an opener
        XCTAssertNil(a[8])  // no bare "{{" token in vocab
        XCTAssertNil(a[0])  // bare opener is never a key
        XCTAssertEqual(a.count, 3)
    }

    func testAdvanceOpenerTolerantHonorsSpacePrefixedOpener() throws {
        // bytes: 0={ 1=" {" 2..11=0..9 12=}
        var toks = [
            VocabToken(id: 0, bytes: [0x7B]),
            VocabToken(id: 1, bytes: [0x20, 0x7B]),
        ]
        toks += (0..<10).map {
            VocabToken(id: UInt32(2 + $0), bytes: [UInt8(0x30 + $0)])
        }
        toks.append(VocabToken(id: 12, bytes: [0x7D]))
        let vocab = try StructuredVocabulary(tokens: toks, eosTokenId: 13)
        let alias = StructuredVocabulary.openerAliases(tokens: toks)
        XCTAssertEqual(alias[1], 0, " { must alias to {")

        let guide = try StructuredGuide(
            index: StructuredIndex(
                regex: #"\{[0-9]\}"#, vocabulary: vocab))
        // Raw space-prefixed opener has no start transition…
        XCTAssertFalse(guide.advance(1))
        // …but the tolerant advance realigns via the alias.
        guide.openerAlias = alias
        XCTAssertTrue(guide.advanceOpenerTolerant(1))
        XCTAssertTrue(guide.advance(7))  // digit '5'
        XCTAssertTrue(guide.advance(12))  // }
        XCTAssertTrue(guide.isFinal)
    }

    func testAdvanceOpenerTolerantNoAliasStillStrict() throws {
        let guide = try StructuredGuide(
            index: StructuredIndex(
                regex: "[0-9][0-9]", vocabulary: digitVocab()))
        // No openerAlias set ⇒ behaves exactly like `advance`.
        XCTAssertFalse(guide.advanceOpenerTolerant(99))
        XCTAssertTrue(guide.advanceOpenerTolerant(3))
        XCTAssertEqual(guide.allowedRollback, 1)
    }
}
