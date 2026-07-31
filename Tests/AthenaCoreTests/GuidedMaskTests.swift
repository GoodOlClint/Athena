import AthenaStructured
import Foundation
import XCTest

@testable import AthenaLLM

/// L5 (M70.3) — the schema-mask seam. Structured enforcement was only
/// validated model-on (env-gated); there was no CI test that an off-schema
/// logit is actually suppressed. `GuidedMask` is the MLX-free seam both
/// guided greedy paths share, so scripted logits drive it directly.
final class GuidedMaskTests: XCTestCase {

    /// allowed bit set ⇒ 0 (kept); clear ⇒ -inf (suppressed).
    func testAdditiveMaskUnpacksBits() {
        // vocab 8, allow {2, 5}: byte0 bits 2 and 5 ⇒ 0b0010_0100 = 0x24.
        let add = GuidedMask.additiveMask(allowed: [0x24], vocab: 8)
        for i in 0 ..< 8 {
            if i == 2 || i == 5 {
                XCTAssertEqual(add[i], 0, "token \(i) allowed ⇒ 0")
            } else {
                XCTAssertEqual(add[i], -.infinity, "token \(i) suppressed")
            }
        }
    }

    /// The unmasked argmax is an OFF-SCHEMA token; the masked pick must be the
    /// highest ALLOWED token — the core schema-enforcement guarantee.
    func testMaskedArgmaxSuppressesOffSchemaPeak() {
        // allow {2, 5}; logits peak at the disallowed index 0.
        let logits: [Float] = [9, 9, 1, 9, 9, 2, 9, 9]
        let pick = GuidedMask.maskedArgmax(
            logits: logits, allowed: [0x24], vocab: 8)
        XCTAssertEqual(pick, 5, "5 is the highest-logit ALLOWED token")
    }

    /// Equal-logit allowed tokens tie-break to the lowest index, matching MLX
    /// `argMax` (first occurrence of the max) — so the CI value equals what
    /// the MLXArray path commits.
    func testMaskedArgmaxTieBreaksLowestIndex() {
        var logits = [Float](repeating: 0, count: 8)
        logits[2] = 5
        logits[5] = 5  // tie between the two allowed tokens
        let pick = GuidedMask.maskedArgmax(
            logits: logits, allowed: [0x24], vocab: 8)
        XCTAssertEqual(pick, 2, "lowest index wins the tie (MLX argMax order)")
    }

    // Byte ids: token id == byte value (mirrors StructuredGuideTests).
    private func byteVocab() throws -> StructuredVocabulary {
        let tokens = (0 ..< 256).map {
            VocabToken(id: UInt32($0), bytes: [UInt8($0)])
        }
        return try StructuredVocabulary(tokens: tokens, eosTokenId: 256)
    }

    /// End-to-end through a REAL guide: an integer schema allows only digits
    /// (and '-') at the opener. Script the global logit peak onto a letter
    /// ('a', off-schema); the seam must instead pick the highest-logit digit.
    func testRealIntegerGuideForcesDigitOverLetter() throws {
        let vocab = 257
        let guide = try StructuredGuide(
            index: StructuredIndex(
                jsonSchema: #"{"type":"integer"}"#, vocabulary: byteVocab()))
        var mask = [UInt8]()
        XCTAssertTrue(guide.allowedMask(into: &mask))

        // Sanity: a letter is NOT allowed, a digit IS, at the integer opener.
        let aByte = Int(UInt8(ascii: "a"))
        let seven = Int(UInt8(ascii: "7"))
        func allows(_ id: Int) -> Bool {
            (mask[id >> 3] >> UInt8(id & 7)) & 1 == 1
        }
        XCTAssertFalse(allows(aByte), "'a' not allowed for an integer")
        XCTAssertTrue(allows(seven), "'7' allowed at the integer opener")

        // Peak on the disallowed letter; '7' is the highest allowed.
        var logits = [Float](repeating: 0, count: vocab)
        logits[aByte] = 10
        logits[seven] = 5
        let pick = GuidedMask.maskedArgmax(
            logits: logits, allowed: mask, vocab: vocab)
        XCTAssertEqual(
            pick, seven,
            "off-schema 'a' peak suppressed; highest allowed digit '7' wins")
        // And the picked token actually advances the real guide.
        XCTAssertTrue(guide.advance(UInt32(pick)))
        XCTAssertTrue(guide.isFinal, "\"7\" is a complete integer")
    }
}
