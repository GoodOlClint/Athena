import Foundation
import XCTest

@testable import AthenaTranscription

/// Pure (MLX-free) SentencePiece detokenization for Parakeet (ADR 020 S2).
/// Always runs in CI (ADR 008/009). Pins special-token stripping + the
/// leading-space trim that the spike's naive vocab-join lacked.
final class ParakeetDetokenizerTests: XCTestCase {

    /// A tiny vocab modelled on the real v3 layout: a block of `<…>` control
    /// tokens, then SentencePiece word pieces with the `▁` boundary marker.
    private let vocab = [
        "<unk>",  // 0
        "<|startoftranscript|>",  // 1
        "<|en|>",  // 2
        "\u{2581}hello",  // 3  ▁hello
        "\u{2581}world",  // 4  ▁world
        "s",  // 5
        "<pad>",  // 6
    ]

    func testStripsLeadingControlTokensAndTrim() {
        // <sot> <en> ▁hello ▁world s  →  "hello worlds"
        let out = ParakeetDetokenizer.detokenize([1, 2, 3, 4, 5], vocabulary: vocab)
        XCTAssertEqual(out, "hello worlds")
    }

    func testInteriorSpecialTokensDropped() {
        // ▁hello <pad> ▁world  →  "hello world" (the <pad> vanishes, spacing
        // from the surrounding ▁ markers is preserved).
        let out = ParakeetDetokenizer.detokenize([3, 6, 4], vocabulary: vocab)
        XCTAssertEqual(out, "hello world")
    }

    func testLeadingSpaceTrimmedOnce() {
        // First real piece carries ▁ → would render " hello"; trim exactly one.
        XCTAssertEqual(
            ParakeetDetokenizer.detokenize([3], vocabulary: vocab), "hello")
    }

    func testAllSpecialIsEmpty() {
        XCTAssertEqual(
            ParakeetDetokenizer.detokenize([0, 1, 2, 6], vocabulary: vocab), "")
    }

    func testOutOfRangeIdsSkipped() {
        XCTAssertEqual(
            ParakeetDetokenizer.detokenize([3, 999, -1, 4], vocabulary: vocab),
            "hello world")
    }

    func testIsSpecialClassification() {
        XCTAssertTrue(ParakeetDetokenizer.isSpecial("<unk>"))
        XCTAssertTrue(ParakeetDetokenizer.isSpecial("<|startoftranscript|>"))
        XCTAssertTrue(ParakeetDetokenizer.isSpecial("<|spltoken29|>"))
        XCTAssertFalse(ParakeetDetokenizer.isSpecial("\u{2581}hello"))
        XCTAssertFalse(ParakeetDetokenizer.isSpecial("world"))
        // A bare "<" or ">" is not a control token.
        XCTAssertFalse(ParakeetDetokenizer.isSpecial("<"))
        XCTAssertFalse(ParakeetDetokenizer.isSpecial(">"))
    }
}
