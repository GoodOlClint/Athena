import Foundation
import XCTest

@testable import AthenaLLM
import AthenaStructured

/// NC6 (M70.3) — `StructuredVocab.build` is the tokenizer→Guide byte mapping.
/// Its doc warns "a mismatch would make the guide mask the wrong tokens," yet
/// it had zero coverage. The pure core (extracted from `tokens(tokenizer:)`)
/// is MLX- and tokenizer-free, so a fixed decode map pins the three
/// load-bearing behaviors: per-token UTF-8 bytes, eos exclusion + separate
/// return, and the C12 no-eos sentinel (`vocabSize`, never a real id).
final class StructuredVocabTests: XCTestCase {

    /// Per-token bytes = the UTF-8 of decoding that single id (the Python
    /// reference's `convert_tokens_to_string([id])`), multi-byte included.
    func testPerTokenBytesAreUTF8OfDecode() {
        let map: [Int: String] = [0: "a", 1: "bc", 2: "é", 3: "🜨"]
        let (tokens, _) = StructuredVocab.build(
            vocabSize: 4, eos: nil) { map[$0] ?? "" }
        // eos==nil ⇒ sentinel 4, so all 4 real ids survive.
        XCTAssertEqual(tokens.count, 4)
        XCTAssertEqual(tokens[0].id, 0)
        XCTAssertEqual(tokens[0].bytes, Array("a".utf8))
        XCTAssertEqual(tokens[1].bytes, Array("bc".utf8))
        XCTAssertEqual(tokens[2].bytes, Array("é".utf8))   // 2-byte
        XCTAssertEqual(tokens[3].bytes, Array("🜨".utf8))  // 4-byte
    }

    /// A real eos id is excluded from the trie tokens and returned separately,
    /// so the guide's allowed set never contains it (the decode loop stops on
    /// the tokenizer's own eos, not via the trie).
    func testKnownEosExcludedAndReturned() {
        let (tokens, eos) = StructuredVocab.build(
            vocabSize: 5, eos: 2) { "t\($0)" }
        XCTAssertEqual(eos, 2)
        XCTAssertEqual(tokens.map(\.id), [0, 1, 3, 4], "eos id 2 dropped")
        XCTAssertFalse(
            tokens.contains { $0.id == 2 }, "eos must not be a trie token")
    }

    /// C12: with no eos, the sentinel is `vocabSize` (one past the real
    /// range), NOT a real id like `vocabSize-1`. The loop never hits it, so
    /// EVERY real token (including the last) survives — a `vocabSize-1`
    /// fallback would silently drop the highest token from the guide.
    func testNoEosSentinelDropsNoRealToken() {
        let (tokens, eos) = StructuredVocab.build(
            vocabSize: 8, eos: nil) { "t\($0)" }
        XCTAssertEqual(eos, 8, "sentinel is one past the real range")
        XCTAssertEqual(tokens.count, 8, "no real id dropped")
        XCTAssertEqual(tokens.map(\.id), Array(0..<8))
        XCTAssertTrue(
            tokens.contains { $0.id == 7 },
            "the highest real token must NOT be sacrificed as a phantom eos")
    }

    /// The built tokens actually compile into a working guide via the real
    /// shim — proves the byte mapping is consumable end-to-end (the only step
    /// past `build` that stays in CI), not just structurally plausible.
    func testBuiltTokensCompileToAGuide() throws {
        // A tiny digit vocab: ids 0..9 decode to "0".."9", id 10 is eos.
        let (tokens, eos) = StructuredVocab.build(
            vocabSize: 11, eos: 10) { String($0) }
        let vocab = try StructuredVocabulary(
            tokens: tokens, eosTokenId: eos)
        let guide = try StructuredGuide(
            index: StructuredIndex(
                jsonSchema: #"{"type":"integer"}"#, vocabulary: vocab))
        XCTAssertGreaterThan(guide.maskLength, 0)
        XCTAssertTrue(guide.advance(UInt32(4)), "'4' (id 4) advances")
        XCTAssertTrue(guide.isFinal)
    }
}
