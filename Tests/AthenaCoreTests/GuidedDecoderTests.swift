import Foundation
import XCTest

@testable import AthenaLLM
import AthenaStructured

/// NC6 (M70.3) — the `GuidedDecoder` IDLE→ENFORCING phase machine guarantees
/// non-empty, schema-valid structured output (idleBudget-forced enforcement,
/// `forceEnforce` on EOS-in-IDLE, `jsonStart` tracking that drives the
/// prefix-drop, tolerant opener advance). A regression silently returns empty
/// or unconstrained structured output for every guided request. `commit` and
/// `forceEnforce` are pure (only `pick` touches MLX), so they're driven here
/// against a REAL byte-vocab guide — no Metal.
final class GuidedDecoderTests: XCTestCase {

    // Byte ids: token id == byte value (mirrors StructuredGuideTests).
    private func byteVocab() throws -> StructuredVocabulary {
        let tokens = (0..<256).map {
            VocabToken(id: UInt32($0), bytes: [UInt8($0)])
        }
        return try StructuredVocabulary(tokens: tokens, eosTokenId: 256)
    }
    private func integerGuide() throws -> StructuredGuide {
        try StructuredGuide(
            index: StructuredIndex(
                jsonSchema: #"{"type":"integer"}"#, vocabulary: byteVocab()))
    }
    private func objectGuide() throws -> StructuredGuide {
        let schema = """
            {"type":"object","properties":{"n":{"type":"integer"}},\
            "required":["n"],"additionalProperties":false}
            """
        return try StructuredGuide(
            index: StructuredIndex(jsonSchema: schema, vocabulary: byteVocab()))
    }
    private let a = Int(UInt8(ascii: "a"))
    private func digit(_ d: Int) -> Int { 0x30 + d }
    private let lbrace = Int(UInt8(ascii: "{"))
    private let quote = Int(UInt8(ascii: "\""))
    private let colon = Int(UInt8(ascii: ":"))
    private let nByte = Int(UInt8(ascii: "n"))

    /// no guide ⇒ every commit is .unconstrained and nothing enforces.
    func testNoGuideIsUnconstrained() {
        var d = GuidedDecoder(guide: nil, vocab: 257, idleBudget: 4)
        XCTAssertEqual(d.commit(digit(5)), .unconstrained)
        XCTAssertFalse(d.enforcing)
        XCTAssertFalse(d.jsonStarted)
    }

    /// The opener (first guide-accepted token) flips IDLE→ENFORCING, returns
    /// .jsonStart exactly once, and every subsequent enforced token is
    /// .jsonBody — this is the jsonStart bookkeeping that drives result()'s
    /// prefix-drop.
    func testOpenerStartsEnforcingAndJsonStartFiresOnce() throws {
        var d = GuidedDecoder(guide: try integerGuide(), vocab: 257, idleBudget: 8)
        XCTAssertFalse(d.enforcing)
        // A digit is the integer schema's opener: accepted by the fresh guide.
        XCTAssertEqual(d.commit(digit(4)), .jsonStart)
        XCTAssertTrue(d.enforcing)
        XCTAssertTrue(d.jsonStarted)
        // Subsequent enforced tokens are body, never another jsonStart.
        XCTAssertEqual(d.commit(digit(2)), .jsonBody)
    }

    /// In IDLE a non-opener token is dropped (.idlePrefix) and does not start
    /// enforcement — the unconstrained <think>/preamble prefix.
    func testIdleNonOpenerIsDroppedPrefix() throws {
        var d = GuidedDecoder(guide: try integerGuide(), vocab: 257, idleBudget: 8)
        XCTAssertEqual(d.commit(a), .idlePrefix, "'a' has no integer opener")
        XCTAssertFalse(d.enforcing)
        XCTAssertFalse(d.jsonStarted)
    }

    /// idleBudget non-opener tokens FORCE enforcement, so the model can't run
    /// past the budget emitting an endless unconstrained preamble. The forcing
    /// commit is still .idlePrefix; the NEXT commit is the enforced opener.
    func testIdleBudgetForcesEnforcement() throws {
        var d = GuidedDecoder(guide: try integerGuide(), vocab: 257, idleBudget: 3)
        XCTAssertEqual(d.commit(a), .idlePrefix)
        XCTAssertFalse(d.enforcing)
        XCTAssertEqual(d.commit(a), .idlePrefix)
        XCTAssertFalse(d.enforcing)
        XCTAssertEqual(d.commit(a), .idlePrefix, "3rd non-opener hits budget")
        XCTAssertTrue(d.enforcing, "idleBudget reached ⇒ forced enforcing")
        XCTAssertTrue(d.jsonStarted)
        // The rejected advances never mutated the guide, so the digit opener
        // still advances; first enforced commit is the jsonStart.
        XCTAssertEqual(d.commit(digit(7)), .jsonStart)
        XCTAssertEqual(d.commit(digit(1)), .jsonBody)
    }

    /// forceEnforce (called by the loop when the model emits EOS while still
    /// in IDLE) flips state WITHOUT consuming a token: no commit happens, and
    /// the first subsequent commit is the jsonStart.
    func testForceEnforceFlipsWithoutConsuming() throws {
        var d = GuidedDecoder(guide: try integerGuide(), vocab: 257, idleBudget: 8)
        d.forceEnforce()
        XCTAssertTrue(d.enforcing)
        XCTAssertTrue(d.jsonStarted)
        XCTAssertEqual(d.commit(digit(9)), .jsonStart, "no token was consumed")
    }

    /// forceEnforce is idempotent / inert once already enforcing and a no-op
    /// without a guide.
    func testForceEnforceInertCases() throws {
        var withGuide = GuidedDecoder(
            guide: try integerGuide(), vocab: 257, idleBudget: 8)
        withGuide.forceEnforce()
        withGuide.forceEnforce()  // second call must not re-arm jsonStart
        XCTAssertEqual(withGuide.commit(digit(3)), .jsonStart)

        var noGuide = GuidedDecoder(guide: nil, vocab: 257, idleBudget: 8)
        noGuide.forceEnforce()
        XCTAssertFalse(noGuide.enforcing, "no guide ⇒ forceEnforce is a no-op")
    }

    /// A full object walk through the phase machine: the `{` opener starts
    /// enforcement (.jsonStart once), then each subsequent byte of {"n":7 is
    /// .jsonBody — the structured response is the whole enforced span.
    func testObjectOpenerWalk() throws {
        var d = GuidedDecoder(guide: try objectGuide(), vocab: 257, idleBudget: 8)
        XCTAssertEqual(d.commit(lbrace), .jsonStart)
        XCTAssertTrue(d.enforcing)
        for b in [quote, nByte, quote, colon, digit(7)] {
            XCTAssertEqual(d.commit(b), .jsonBody, "byte \(b) is enforced body")
        }
    }
}

/// CommitResult is payload-free, so Equatable synthesizes — lets the
/// assertions above compare directly. (@testable-visible internal type.)
extension CommitResult: @retroactive Equatable {}
