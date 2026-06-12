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

    // MARK: - M65.1 — hostile response_format degrades, never crashes
    //
    // A `/v1` caller controls the `response_format.json_schema` string that
    // reaches the Rust shim (MLXLLMModule builds `StructuredIndex(jsonSchema:)`
    // from it). After the G1/G3 hardening, a hostile schema must come back as
    // a thrown `StructuredError` — which the server boundary turns into the
    // standard `{"error":{message,type,code}}` envelope — rather than panic
    // across the C ABI and abort the daemon. We can't drive the full HTTP
    // handler until the Server target is testable (M70), so this pins the
    // throw at the structured-compile layer the handler delegates to, and
    // proves the process survives by compiling a valid schema afterwards.

    func testHostileSchemaThrowsInsteadOfCrashing() throws {
        let vocab = try digitVocab()

        // Repetition bound past the shim cap (would be a pathological
        // grammar; rejected before compile).
        let hugeRepetition =
            #"{"type":"array","items":{"type":"integer"},"maxItems":100001}"#
        XCTAssertThrowsError(
            try StructuredIndex(jsonSchema: hugeRepetition, vocabulary: vocab)
        ) { err in
            XCTAssertTrue(
                err is StructuredError,
                "hostile schema surfaces a typed StructuredError, not a crash")
        }

        // Oversized raw schema string (past the 1 MiB shim cap).
        let oversized =
            "{\"type\":\"string\",\"description\":\""
            + String(repeating: "x", count: 1_100_000) + "\"}"
        XCTAssertThrowsError(
            try StructuredIndex(jsonSchema: oversized, vocabulary: vocab))

        // Malformed JSON is likewise a clean throw.
        XCTAssertThrowsError(
            try StructuredIndex(jsonSchema: #"{"type":"#, vocabulary: vocab))

        // The daemon path is still alive: a normal schema compiles after the
        // hostile ones (a panic-across-FFI would have aborted the process and
        // we'd never reach here).
        let guide = try StructuredGuide(
            index: StructuredIndex(
                jsonSchema: #"{"type":"integer"}"#, vocabulary: vocab))
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

    // MARK: - M49.1 — shared-index, independent-walker contract
    //
    // The M49.1 cache reuses one compiled `StructuredIndex` across many
    // requests. Correctness depends on the contract that each
    // `StructuredGuide(index:)` instance has its OWN walker state —
    // advance/rollback on one walker MUST NOT affect another walker
    // built from the same index. These tests pin that contract.

    func testSharedIndexProducesIndependentWalkers() throws {
        let index = try StructuredIndex(
            regex: "[0-9][0-9]", vocabulary: digitVocab())
        let a = try StructuredGuide(index: index)
        let b = try StructuredGuide(index: index)

        // Drive guide A forward; guide B must stay at the start.
        XCTAssertTrue(a.advance(3))
        XCTAssertEqual(a.allowedRollback, 1)
        XCTAssertEqual(
            b.allowedRollback, 0,
            "advancing one walker must NOT mutate the shared index in a "
                + "way that affects sibling walkers built from it")

        // Drive A to the final state — B is still at start.
        XCTAssertTrue(a.advance(7))
        XCTAssertTrue(a.isFinal)
        XCTAssertFalse(b.isFinal)

        // B can independently walk its own path.
        XCTAssertTrue(b.advance(5))
        XCTAssertTrue(b.advance(9))
        XCTAssertTrue(b.isFinal)
        XCTAssertEqual(b.allowedRollback, 2)
    }

    func testSharedIndexSurvivesWalkerDeinit() throws {
        // Build the index, then build and drop multiple walkers off
        // it. The DFA must remain valid for the next walker — i.e.
        // walker deinit must NOT invalidate the shared compiled index.
        let index = try StructuredIndex(
            regex: "[0-9]", vocabulary: digitVocab())
        for _ in 0..<8 {
            let g = try StructuredGuide(index: index)
            XCTAssertTrue(g.advance(5))
            XCTAssertTrue(g.isFinal)
            // g drops at end of scope
        }
        // After many walker deinits, a fresh walker on the same index
        // must still work — this is the M49.1 hot path.
        let g = try StructuredGuide(index: index)
        XCTAssertTrue(g.advance(0))
        XCTAssertTrue(g.isFinal)
    }

    // Compile-time check: M49.1 requires `StructuredIndex` to cross
    // actor boundaries (the cached DFA is captured into a
    // `container.perform` Sendable closure). If this stops compiling,
    // someone removed the @unchecked Sendable annotation and the
    // cache will stop working.
    func testStructuredIndexIsSendableForCrossActorCapture() async throws {
        let index = try StructuredIndex(
            regex: "[0-9]", vocabulary: digitVocab())
        let isSendable: @Sendable () -> StructuredIndex = { index }
        // If StructuredIndex weren't Sendable, the closure capture
        // above would fail at compile time. Smoke-test the captured
        // value still works after crossing into a Task.
        let detached = Task {
            isSendable()
        }
        let recovered = await detached.value
        let g = try StructuredGuide(index: recovered)
        XCTAssertTrue(g.advance(3))
        XCTAssertTrue(g.isFinal)
    }
}
