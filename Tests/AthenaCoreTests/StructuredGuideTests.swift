import XCTest

@testable import AthenaStructured

/// Model-free (no MLX/SSD): mirrors the Rust `cargo test` vocab walk through
/// the safe Swift wrapper, exercising the full FFI lifecycle (vocab → index →
/// guide → mask/advance/is_final/free).
///
/// M70.2 (audit NG1): these walks used to compile a raw `regex:` index and
/// assert a 32-entry rollback ring, but since M53 the engine is llguidance
/// (grammar/JSON-schema-only) — `StructuredIndex(regex:)` always throws
/// (`oc_index_from_regex` is an ABI-symmetry stub), and the rollback ring is
/// GONE (Athena only ever advances the guide on COMMITTED tokens, so
/// `allowedRollback` is a no-op that always returns 0 and `rollback(n)`
/// succeeds only for n==0). The walks are rebuilt here onto REAL JSON-schema
/// compilation against a 256-byte vocab (token id == byte value), mirroring
/// the shim's own `integer_schema_masks_and_advances` /
/// `object_schema_accepts_valid_doc_and_rejects_invalid` cargo tests so the
/// mask/advance/isFinal contract is pinned against the production engine.
final class StructuredGuideTests: XCTestCase {

    // ASCII byte ids (token id == byte value in this vocab).
    private let lbrace: UInt32 = 0x7B  // {
    private let rbrace: UInt32 = 0x7D  // }
    private let quote: UInt32 = 0x22  // "
    private let colon: UInt32 = 0x3A  // :
    private let comma: UInt32 = 0x2C  // ,
    private let nByte: UInt32 = 0x6E  // n
    private let aByte: UInt32 = 0x61  // a
    private let eos: UInt32 = 256
    private func digit(_ d: Int) -> UInt32 { UInt32(0x30 + d) }

    /// 256 single-byte tokens (id == byte) + eos at id 256 — the Swift mirror
    /// of the shim's `byte_vocab()`.
    private func byteVocab() throws -> StructuredVocabulary {
        let tokens = (0..<256).map {
            VocabToken(id: UInt32($0), bytes: [UInt8($0)])
        }
        return try StructuredVocabulary(tokens: tokens, eosTokenId: 256)
    }

    /// True if token `id`'s bit is set in a freshly filled allowed-mask.
    private func maskAllows(_ guide: StructuredGuide, _ id: UInt32) -> Bool {
        var mask = [UInt8]()
        guard guide.allowedMask(into: &mask) else { return false }
        let byte = Int(id) >> 3
        guard byte < mask.count else { return false }
        return (mask[byte] & (1 << (Int(id) & 7))) != 0
    }

    // MARK: - integer schema: mask + advance + final

    func testIntegerSchemaMaskAndAdvance() throws {
        let guide = try StructuredGuide(
            index: StructuredIndex(
                jsonSchema: #"{"type":"integer"}"#, vocabulary: byteVocab()))
        XCTAssertFalse(guide.isFinal, "no digits yet: not accepting")

        // Opening mask allows a digit, never a letter.
        XCTAssertTrue(maskAllows(guide, digit(4)), "digit allowed at start")
        XCTAssertFalse(maskAllows(guide, aByte), "letter not allowed")

        XCTAssertTrue(guide.advance(digit(4)), "'4' accepted")
        XCTAssertTrue(guide.advance(digit(2)), "'2' accepted")
        XCTAssertTrue(guide.isFinal, "\"42\" is a complete integer")
        // A letter has no transition and must NOT mutate state.
        XCTAssertFalse(guide.advance(aByte), "letter rejected, no transition")
        XCTAssertTrue(guide.isFinal, "state unchanged after rejected advance")
    }

    func testInvalidTokenNotAdvanced() throws {
        let guide = try StructuredGuide(
            index: StructuredIndex(
                jsonSchema: #"{"type":"integer"}"#, vocabulary: byteVocab()))
        // A letter (in-range but disallowed) and eos (no value yet) both fail.
        XCTAssertFalse(guide.advance(aByte), "letter has no start transition")
        XCTAssertFalse(guide.advance(eos), "eos disallowed before any digit")
        XCTAssertFalse(guide.isFinal, "rejected advances left state at start")
        // The valid token still works afterward (process/parser is alive).
        XCTAssertTrue(guide.advance(digit(3)))
        XCTAssertTrue(guide.isFinal)
    }

    // MARK: - object schema: bounded walk + additionalProperties:false

    func testObjectSchemaWalkAndRejectsExtra() throws {
        let schema = """
            {"type":"object","properties":{"n":{"type":"integer"}},\
            "required":["n"],"additionalProperties":false}
            """
        let guide = try StructuredGuide(
            index: StructuredIndex(jsonSchema: schema, vocabulary: byteVocab()))
        // Every byte of {"n":7} must be permitted by the mask in turn, and the
        // parser ends accepting with eos then allowed.
        let doc: [UInt32] = [lbrace, quote, nByte, quote, colon, digit(7), rbrace]
        for b in doc {
            XCTAssertTrue(maskAllows(guide, b), "byte \(b) permitted in sequence")
            XCTAssertTrue(guide.advance(b), "advance \(b)")
        }
        XCTAssertTrue(guide.isFinal, "complete object is accepting")
        XCTAssertTrue(maskAllows(guide, eos), "eos allowed once object complete")

        // additionalProperties:false — after {"n":7 the only continuations are
        // more digits or }, never a comma.
        let g2 = try StructuredGuide(
            index: StructuredIndex(jsonSchema: schema, vocabulary: byteVocab()))
        for b in [lbrace, quote, nByte, quote, colon, digit(7)] {
            XCTAssertTrue(g2.advance(b), "prefix advance \(b)")
        }
        XCTAssertFalse(g2.advance(comma), "additionalProperties:false rejects ,")
    }

    // MARK: - rollback is a retired no-op (llguidance, M53)

    func testRollbackIsNoOp() throws {
        let guide = try StructuredGuide(
            index: StructuredIndex(
                jsonSchema: #"{"type":"integer"}"#, vocabulary: byteVocab()))
        XCTAssertEqual(guide.allowedRollback, 0)
        XCTAssertTrue(guide.advance(digit(4)))
        XCTAssertTrue(guide.advance(digit(2)))
        // Monotonic guide: the ring is gone, so allowedRollback stays 0…
        XCTAssertEqual(
            guide.allowedRollback, 0, "rollback ring is a no-op under llguidance")
        // …and only the no-op rollback(0) "succeeds".
        XCTAssertTrue(guide.rollback(0))
        XCTAssertFalse(guide.rollback(1))
        XCTAssertFalse(guide.rollback(2))
    }

    func testJSONSchemaCompilesToGuide() throws {
        let guide = try StructuredGuide(
            index: StructuredIndex(
                jsonSchema: #"{"type":"integer"}"#,
                vocabulary: byteVocab()))
        XCTAssertGreaterThan(guide.maskLength, 0)
    }

    // MARK: - M65.1 — hostile response_format degrades, never crashes
    //
    // A `/v1` caller controls the `response_format.json_schema` string that
    // reaches the Rust shim (MLXLLMModule builds `StructuredIndex(jsonSchema:)`
    // from it). After the G1/G3 hardening, a hostile schema must come back as
    // a thrown `StructuredError` — which the server boundary turns into the
    // standard `{"error":{message,type,code}}` envelope — rather than panic
    // across the C ABI and abort the daemon. We pin the throw at the
    // structured-compile layer the handler delegates to, and prove the process
    // survives by compiling a valid schema afterwards.

    func testHostileSchemaThrowsInsteadOfCrashing() throws {
        let vocab = try byteVocab()

        // Repetition bound past the shim cap (pathological grammar).
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
        // hostile ones (a panic-across-FFI would have aborted the process).
        let guide = try StructuredGuide(
            index: StructuredIndex(
                jsonSchema: #"{"type":"integer"}"#, vocabulary: vocab))
        XCTAssertGreaterThan(guide.maskLength, 0)
    }

    // MARK: - the raw-regex constructor is a retired ABI stub
    //
    // llguidance has no raw-regex compile path; `StructuredIndex(regex:)` is
    // retained only for ABI symmetry and must throw, never silently produce an
    // unconstrained guide. Pin that so a future engine swap can't quietly
    // re-enable a half-working regex path.

    func testRegexConstructorThrows() throws {
        let vocab = try byteVocab()
        XCTAssertThrowsError(
            try StructuredIndex(regex: "[0-9][0-9]", vocabulary: vocab)
        ) { err in
            XCTAssertTrue(err is StructuredError)
        }
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

    func testAdvanceOpenerTolerantFallsBackViaAlias() throws {
        // A synthetic alias ('a' → '4') exercises the realignment branch
        // deterministically: 'a' is in-range but disallowed for an integer (no
        // native transition), and advanceOpenerTolerant retries with the
        // aliased '4', which IS allowed. (Under llguidance a real
        // space-prefixed opener advances natively — leading whitespace is
        // in-grammar — so this fallback is the defensive inert path the
        // IDLE→ENFORCING probe keeps; here we drive it directly.)
        let guide = try StructuredGuide(
            index: StructuredIndex(
                jsonSchema: #"{"type":"integer"}"#, vocabulary: byteVocab()))
        XCTAssertFalse(guide.advance(aByte), "'a' has no native transition")
        guide.openerAlias = [aByte: digit(4)]
        XCTAssertTrue(
            guide.advanceOpenerTolerant(aByte), "realigned to the bare '4'")
        XCTAssertTrue(guide.isFinal, "\"4\" is a complete integer")
    }

    func testAdvanceOpenerTolerantNoAliasStillStrict() throws {
        let guide = try StructuredGuide(
            index: StructuredIndex(
                jsonSchema: #"{"type":"integer"}"#, vocabulary: byteVocab()))
        // No openerAlias set ⇒ behaves exactly like `advance`.
        XCTAssertFalse(guide.advanceOpenerTolerant(aByte))
        XCTAssertTrue(guide.advanceOpenerTolerant(digit(4)))
        XCTAssertTrue(guide.isFinal)
    }

    // MARK: - shared-index, independent-walker contract
    //
    // The structured-output hot path reuses one compiled `StructuredIndex`
    // across many requests. Correctness depends on the contract that each
    // `StructuredGuide(index:)` instance has its OWN walker state — advancing
    // one walker MUST NOT affect another walker built from the same index.

    func testSharedIndexProducesIndependentWalkers() throws {
        let index = try StructuredIndex(
            jsonSchema: #"{"type":"integer"}"#, vocabulary: byteVocab())
        let a = try StructuredGuide(index: index)
        let b = try StructuredGuide(index: index)

        // Drive guide A to a complete value; guide B must stay at the start.
        XCTAssertTrue(a.advance(digit(4)))
        XCTAssertTrue(a.advance(digit(2)))
        XCTAssertTrue(a.isFinal)
        XCTAssertFalse(
            b.isFinal,
            "advancing one walker must NOT mutate a sibling built from the "
                + "same shared index")

        // B can independently walk its own path.
        XCTAssertTrue(b.advance(digit(5)))
        XCTAssertTrue(b.advance(digit(9)))
        XCTAssertTrue(b.isFinal)
    }

    func testSharedIndexSurvivesWalkerDeinit() throws {
        // Build the index, then build and drop multiple walkers off it. The
        // grammar must remain valid for the next walker — walker deinit must
        // NOT invalidate the shared compiled index.
        let index = try StructuredIndex(
            jsonSchema: #"{"type":"integer"}"#, vocabulary: byteVocab())
        for _ in 0..<8 {
            let g = try StructuredGuide(index: index)
            XCTAssertTrue(g.advance(digit(5)))
            XCTAssertTrue(g.isFinal)
            // g drops at end of scope
        }
        // After many walker deinits, a fresh walker on the same index must
        // still work — this is the shared-index hot path.
        let g = try StructuredGuide(index: index)
        XCTAssertTrue(g.advance(digit(7)))
        XCTAssertTrue(g.isFinal)
    }

    // Compile-time check: the shared-index hot path requires `StructuredIndex`
    // to cross actor boundaries (the cached factory is captured into a
    // `container.perform` Sendable closure). If this stops compiling, someone
    // removed the @unchecked Sendable annotation and the cache will break.
    func testStructuredIndexIsSendableForCrossActorCapture() async throws {
        let index = try StructuredIndex(
            jsonSchema: #"{"type":"integer"}"#, vocabulary: byteVocab())
        let isSendable: @Sendable () -> StructuredIndex = { index }
        let detached = Task { isSendable() }
        let recovered = await detached.value
        let g = try StructuredGuide(index: recovered)
        XCTAssertTrue(g.advance(digit(3)))
        XCTAssertTrue(g.isFinal)
    }
}
