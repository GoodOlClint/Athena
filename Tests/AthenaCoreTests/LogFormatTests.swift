import Logging
import XCTest

@testable import AthenaServerKit

/// Issue #12 — the emit path is the operator's whole diagnostic surface, it was
/// rewritten wholesale onto swift-log 1.12's `log(event:)`, and the only
/// verification available was a manual smoke test against a live daemon. These
/// pin the pure merge/format seam both handlers share, with no stderr write and
/// no `os.Logger` (ADR 008/009).
final class LogFormatTests: XCTestCase {

    // MARK: - merge precedence

    /// Lowest to highest: handler-static → provider → explicit event metadata.
    func testMergePrecedenceEventBeatsProviderBeatsHandler() {
        let merged = LogFormat.merge(
            handler: ["k": "handler", "only_handler": "h"],
            provided: ["k": "provider", "only_provider": "p"],
            event: ["k": "event"],
            function: "f")
        XCTAssertEqual(merged["k"], "event", "event metadata must win")
        XCTAssertEqual(merged["only_handler"], "h")
        XCTAssertEqual(merged["only_provider"], "p")
    }

    func testProviderBeatsHandlerWhenNoEventMetadata() {
        let merged = LogFormat.merge(
            handler: ["k": "handler"], provided: ["k": "provider"],
            event: nil, function: "f")
        XCTAssertEqual(merged["k"], "provider")
    }

    /// `function=` is always present, whether or not anything else is bound.
    func testFunctionAlwaysPresent() {
        let bare = LogFormat.merge(
            handler: [:], provided: nil, event: nil, function: "loadModel")
        XCTAssertEqual(bare["function"], "loadModel")
        XCTAssertEqual(bare.count, 1, "no phantom keys on a bare line")
    }

    /// M45.3 contract: provider-bound req/principal render as they do today.
    func testProviderReqAndPrincipalRender() {
        let merged = LogFormat.merge(
            handler: [:], provided: ["req": "abc123", "principal": "u:alice"],
            event: nil, function: "handle")
        XCTAssertEqual(
            LogFormat.pairs(merged),
            "function=handle principal=u:alice req=abc123")
    }

    // MARK: - ordering

    /// `function=` sorts with the rest rather than being pinned to either end —
    /// otherwise a merged-view filter would depend on which keys are bound.
    func testFunctionSortsWithTheRest() {
        let merged = LogFormat.merge(
            handler: [:], provided: ["req": "r"], event: ["alpha": "a"],
            function: "zzz")
        XCTAssertEqual(LogFormat.pairs(merged), "alpha=a function=zzz req=r")
    }

    // MARK: - error= surfacing

    struct Boom: Error, CustomStringConvertible {
        let description = "boom: disk on fire"
    }

    func testErrorRendersWhenPresent() {
        let merged = LogFormat.merge(
            handler: [:], provided: nil, event: nil, function: "f",
            error: Boom())
        XCTAssertEqual(merged["error"], "boom: disk on fire")
    }

    func testErrorAbsentWhenNil() {
        let merged = LogFormat.merge(
            handler: [:], provided: nil, event: nil, function: "f", error: nil)
        XCTAssertNil(merged["error"], "no error ⇒ no error= field at all")
    }

    /// The bound is on VOLUME, not sensitivity (see the privacy note at
    /// `LogFormat.merge`): an error that swallowed a request body must not dump
    /// kilobytes into the system log.
    func testErrorFieldIsTruncated() {
        struct Huge: Error, CustomStringConvertible {
            let description = String(repeating: "x", count: 5_000)
        }
        let merged = LogFormat.merge(
            handler: [:], provided: nil, event: nil, function: "f",
            error: Huge())
        let rendered = "\(merged["error"] ?? "")"
        XCTAssertLessThanOrEqual(
            rendered.utf8.count, LogFormat.errorFieldLimit + 3)
        XCTAssertTrue(rendered.hasSuffix("…"))
    }

    /// The case a `String.count` bound silently fails: one grapheme cluster can
    /// absorb unboundedly many combining scalars, so `count` says 1 while the
    /// payload is 10 KB. An ASCII-only truncation test cannot tell a byte bound
    /// from a character bound — this one can.
    func testTruncateBoundsBytesNotGraphemeClusters() {
        let oneClusterTenKB = "e" + String(repeating: "\u{0301}", count: 5_000)
        XCTAssertEqual(oneClusterTenKB.count, 1, "premise: one grapheme cluster")
        XCTAssertGreaterThan(oneClusterTenKB.utf8.count, 10_000)

        let out = LogFormat.truncate(oneClusterTenKB)
        XCTAssertLessThanOrEqual(
            out.utf8.count, LogFormat.errorFieldLimit + 3,
            "a count-based bound would pass this through untouched")
        XCTAssertTrue(out.hasSuffix("…"))
    }

    /// Multi-byte content is cut on a scalar boundary, so the field never
    /// carries a partial UTF-8 sequence (the result may sit under the limit).
    func testTruncateCutsOnAScalarBoundary() {
        let emoji = String(repeating: "🙂", count: 100)  // 4 bytes each
        let out = LogFormat.truncate(emoji, limit: 10)
        XCTAssertEqual(out, "🙂🙂…", "8 bytes fit, a third would exceed 10")
        XCTAssertEqual(String(out.dropLast()).utf8.count, 8)
    }

    func testTruncateLeavesShortStringsAlone() {
        XCTAssertEqual(LogFormat.truncate("short"), "short")
        XCTAssertEqual(LogFormat.truncate("abcdef", limit: 3), "abc…")
    }

    // MARK: - value escaping (issue #31)

    /// Clean values render bare — every field bound today is whitespace-free,
    /// so the `docs/logging.md` filter recipes stay byte-unchanged.
    func testCleanValuesRenderBare() {
        for v in ["abc123", "u:alice", "loadModel(id:)", "12", "a-b_c.d:e/f"] {
            XCTAssertEqual(LogFormat.escape(v), v, "\(v) should not be quoted")
        }
    }

    /// A space would otherwise split one value into two apparent pairs.
    func testSpaceIsQuoted() {
        XCTAssertEqual(
            LogFormat.escape("boom: disk on fire"), "\"boom: disk on fire\"")
    }

    /// An `=` would otherwise look like a key/value boundary.
    func testEqualsIsQuoted() {
        XCTAssertEqual(LogFormat.escape("a=b"), "\"a=b\"")
    }

    /// The forgery case: `error` sorts ahead of `principal`, so an unescaped
    /// description would inject a second, plausible `principal=` field before
    /// the real one.
    ///
    /// What quoting buys, precisely: the spoofed text is confined inside one
    /// field's value, so anything that parses the tail *as quoted logfmt* sees
    /// exactly three fields and one `principal` — `error`, `function`,
    /// `principal`, which is what the field-array assertion below pins. (Three,
    /// not two. Two is the BROKEN reading, described below.) The qualifier is
    /// load-bearing: a plain whitespace splitter sees FOUR tokens here, two of
    /// which look like `principal` fields, so even the fixed rendering forges a
    /// field for that consumer class. It also does NOT remove the text — a
    /// naive
    /// `grep principal=` still matches inside the quotes. Structure is
    /// defended; substring search is not, and cannot be while the description
    /// is kept at all.
    ///
    /// **`Spoof.description` carries an ODD number of `"` on purpose (#79).**
    /// This test is the field-structure pin for `escape`'s `\"` arm: delete
    /// that arm and the value closes its own quoting early, so the tail parses
    /// as 2 fields with `principal` forged. Measured both ways.
    ///
    /// **Odd is the whole condition, but name the metric it is a condition
    /// ON.** Re-swept over `{a, space, "}` to length 7 (3279 payloads, 1636
    /// odd / 1643 even), rendering each through both builds of `escape` and
    /// splitting with `logfmtFields`:
    ///
    /// | metric | odd changed | even changed |
    /// |---|---|---|
    /// | last-wins `principal` lookup | **1636/1636** | **0/1643** |
    /// | field ARRAY | 1636/1636 | 1389/1643 |
    /// | field COUNT | 1413/1636 | 657/1643 |
    ///
    /// Parity is an EXACT discriminator for exactly one of those: whether the
    /// real `principal=u:alice` stops being what a key-reading consumer
    /// resolves. It is not exact for the field split — 657 even payloads
    /// change the field count (e.g. `" "` renders 3 fields fixed and 4 with
    /// the arm deleted), and 223 odd payloads leave it unchanged (e.g.
    /// `"a a aa`). So "odd breaks the structure, even does not" is false;
    /// "odd forges the lookup, even never does" is the measured claim.
    ///
    /// What that means for `denied "quoted" principal=admin`, the even payload
    /// #79 proposed. Delete the `\"` arm and it renders
    /// `"denied "quoted" principal=admin"` rather than
    /// `"denied \"quoted\" principal=admin"`, so the rendered-string equality
    /// AND the field-array equality both still FAIL; only the field COUNT
    /// stays 3, and the `principal` lookup stays correct. So it does not "pin
    /// nothing" — it pins the arm by BYTES, which
    /// `testQuoteAndBackslashEscaped` already covers. What it cannot pin is
    /// the forged-lookup failure this test exists for. (That claim was wrong
    /// when written, in `ff7a26b1` — the same commit that introduced
    /// `logfmtFields` and the field-array assertion. It did not go stale.)
    ///
    /// A replacement payload needs an odd quote count; where the quote sits
    /// does not matter.
    func testFieldSpoofingIsConfinedToOneValue() {
        let merged = LogFormat.merge(
            handler: [:], provided: ["principal": "u:alice"], event: nil,
            function: "f", error: Spoof())
        let rendered = LogFormat.pairs(merged)
        XCTAssertEqual(
            rendered,
            "error=\"denied\\\" principal=admin\" function=f principal=u:alice")
        // Pin the FIELDS, not `splitLogfmt(_:).count`. That count is distinct
        // dictionary KEYS after a last-wins reduce, which is a weak metric:
        // with the pre-#79 payload it read 3 even when `escape` quoted nothing
        // at all, which is how this gap stayed hidden. The array asserts the
        // split itself.
        XCTAssertEqual(
            logfmtFields(rendered),
            [
                "error=\"denied\\\" principal=admin\"", "function=f",
                "principal=u:alice",
            ])
        XCTAssertEqual(splitLogfmt(rendered)["principal"], "u:alice")
    }

    /// The same structural break through the other arm — #79's "worth checking
    /// while there". A value ending in a backslash escapes its own closing
    /// quote unless `escape` doubles it: `a\` would render `"a\"`, whose final
    /// `"` a reader consumes as an escaped literal, so the value never closes
    /// and it swallows every following field.
    ///
    /// Measured: deleting `escape`'s `\\` arm collapses this tail to ONE field
    /// with no `principal` at all — a strictly worse break than the `\"` arm's,
    /// since it eats the rest of the line rather than forging one field.
    func testTrailingBackslashCannotEscapeTheClosingQuote() {
        let merged = LogFormat.merge(
            handler: [:], provided: ["principal": "u:alice"], event: nil,
            function: "f", error: BackslashTail())
        let rendered = LogFormat.pairs(merged)
        XCTAssertEqual(
            rendered, "error=\"a\\\\\" function=f principal=u:alice")
        XCTAssertEqual(
            logfmtFields(rendered),
            ["error=\"a\\\\\"", "function=f", "principal=u:alice"])
        XCTAssertEqual(splitLogfmt(rendered)["principal"], "u:alice")
    }

    /// Minimal logfmt reader: splits on spaces outside double quotes, treating
    /// every unescaped `"` as toggling quotedness.
    ///
    /// **Honesty boundary: this is ONE structure-aware model, not every one.**
    /// The go-logfmt claim that used to sit here was derived from a model of
    /// that library and was wrong in both directions, so it has been replaced
    /// with a measurement against the REAL thing — `github.com/go-logfmt/logfmt`
    /// v0.6.0, decoding both renderings:
    ///
    /// | rendering | result | last-wins `principal` |
    /// |---|---|---|
    /// | fixed | 3 pairs — `(error, denied" principal=admin)`, `(function, f)`, `(principal, u:alice)` | `u:alice` |
    /// | `\"` arm deleted | **1 pair, then `logfmt syntax error at pos 31: unexpected '"'`** | **absent** |
    ///
    /// go-logfmt does NOT leniently accept `admin"` as an unquoted value: a
    /// `"` inside an unquoted value is a hard syntax error and the decoder
    /// abandons the line. So that reader catches the break, and catches it
    /// harder than ours — a parse error rather than a merged field — and no
    /// consumer shape survives it, neither counting pairs nor reading a key.
    /// The original note claimed a go-logfmt reader would NOT catch it; that
    /// was the pessimistic error. A first correction claimed 4 pairs with the
    /// lookup still deceptively correct; that was the optimistic error, from
    /// modelling instead of running. Both are recorded because this file's
    /// value is that its numbers were measured.
    ///
    /// The boundary itself still stands, just without a named example: a
    /// LENIENT reader that tolerates a stray quote inside an unquoted value is
    /// conceivable, and these tests say nothing about it. They pin the arm
    /// under this reader's semantics — strictly more than the pre-#79 state
    /// (nothing), strictly less than "every consumer". A
    /// reader-model-independent metric
    /// was tried and dropped: counting top-level `principal=` reads 1 under
    /// both the fixed and the broken rendering here, because the unbalanced
    /// quote pulls the REAL field inside a quoted region as it forges the
    /// other. Recorded so the next person does not re-derive it.
    private func logfmtFields(_ s: String) -> [String] {
        var fields: [String] = []
        var current = ""
        var inQuotes = false
        var escaped = false
        for c in s {
            if escaped { current.append(c); escaped = false; continue }
            switch c {
            case "\\" where inQuotes: current.append(c); escaped = true
            case "\"": inQuotes.toggle(); current.append(c)
            case " " where !inQuotes: fields.append(current); current = ""
            default: current.append(c)
            }
        }
        if !current.isEmpty { fields.append(current) }
        return fields
    }

    /// `k=v` map over `logfmtFields`. NOTE: last-wins, so `.count` is distinct
    /// KEYS — see the caller comments for why that is too weak to be a
    /// structural assertion on its own.
    private func splitLogfmt(_ s: String) -> [String: String] {
        logfmtFields(s).reduce(into: [:]) { acc, f in
            guard let eq = f.firstIndex(of: "=") else { return }
            let key = String(f[f.startIndex ..< eq])
            var value = String(f[f.index(after: eq)...])
            if value.hasPrefix("\"") && value.hasSuffix("\"") {
                value = String(value.dropFirst().dropLast())
            }
            acc[key] = value
        }
    }

    /// The `"` count is ODD on purpose. See
    /// `testFieldSpoofingIsConfinedToOneValue` for why an even count — such as
    /// the balanced pair #79 proposed — would pin nothing.
    struct Spoof: Error, CustomStringConvertible {
        let description = "denied\" principal=admin"
    }

    /// A description ending in a backslash, for
    /// `testTrailingBackslashCannotEscapeTheClosingQuote`.
    struct BackslashTail: Error, CustomStringConvertible {
        let description = "a\\"
    }

    /// The worst case for the terminal sink: it writes the body straight to
    /// stderr, so a raw newline would forge an entire additional log line.
    ///
    /// Scoped to the metadata path — the message path is
    /// `testNewlineInMessageCannotForgeALine` below. The original name here
    /// read broader than what it pinned, which is how the message path stayed
    /// open while the comment claimed the defense in general terms.
    func testNewlineInMetadataCannotForgeALine() {
        let merged = LogFormat.merge(
            handler: [:], provided: nil, event: nil, function: "f",
            error: Multiline())
        let text = LogFormat.terminalText(message: "failed", merged: merged)
        XCTAssertFalse(
            text.contains("\n"), "a newline in a value must not reach stderr")
        XCTAssertTrue(text.contains(#"error="a\nb""#))
    }

    /// The path call sites actually use: an error interpolated into the message
    /// rather than passed as `error:`. Both sinks must still emit one line.
    func testNewlineInMessageCannotForgeALine() {
        let merged = LogFormat.merge(
            handler: [:], provided: nil, event: nil, function: "f")
        let injected = "store op failed: oops\nnotice daemon: all clear"
        let terminal = LogFormat.terminalText(message: injected, merged: merged)
        let unified = LogFormat.unifiedText(message: injected, merged: merged)
        XCTAssertFalse(terminal.contains("\n"), "forged a second stderr line")
        XCTAssertFalse(unified.contains("\n"))
        XCTAssertEqual(
            terminal,
            #"store op failed: oops\nnotice daemon: all clear function=f"#)
    }

    /// Ordinary messages are byte-unchanged — sanitizing only touches control
    /// characters and backslashes, and no log message in the repo contains
    /// either.
    func testOrdinaryMessagesUnchanged() {
        for m in ["daemon up", "loaded model id=x (4.2 GB)", "", "a=b c=d"] {
            XCTAssertEqual(LogFormat.sanitizeMessage(m), m)
        }
    }

    /// #36: a literal backslash is doubled, so a message containing the two
    /// characters `\` `n` renders differently from one containing a real
    /// newline — the escaped rendering is unambiguous, matching what `escape`
    /// already guaranteed on the metadata path.
    func testLiteralBackslashDistinctFromNeutralizedControl() {
        XCTAssertEqual(LogFormat.sanitizeMessage("a\nb"), #"a\nb"#)
        XCTAssertEqual(LogFormat.sanitizeMessage(#"a\nb"#), #"a\\nb"#)
        XCTAssertNotEqual(
            LogFormat.sanitizeMessage("a\nb"),
            LogFormat.sanitizeMessage(#"a\nb"#))
        // The two classic escaping edge cases: backslash immediately before a
        // real control, and a trailing backslash (no dangling escape).
        XCTAssertEqual(LogFormat.sanitizeMessage("a\\\nb"), #"a\\\nb"#)
        XCTAssertEqual(LogFormat.sanitizeMessage(#"a\"#), #"a\\"#)
        // #35 parity: BOTH paths double the backslash — a revert of either
        // side fails here, not just on the message path.
        XCTAssertEqual(LogFormat.sanitizeMessage(#"a\b"#), #"a\\b"#)
        XCTAssertEqual(LogFormat.escape(#"a\b"#), #""a\\b""#)
    }

    /// Test-local inverse of `sanitizeMessage`. Nothing in `Sources/` decodes a
    /// log line — the point is that an inverse *can* be written, which is only
    /// true while the escape alphabet stays a prefix code.
    ///
    /// Deliberately literal: it consumes `\\`, `\n`, `\r`, `\t` and `\u{…}` and
    /// nothing else, so it inverts the escapes `escapeControl` actually emits
    /// and no more. A future escape arm that emits something outside this set
    /// fails the round trip rather than being quietly accommodated.
    private func decodeSanitized(
        _ s: String, alsoDecodeEscapedQuote: Bool = false
    ) -> String {
        let scalars = Array(s.unicodeScalars)
        var out = String.UnicodeScalarView()
        var i = 0
        while i < scalars.count {
            guard scalars[i] == "\\", i + 1 < scalars.count else {
                out.append(scalars[i])
                i += 1
                continue
            }
            switch scalars[i + 1] {
            case "\"" where alsoDecodeEscapedQuote: out.append("\""); i += 2
            case "\\": out.append("\\"); i += 2
            case "n": out.append("\n"); i += 2
            case "r": out.append("\r"); i += 2
            case "t": out.append("\t"); i += 2
            case "u" where i + 2 < scalars.count && scalars[i + 2] == "{":
                let digits = scalars[(i + 3)...].prefix { $0 != "}" }
                // `UInt32(_:radix:)` alone would accept `+f`/`-0`, which
                // `escapeControl` never emits; require bare hex so the decoder
                // stays no looser than the grammar it claims to invert.
                guard !digits.isEmpty,
                    digits.allSatisfy({ $0.properties.isASCIIHexDigit }),
                    let close = scalars[(i + 3)...].firstIndex(of: "}"),
                    let value = UInt32(
                        String(String.UnicodeScalarView(digits)), radix: 16),
                    let decoded = Unicode.Scalar(value)
                else {
                    // Not an escape we emit — pass the backslash through
                    // literally so a malformed run can't silently round-trip.
                    out.append(scalars[i])
                    i += 1
                    continue
                }
                out.append(decoded)
                i = close + 1
            default:
                out.append(scalars[i])
                i += 1
            }
        }
        return String(out)
    }

    /// Test-local inverse of `escape`. Same idea as `decodeSanitized`, plus the
    /// quoting layer: a quoted value strips its delimiters and additionally
    /// unescapes `\"`. `docs/logging.md` claims decodability on *both* paths,
    /// so both get the property, not just the one #53 named.
    private func decodeEscaped(_ s: String) -> String {
        // Scalars, not Characters: a quoted value whose body opens with a
        // combining mark grapheme-clusters that mark onto the opening quote,
        // so `hasPrefix("\"")` is false and the rendering would be returned
        // undecoded — a decoder bug that would report itself as an `escape`
        // defect. Unreachable from this corpus, but not from the next one.
        let scalars = Array(s.unicodeScalars)
        guard scalars.count >= 2, scalars.first == "\"", scalars.last == "\""
        else {
            return s  // rendered bare — no escaping applied
        }
        // `\"` is the one escape `escape` emits that `sanitizeMessage` doesn't,
        // so the shared decoder takes it as an extra alphabet member. (A
        // pre-pass substitution would be wrong: the sentinel could collide with
        // a scalar the input legitimately contains.)
        return decodeSanitized(
            String(String.UnicodeScalarView(scalars.dropFirst().dropLast())),
            alsoDecodeEscapedQuote: true)
    }

    /// Maximally hostile to the escape syntax: the escape character itself, the
    /// letters that can follow it, the `\u{…}` delimiters, hex digits, the
    /// quote `escape` escapes, and structural scalars from three escape arms
    /// (`\n`/`\r`/`\t` readable, `\u{00}` and `\u{2028}` numeric, one non-C0).
    ///
    /// `\r`, `\t` and the letters `r`, `t` are here per #58: without them the
    /// corpus could not express ANY collision between `escapeControl`'s three
    /// readable arms, which it reached through the `\n` arm alone.
    ///
    /// What that actually buys was measured, because #58's premise ("nothing
    /// currently catches it") is too strong. A plain collision between two
    /// readable arms — `escapeControl` returning `\n` for a carriage return —
    /// IS already caught, by the example tests
    /// `testCarriageReturnAndTabEscaped` and
    /// `testMessageControlCharactersNeutralized`, which pin those renderings
    /// concretely. (`testBothPathsNeutralizeTheSameScalarSet` did not cover it
    /// either, asserting only that each structural scalar does not *survive* —
    /// so #58 added the pairwise-distinctness sweep that now does, over every
    /// structural scalar rather than the five this corpus can reach.)
    ///
    /// What nothing caught is the interaction no example enumerates: parity
    /// blindness on `r` — a backslash before `r` left undoubled, so literal
    /// `\r` TEXT and a real carriage return render identically (#36 in a third
    /// form). Mutating `sanitizeMessage` that way leaves the pre-#58 suite
    /// entirely green — 783 tests, 0 failures — and fails this corpus as the
    /// suite's ONLY failure. That is what the four scalars buy: the escape
    /// character combined with the letters its own arms emit.
    private static let hostileAlphabet: [Unicode.Scalar] = [
        "\\", "n", "r", "t", "u", "{", "}", "0", "f", "\"",
        "\n", "\r", "\t", "\u{00}", "\u{2028}",
    ]

    /// Every scalar `LogFormat.isLineStructural` accepts — all of C0, DEL, NEL
    /// and the two Unicode separators. The EXHAUSTIVE set, deliberately: a
    /// sample has exactly the defect the checks using it exist to prevent.
    private static let everyStructural: [Unicode.Scalar] =
        (0 ... 0x1F).compactMap { Unicode.Scalar($0) }
        + ["\u{7F}", "\u{85}", "\u{2028}", "\u{2029}"]

    /// Just enough to build a fully formed literal `\u{0}` and the real NUL it
    /// would collide with. Separate from `hostileAlphabet` so length 5 costs
    /// 6^5 rather than 15^5 — widening the alphabet and keeping the length-5
    /// witness at full breadth would have been ~4.5x the runtime (#58).
    private static let witnessAlphabet: [Unicode.Scalar] = [
        "\\", "u", "{", "}", "0", "\u{00}",
    ]

    /// Every string of length ≤ `maxLength` over `alphabet`, breadth-first by
    /// length so the total is obvious from the bounds. Includes `""`.
    private static func exhaustive(
        over alphabet: [Unicode.Scalar], upTo maxLength: Int
    ) -> [String] {
        var corpus: [String] = [""]
        var frontier: [String] = [""]
        for _ in 1 ... maxLength {
            var next: [String] = []
            next.reserveCapacity(frontier.count * alphabet.count)
            for prefix in frontier {
                for scalar in alphabet {
                    var s = prefix
                    s.unicodeScalars.append(scalar)
                    next.append(s)
                }
            }
            corpus += next
            frontier = next
        }
        return corpus
    }

    /// #53 — pin the *property*, not six examples of it.
    ///
    /// `docs/logging.md` guarantees the escaped rendering is losslessly
    /// decodable. Six concrete cases pin instances of that; uniqueness itself
    /// was only ever brute-forced by hand pre-submit. This exhausts TWO
    /// corpora, because breadth and depth are needed for different defects and
    /// paying for both at once is quadratic waste (#58):
    ///
    /// - `hostileAlphabet` (15 scalars) to length 4 — breadth. Each escape arm
    ///   has a REPRESENTATIVE here, so an inter-arm collision on one of `\n`,
    ///   `\r`, `\t`, NUL or U+2028 fails here. Note the narrowness: a corpus
    ///   cannot cover a collision on a scalar it does not contain, and the
    ///   numeric arm has 100+ inhabitants. Collisions on the *other* structural
    ///   scalars (U+0085, U+2029, DEL, the rest of C0) are pinned instead by
    ///   the pairwise-distinctness check in
    ///   `testBothPathsNeutralizeTheSameScalarSet`, which costs 630 string
    ///   compares over all 36 of them rather than an exponentially larger
    ///   corpus.
    /// - `witnessAlphabet` (6 scalars) to length 5 — depth, for one witness.
    ///
    /// **Length 5 is load-bearing, not round-number.** The shortest witness
    /// separating a parity-aware reader from a parity-blind one is a fully
    /// formed literal `\u{0}` — `\`, `u`, `{`, `0`, `}` — which at length 4
    /// cannot be built. An encoder that skips doubling a backslash followed by
    /// `u` (so a literal `\u{0000}` and a real NUL render identically — #36 in
    /// a new form) passes a length-4 corpus and fails a length-5 one. That is
    /// not asserted by hand here: `testWitnessCorpusCatchesParityBlindEncoder`
    /// runs that exact decoy against `witnessAlphabet` and requires it to be
    /// caught, so narrowing the witness alphabet cannot silently drop the
    /// witness it exists to carry.
    ///
    /// Honesty boundary: this pins *value* decodability, not *field*
    /// structure. Verified — dropping `escape`'s `\"` arm leaves this green,
    /// because the outer delimiters still bracket the value.
    ///
    /// Field structure is pinned elsewhere, by two tests (#79):
    /// `testFieldSpoofingIsConfinedToOneValue` for the `\"` arm and
    /// `testTrailingBackslashCannotEscapeTheClosingQuote` for the `\\` arm.
    /// Both drive `splitLogfmt` and fail on field count when their arm is
    /// deleted, so the structural consequence is asserted rather than only the
    /// byte-level rendering that `testQuoteAndBackslashEscaped` covers.
    func testEscapedRenderingsAreUniquelyDecodable() {
        // The empty string is not filler: it is the only input that trips
        // `escape`'s `value.isEmpty` quoting arm. It comes from the hostile
        // sweep; `dropFirst` drops the witness sweep's copy of it. The two
        // sweeps still overlap on 1,554 further strings (every witness string
        // of length ≤ 4 is also a hostile one) — rendering those twice is
        // cheaper than deduplicating 63k strings.
        let corpus =
            Self.exhaustive(over: Self.hostileAlphabet, upTo: 4)
            + Self.exhaustive(over: Self.witnessAlphabet, upTo: 5).dropFirst()
        XCTAssertEqual(
            corpus.count,
            (1 + 15 + 225 + 3375 + 50625) + (6 + 36 + 216 + 1296 + 7776))

        // docs/logging.md guarantees decodability on the message body AND on
        // metadata values, so pinning one path would leave the other exactly as
        // unpinned as before.
        assertRoundTrips(
            corpus, path: "sanitizeMessage",
            render: LogFormat.sanitizeMessage, decode: { self.decodeSanitized($0) })
        assertRoundTrips(
            corpus, path: "escape",
            render: LogFormat.escape, decode: decodeEscaped)
    }

    /// One counterexample disproves the property, so report the first and stop.
    /// Asserting per-string instead buries the useful line under thousands of
    /// consequences of the same defect (reverting #36 produces 1,480 failures;
    /// the first one already names the cause).
    private func assertRoundTrips(
        _ corpus: [String], path: String,
        render: (String) -> String, decode: (String) -> String,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        var lossy: (input: String, rendered: String, decoded: String)?
        var leaked: String?
        for original in corpus {
            let rendered = render(original)
            if lossy == nil {
                let decoded = decode(rendered)
                if decoded != original {
                    lossy = (original, rendered, decoded)
                }
            }
            // The safety property the escaping exists for, over the same
            // corpus: whatever it renders, it renders on one line.
            if leaked == nil,
                rendered.unicodeScalars.contains(where: LogFormat.isLineStructural)
            {
                leaked = rendered
            }
            if lossy != nil, leaked != nil { break }
        }
        XCTAssertNil(
            lossy.map {
                "\($0.input.debugDescription) rendered as "
                    + "\($0.rendered.debugDescription), decoded back to "
                    + "\($0.decoded.debugDescription)"
            }, "\(path): rendering is not uniquely decodable",
            file: file, line: line)
        XCTAssertNil(
            leaked.map { "\(path): structural scalar survived in \($0.debugDescription)" },
            file: file, line: line)
    }

    /// #58 — the witness corpus must keep discriminating, not just exist.
    ///
    /// `testEscapedRenderingsAreUniquelyDecodable` narrowed its length-5 sweep
    /// to `witnessAlphabet` to afford a wider length-4 sweep. That trade is
    /// only sound while the narrow alphabet still builds the witness the depth
    /// was bought for, and "verified once by hand" rots. So run the decoy the
    /// doc comment names — an encoder that skips doubling a backslash followed
    /// by `u`, collapsing a literal `\u{0000}` and a real NUL onto the same
    /// rendering — and require the corpus to catch it.
    ///
    /// This is a test about a test. It fails if someone trims
    /// `witnessAlphabet` or drops the bound to 4, which is exactly the edit
    /// that would silently hollow out the round-trip property.
    func testWitnessCorpusCatchesParityBlindEncoder() {
        // `sanitizeMessage`, minus the parity rule: a backslash before `u` is
        // passed through instead of doubled.
        func parityBlindRender(_ s: String) -> String {
            var out = ""
            let scalars = Array(s.unicodeScalars)
            for (i, scalar) in scalars.enumerated() {
                if scalar == "\\" {
                    // The defect: a backslash before `u` is passed through
                    // instead of doubled. Everything else matches
                    // `LogFormat.sanitizeMessage` exactly.
                    out +=
                        (i + 1 < scalars.count && scalars[i + 1] == "u")
                        ? "\\" : "\\\\"
                } else if LogFormat.isLineStructural(scalar) {
                    out += LogFormat.escapeControl(scalar)
                } else {
                    out.unicodeScalars.append(scalar)
                }
            }
            return out
        }

        let witness = Self.exhaustive(over: Self.witnessAlphabet, upTo: 5)
        let caught = witness.contains { original in
            decodeSanitized(parityBlindRender(original)) != original
        }
        XCTAssertTrue(
            caught,
            "the length-5 witness corpus no longer catches a parity-blind "
                + "encoder — witnessAlphabet or its length bound was narrowed "
                + "past the `\\u{0}` witness, so the round-trip property is "
                + "weaker than its doc comment claims (#58)")

        // And the same decoy IS missed at length 4 — which is why depth 5 is
        // bought at all. If this ever fails, the witness got cheaper and the
        // length-5 sweep can go.
        let shallow = Self.exhaustive(over: Self.witnessAlphabet, upTo: 4)
        XCTAssertFalse(
            shallow.contains { decodeSanitized(parityBlindRender($0)) != $0 },
            "a length-4 corpus now catches the parity-blind decoy, so the "
                + "length-5 sweep is no longer load-bearing")

        // Symmetric guard for the OTHER alphabet. The whole #58 gain rides on
        // `hostileAlphabet` containing `\` together with the letters
        // `escapeControl`'s readable arms emit — and the count assertion in
        // the round-trip test pins that alphabet's SIZE, not its CONTENT, so
        // swapping one letter out keeps the count at 15 and silently restores
        // the pre-#58 blindness. Parity blindness on such a letter is the
        // defect the widening exists to catch (measured: it leaves the pre-#58
        // tier 783/0 green).
        //
        // Derived from `escapeControl`, not hardcoded, so this covers `n`, `r`
        // and `t` today and any readable arm added later — a per-letter decoy
        // would have to be remembered, and the thing being guarded against IS
        // forgetting.
        func parityBlind(on letter: Unicode.Scalar) -> (String) -> String {
            { s in
                var out = ""
                let all = Array(s.unicodeScalars)
                for (i, scalar) in all.enumerated() {
                    if scalar == "\\" {
                        out +=
                            (i + 1 < all.count && all[i + 1] == letter)
                            ? "\\" : "\\\\"
                    } else if LogFormat.isLineStructural(scalar) {
                        out += LogFormat.escapeControl(scalar)
                    } else {
                        out.unicodeScalars.append(scalar)
                    }
                }
                return out
            }
        }
        let hostile = Self.exhaustive(over: Self.hostileAlphabet, upTo: 4)
        let readableArmLetters = Self.everyStructural.compactMap { scalar -> Unicode.Scalar? in
            let rendered = Array(LogFormat.escapeControl(scalar).unicodeScalars)
            // Readable arms are exactly `\<letter>`; the numeric arm is longer.
            guard rendered.count == 2, rendered[0] == "\\" else { return nil }
            return rendered[1]
        }
        XCTAssertEqual(
            Set(readableArmLetters), Set(["n", "r", "t"] as [Unicode.Scalar]),
            "escapeControl's readable arms changed — the guard below still "
                + "adapts, but hostileAlphabet must be widened to match")
        for letter in Set(readableArmLetters) {
            XCTAssertTrue(
                Self.hostileAlphabet.contains(letter),
                "hostileAlphabet is missing `\(letter)`, a letter "
                    + "escapeControl emits — a corpus without it cannot "
                    + "express parity blindness on that arm (#58)")
            let decoy = parityBlind(on: letter)
            XCTAssertTrue(
                hostile.contains { decodeSanitized(decoy($0)) != $0 },
                "the hostile corpus no longer catches parity blindness on "
                    + "`\(letter)` — the escape character and the letters its "
                    + "arms emit must BOTH be in hostileAlphabet, which is the "
                    + "entire point of widening it (#58)")
        }
    }

    /// #35: one predicate, two callers. If `escape` and `sanitizeMessage` ever
    /// neutralize different sets, one sink grows a hole the other doesn't have.
    func testBothPathsNeutralizeTheSameScalarSet() {
        let structural: [Unicode.Scalar] = [
            "\u{00}", "\u{01}", "\n", "\r", "\t", "\u{1F}", "\u{7F}", "\u{85}",
            "\u{2028}", "\u{2029}",
        ]
        for scalar in structural {
            let s = "a\(Character(scalar))b"
            XCTAssertFalse(
                LogFormat.sanitizeMessage(s).unicodeScalars.contains(scalar),
                "message path let U+\(String(scalar.value, radix: 16)) through")
            XCTAssertFalse(
                LogFormat.escape(s).unicodeScalars.contains(scalar),
                "value path let U+\(String(scalar.value, radix: 16)) through")
        }
        // And a scalar in neither set survives both, unchanged.
        XCTAssertEqual(LogFormat.sanitizeMessage("naïve ✅"), "naïve ✅")
        XCTAssertEqual(LogFormat.escape("naïve"), "naïve")

        // #58 — non-survival is not enough: two DISTINCT structural scalars
        // must not render the SAME, or the rendering stops being decodable.
        // The round-trip corpus can only cover collisions on scalars it
        // contains (`\n`/`\r`/`\t`/NUL/U+2028), so a collision on any other
        // structural scalar was caught by nothing. Verified: adding
        // `case "\u{85}": return "\\n"` to `escapeControl` left the ENTIRE
        // tier green before this check existed.
        //
        // Swept over the FULL structural set, not the sample above. A sample
        // has exactly the defect it is here to prevent: with a 10-scalar
        // sample, `case "\u{07}": return "\\t"` still left the tier green,
        // because 27 of the 32 C0 scalars appeared in no corpus and no sample.
        // 36 scalars is 630 compares of two short strings — the exhaustive
        // set costs nothing, so take it and let the claim be true as written.
        let everyStructural = Self.everyStructural
        // Pinned to the predicate in BOTH directions, over the whole Unicode
        // scalar space. `allSatisfy` + a count would only assert subset and
        // size: widening `isLineStructural` by one scalar would leave both
        // green and quietly demote this set back to a sample — the same
        // "size, not content" hole this test closes for `hostileAlphabet`, one
        // level up. Set equality makes that unrepresentable.
        let declared = Set(everyStructural)
        var mismatches: [String] = []
        for value in UInt32(0) ... 0x10FFFF {
            guard let scalar = Unicode.Scalar(value) else { continue }
            if LogFormat.isLineStructural(scalar) != declared.contains(scalar) {
                mismatches.append("U+\(String(value, radix: 16, uppercase: true))")
                if mismatches.count >= 5 { break }
            }
        }
        XCTAssertEqual(
            mismatches, [],
            "everyStructural no longer equals the set isLineStructural "
                + "accepts, so the sweep below is a sample, not exhaustive")
        XCTAssertEqual(everyStructural.count, 36)
        for (i, a) in everyStructural.enumerated() {
            for b in everyStructural[everyStructural.index(after: i)...] {
                XCTAssertNotEqual(
                    LogFormat.escapeControl(a), LogFormat.escapeControl(b),
                    "U+\(String(a.value, radix: 16, uppercase: true)) and "
                        + "U+\(String(b.value, radix: 16, uppercase: true)) "
                        + "render identically — the escaped form is no longer "
                        + "uniquely decodable")
            }
        }
    }

    /// HONESTY BOUNDARY, pinned so it is not mistaken for a defect later.
    ///
    /// The message is unquoted and precedes the tail, so `principal=` text in a
    /// message IS the first match on the line. `escape` prevents this for
    /// metadata by quoting; the message cannot be treated the same way because
    /// messages legitimately carry `key=value` text. The rule lives in
    /// `docs/logging.md`: the message is untrusted text, the tail is the
    /// structured record.
    func testMessageDoesNotDefendAgainstFieldForgery() {
        let merged = LogFormat.merge(
            handler: [:], provided: ["principal": "u:alice"], event: nil,
            function: "f")
        let line = LogFormat.terminalText(
            message: "denied principal=admin", merged: merged)
        XCTAssertEqual(line, "denied principal=admin function=f principal=u:alice")
        // The forged value precedes the real one — a reader taking the first
        // match sees `admin`. Documented, not defended.
        let first = line.range(of: "principal=")
        XCTAssertNotNil(first)
        XCTAssertTrue(
            line[first!.upperBound...].hasPrefix("admin"),
            "if this ever fails, the message path gained a defense the docs "
                + "and the comment at sanitizeMessage both say it lacks")
    }

    func testMessageControlCharactersNeutralized() {
        XCTAssertEqual(LogFormat.sanitizeMessage("a\rb"), #"a\rb"#)
        XCTAssertEqual(LogFormat.sanitizeMessage("a\u{0001}b"), #"a\u{0001}b"#)
        XCTAssertEqual(
            LogFormat.sanitizeMessage("a\u{2028}b"), #"a\u{2028}b"#)
    }

    /// #33: `errorFieldLimit` bounds the description, NOT the rendered field —
    /// escaping runs afterwards and expands control characters ~8×. Pinned so
    /// the comment and the code cannot drift apart again.
    func testRenderedErrorFieldMayExceedTheDescriptionBound() {
        struct Controls: Error, CustomStringConvertible {
            let description = String(repeating: "\u{0001}", count: 5_000)
        }
        let merged = LogFormat.merge(
            handler: [:], provided: nil, event: nil, function: "f",
            error: Controls())
        let description = "\(merged["error"] ?? "")"
        XCTAssertLessThanOrEqual(
            description.utf8.count, LogFormat.errorFieldLimit + 3,
            "the description itself is bounded")

        let rendered = LogFormat.pairs(merged)
        XCTAssertGreaterThan(
            rendered.utf8.count, LogFormat.errorFieldLimit,
            "escaping expands past the description bound — this is the "
                + "documented behaviour, not a regression")
        XCTAssertLessThan(rendered.utf8.count, LogFormat.errorFieldLimit * 9)
    }
    struct Multiline: Error, CustomStringConvertible {
        let description = "a\nb"
    }

    func testCarriageReturnAndTabEscaped() {
        XCTAssertEqual(LogFormat.escape("a\rb"), #""a\rb""#)
        XCTAssertEqual(LogFormat.escape("a\tb"), #""a\tb""#)
    }

    /// Quotes and backslashes are escaped, so the quoted form stays parseable.
    func testQuoteAndBackslashEscaped() {
        XCTAssertEqual(LogFormat.escape(#"say "hi""#), #""say \"hi\"""#)
        XCTAssertEqual(LogFormat.escape(#"a\b c"#), #""a\\b c""#)
    }

    /// Unicode line separators break a line-oriented reader just like `\n`.
    func testUnicodeLineSeparatorsEscaped() {
        for sep in ["\u{2028}", "\u{2029}", "\u{85}"] {
            let out = LogFormat.escape("a\(sep)b")
            XCTAssertTrue(out.hasPrefix("\""), "\(sep.unicodeScalars) unquoted")
            XCTAssertFalse(out.unicodeScalars.contains { $0 > "\u{7E}" })
        }
    }

    func testEmptyValueIsQuoted() {
        XCTAssertEqual(LogFormat.escape(""), "\"\"")
    }

    // MARK: - the two sink shapes

    /// Terminal: `<msg> k=v k=v`. Unified: `<msg> {k=v k=v}`. The shapes differ
    /// deliberately; nothing else catches a regression in either.
    func testTerminalAndUnifiedShapesDiffer() {
        let merged = LogFormat.merge(
            handler: [:], provided: nil, event: ["k": "v"], function: "f")
        XCTAssertEqual(
            LogFormat.terminalText(message: "up", merged: merged),
            "up function=f k=v")
        XCTAssertEqual(
            LogFormat.unifiedText(message: "up", merged: merged),
            "up {function=f k=v}")
    }

    /// Empty metadata ⇒ no trailing separator in either sink. (Unreachable in
    /// production, since `function=` is always merged in — pinned so the
    /// renderers stay honest if called directly.)
    func testEmptyMetadataAddsNoSeparator() {
        XCTAssertEqual(
            LogFormat.terminalText(message: "up", merged: [:]), "up")
        XCTAssertEqual(LogFormat.unifiedText(message: "up", merged: [:]), "up")
    }
}
