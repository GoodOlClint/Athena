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
    /// field's value, so anything that parses the tail sees exactly two fields
    /// and one `principal`. It does NOT remove the text — a naive
    /// `grep principal=` still matches inside the quotes. Structure is
    /// defended; substring search is not, and cannot be while the description
    /// is kept at all.
    func testFieldSpoofingIsConfinedToOneValue() {
        let merged = LogFormat.merge(
            handler: [:], provided: ["principal": "u:alice"], event: nil,
            function: "f", error: Spoof())
        let rendered = LogFormat.pairs(merged)
        XCTAssertEqual(
            rendered,
            "error=\"denied principal=admin\" function=f principal=u:alice")
        // Unquoted, the tail would have parsed as four fields with a forged
        // `principal=admin` ahead of the real one; quoted, it is three.
        XCTAssertEqual(splitLogfmt(rendered).count, 3)
        XCTAssertEqual(splitLogfmt(rendered)["principal"], "u:alice")
    }

    /// Minimal logfmt reader: splits on spaces outside double quotes. Stands in
    /// for any structure-aware consumer of the tail.
    private func splitLogfmt(_ s: String) -> [String: String] {
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
        return fields.reduce(into: [:]) { acc, f in
            guard let eq = f.firstIndex(of: "=") else { return }
            let key = String(f[f.startIndex ..< eq])
            var value = String(f[f.index(after: eq)...])
            if value.hasPrefix("\"") && value.hasSuffix("\"") {
                value = String(value.dropFirst().dropLast())
            }
            acc[key] = value
        }
    }
    struct Spoof: Error, CustomStringConvertible {
        let description = "denied principal=admin"
    }

    /// The worst case for the terminal sink: it writes the body straight to
    /// stderr, so a raw newline would forge an entire additional log line.
    func testNewlineCannotForgeALine() {
        let merged = LogFormat.merge(
            handler: [:], provided: nil, event: nil, function: "f",
            error: Multiline())
        let text = LogFormat.terminalText(message: "failed", merged: merged)
        XCTAssertFalse(
            text.contains("\n"), "a newline in a value must not reach stderr")
        XCTAssertTrue(text.contains(#"error="a\nb""#))
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
