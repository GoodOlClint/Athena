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
