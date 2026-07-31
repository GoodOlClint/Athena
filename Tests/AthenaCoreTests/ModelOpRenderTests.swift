import AthenaClient
import Foundation
import XCTest

/// Usability audit 2026-07-02 §2/§3 — the pure render state behind the
/// ollama-style multi-row progress. `ModelOpState.lines` is what makes local
/// and remote render identically, so it is unit-pinned (ADR 008/009).
final class ModelOpRenderTests: XCTestCase {
    func testBar() {
        XCTAssertEqual(renderBar(0, width: 4), "[----]")
        XCTAssertEqual(renderBar(0.5, width: 4), "[##--]")
        XCTAssertEqual(renderBar(1, width: 4), "[####]")
        XCTAssertEqual(renderBar(2, width: 4), "[####]")  // clamps
    }

    func testMultiFileRowsThenTotal() {
        var s = ModelOpState(label: "pull")
        s.file(name: "a.safetensors", bytes: 50, total: 100, done: false)
        s.file(name: "b.safetensors", bytes: 100, total: 100, done: true)
        s.download(fraction: 0.75, bytes: 150, total: 200)
        let lines = s.lines()
        // One row for the in-progress file, plus a total row; the done file is
        // collapsed into the "(1 done)" summary, not its own row.
        XCTAssertTrue(lines.contains { $0.contains("a.safetensors") })
        XCTAssertFalse(lines.contains { $0.contains("b.safetensors") })
        XCTAssertTrue(lines.contains { $0.contains("75%") })
        XCTAssertTrue(lines.contains { $0.contains("(1 done)") })
    }

    func testActiveRowCapCollapses() {
        var s = ModelOpState(label: "pull")
        for i in 0 ..< 12 {
            s.file(name: "shard\(i)", bytes: 1, total: 10, done: false)
        }
        let lines = s.lines(maxRows: 8)
        XCTAssertTrue(lines.contains { $0.contains("+4 more downloading") })
    }

    func testQuantizeRow() {
        var s = ModelOpState(label: "convert")
        s.phase("quantize")
        s.quantize(index: 715, count: 1743)
        let lines = s.lines()
        XCTAssertTrue(lines.contains { $0.contains("715/1743") })
        XCTAssertTrue(lines.contains { $0.contains("quantize") })
    }

    func testPhaseOnlyBeforeAnyBytes() {
        var s = ModelOpState(label: "convert")
        s.phase("load")
        XCTAssertTrue(s.lines().contains { $0.contains("load") })
    }
}
