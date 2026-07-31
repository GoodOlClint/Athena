import AthenaCore
import AthenaServerKit
import Foundation
import XCTest

/// Usability audit 2026-07-02 §2/§3 — model-op progress SSE frames + throttle.
/// Pins the wire format and the emit-rate decision (ADR 008/009), including the
/// two-way backward compatibility the additive events depend on.
final class ModelOpProgressFrameTests: XCTestCase {
    private func decode(_ p: ModelOpProgress) -> [String: Any] {
        try! JSONSerialization.jsonObject(
            with: ModelOpProgressFrame.json(p)) as! [String: Any]
    }

    func testFrameShapes() {
        let dl = decode(.download(fraction: 0.5, bytes: 5, total: 10))
        XCTAssertEqual(dl["event"] as? String, "progress")
        XCTAssertEqual(dl["fraction"] as? Double, 0.5)
        XCTAssertEqual((dl["bytes"] as? NSNumber)?.intValue, 5)
        XCTAssertEqual((dl["total"] as? NSNumber)?.intValue, 10)

        let f = decode(
            .file(name: "shard1", index: 2, count: 6, bytes: 3, total: 9, done: false))
        XCTAssertEqual(f["event"] as? String, "file")
        XCTAssertEqual(f["name"] as? String, "shard1")
        XCTAssertEqual((f["index"] as? NSNumber)?.intValue, 2)
        XCTAssertEqual((f["count"] as? NSNumber)?.intValue, 6)
        XCTAssertEqual(f["done"] as? Bool, false)

        XCTAssertEqual(decode(.phase("quantize"))["phase"] as? String, "quantize")
        let q = decode(.quantize(index: 715, count: 1743))
        XCTAssertEqual(q["event"] as? String, "quantize")
        XCTAssertEqual((q["index"] as? NSNumber)?.intValue, 715)
        XCTAssertEqual((q["count"] as? NSNumber)?.intValue, 1743)
    }

    /// Backward compat both ways: (a) an OLD client reads only
    /// `event=="progress"` + `fraction` — the download frame still carries them;
    /// (b) a NEW client seeing a daemon that only emits `progress` still gets a
    /// usable aggregate. The additive events carry a distinct `event` an old
    /// client ignores.
    func testProgressFrameRemainsOldClientReadable() {
        let dl = decode(.download(fraction: 0.25, bytes: 0, total: 0))
        XCTAssertEqual(dl["event"] as? String, "progress")
        XCTAssertNotNil(dl["fraction"] as? Double)
        for p: ModelOpProgress in [
            .file(name: "x", index: 1, count: 1, bytes: 0, total: 0, done: true),
            .phase("load"), .quantize(index: 1, count: 2),
        ] {
            XCTAssertNotEqual(decode(p)["event"] as? String, "progress")
        }
    }

    func testThrottle() {
        // First frame always emits.
        XCTAssertTrue(
            ModelOpProgressFrame.shouldEmit(
                nowMs: 0, lastMs: nil, fraction: 0, lastFraction: nil))
        // Within interval + sub-1% delta ⇒ suppress.
        XCTAssertFalse(
            ModelOpProgressFrame.shouldEmit(
                nowMs: 100, lastMs: 0, fraction: 0.105, lastFraction: 0.10))
        // ≥1% delta ⇒ emit even within interval.
        XCTAssertTrue(
            ModelOpProgressFrame.shouldEmit(
                nowMs: 100, lastMs: 0, fraction: 0.13, lastFraction: 0.10))
        // ≥500ms elapsed ⇒ emit even on tiny delta.
        XCTAssertTrue(
            ModelOpProgressFrame.shouldEmit(
                nowMs: 600, lastMs: 0, fraction: 0.101, lastFraction: 0.10))
        // `done` always emits.
        XCTAssertTrue(
            ModelOpProgressFrame.shouldEmit(
                nowMs: 1, lastMs: 0, fraction: 1, lastFraction: 1, done: true))
    }
}
