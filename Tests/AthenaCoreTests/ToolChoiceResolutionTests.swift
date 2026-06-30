import Foundation
import XCTest

@testable import AthenaServerKit

/// ADR 034 — pins the `tool_choice` → (force, advertise) algebra. The fix:
/// `auto`/absent must NOT force a tool call (the old bug treated it as
/// `required`, grammar-masking the model into an infinite tool loop). MLX-free,
/// fast tier (ADR 009).
final class ToolChoiceResolutionTests: XCTestCase {
    private let two = ["web_search", "calc"]

    func testAutoDoesNotForceButAdvertises() {
        let r = resolveToolChoice(mode: .auto, toolNames: two)
        XCTAssertEqual(r.forcedToolNames, [])  // the fix: no forcing
        XCTAssertTrue(r.advertiseMenu)
    }

    func testRequiredForcesAll() {
        let r = resolveToolChoice(mode: .required, toolNames: two)
        XCTAssertEqual(r.forcedToolNames, two)
        XCTAssertTrue(r.advertiseMenu)
    }

    func testNamedHitForcesOne() {
        let r = resolveToolChoice(mode: .named("calc"), toolNames: two)
        XCTAssertEqual(r.forcedToolNames, ["calc"])
        XCTAssertTrue(r.advertiseMenu)
    }

    /// A named choice matching no declared tool falls through to auto
    /// (no force, menu advertised) — preserves prior forced-name-miss behavior.
    func testNamedMissFallsThroughToAuto() {
        let r = resolveToolChoice(mode: .named("nope"), toolNames: two)
        XCTAssertEqual(r.forcedToolNames, [])
        XCTAssertTrue(r.advertiseMenu)
    }

    /// `none` forbids tool calls: no force AND no menu (the model cannot call).
    func testNoneSuppressesMenu() {
        let r = resolveToolChoice(mode: .none, toolNames: two)
        XCTAssertEqual(r.forcedToolNames, [])
        XCTAssertFalse(r.advertiseMenu)
    }

    /// No tools declared ⇒ nothing to force or advertise, whatever the mode.
    func testNoToolsIsInert() {
        for mode: ToolChoiceMode in [.auto, .required, .none, .named("x")] {
            let r = resolveToolChoice(mode: mode, toolNames: [])
            XCTAssertEqual(r.forcedToolNames, [])
            XCTAssertFalse(r.advertiseMenu)
        }
    }

    func testSingleToolAutoStillFree() {
        let r = resolveToolChoice(mode: .auto, toolNames: ["only"])
        XCTAssertEqual(r.forcedToolNames, [])
        XCTAssertTrue(r.advertiseMenu)
    }
}
