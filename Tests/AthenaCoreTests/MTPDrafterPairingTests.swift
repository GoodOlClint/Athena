import Foundation
import XCTest

@testable import AthenaCore

/// ADR 032 — the pure MTP target↔drafter pairing decision + the TOML map
/// parser, plus a smoke check that the bundled seed map loads (ADR 008/009).
final class MTPDrafterPairingTests: XCTestCase {
    private let map = [
        "gemma-4-e4b-it-4bit": "mlx-community/gemma-4-E4B-it-assistant-bf16"
    ]

    // MARK: resolve

    func testExplicitKeyWins() {
        let d = MTPDrafterPairing.resolve(
            targetID: "mlx-community/gemma-4-e4b-it-4bit",
            explicit: "my-org/custom-drafter", defaults: map)
        XCTAssertEqual(d, "my-org/custom-drafter")
    }

    func testBlankExplicitFallsThroughToMap() {
        let d = MTPDrafterPairing.resolve(
            targetID: "gemma-4-e4b-it-4bit", explicit: "   ", defaults: map)
        XCTAssertEqual(d, "mlx-community/gemma-4-E4B-it-assistant-bf16")
    }

    func testMapLookupStripsOrgAndIgnoresCase() {
        // Org prefix dropped, case-insensitive — a local conversion with the
        // same basename still pairs.
        let d = MTPDrafterPairing.resolve(
            targetID: "local/Gemma-4-E4B-IT-4bit", explicit: nil, defaults: map)
        XCTAssertEqual(d, "mlx-community/gemma-4-E4B-it-assistant-bf16")
    }

    func testUnmappedTargetResolvesNil() {
        XCTAssertNil(
            MTPDrafterPairing.resolve(
                targetID: "mlx-community/some-other-llm", explicit: nil,
                defaults: map))
    }

    // MARK: parse

    func testParseIgnoresCommentsHeadersBlanksAndUnquotes() {
        let toml = """
            # a comment
            [drafters]

            "gemma-4-e4b-it-4bit" = "mlx-community/gemma-4-E4B-it-assistant-bf16"
            # another
            "org/gemma-4-e2b-it-4bit" = "mlx-community/gemma-4-E2B-it-assistant-bf16"
            """
        let parsed = MTPDrafterPairing.parse(toml)
        XCTAssertEqual(parsed.count, 2)
        // Key normalized to lowercased basename (org prefix stripped).
        XCTAssertEqual(
            parsed["gemma-4-e4b-it-4bit"],
            "mlx-community/gemma-4-E4B-it-assistant-bf16")
        XCTAssertEqual(
            parsed["gemma-4-e2b-it-4bit"],
            "mlx-community/gemma-4-E2B-it-assistant-bf16")
    }

    // MARK: bundled seed

    func testBundledSeedMapResolvesKnownPair() {
        // Validates the SwiftPM resource is bundled and parseable end-to-end.
        let seed = MTPDrafterPairing.defaultMap(dataDir: nil)
        XCTAssertEqual(
            MTPDrafterPairing.resolve(
                targetID: "mlx-community/gemma-4-e4b-it-4bit", explicit: nil,
                defaults: seed),
            "mlx-community/gemma-4-E4B-it-assistant-bf16")
    }
}
