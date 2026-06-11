import AthenaDeploy
import XCTest

@testable import AthenaModels

/// M63.3b — CI-safe (no MLX) tests for the DFlash config + registry
/// touchpoints: the target→drafter mapping and the `dflash_enabled` TOML
/// key. The drafter resolves on the resident model's STORE name (e.g.
/// `gemma-4-31b-it-4bit`), not an HF snapshot SHA — a real consideration
/// for the dispatch path.
final class DFlashConfigTests: XCTestCase {

    func testRegistryResolvesKnownTargets() {
        XCTAssertEqual(
            DFlashRegistry.draftId(forModel: "gemma-4-31b-it-4bit"),
            "z-lab/gemma-4-31B-it-DFlash")
        XCTAssertEqual(
            DFlashRegistry.draftId(forModel: "gemma-4-26b-a4b-it-4bit"),
            "z-lab/gemma-4-26B-A4B-it-DFlash")
        // Case-insensitive substring.
        XCTAssertEqual(
            DFlashRegistry.draftId(forModel: "Gemma-4-31B-it-8bit"),
            "z-lab/gemma-4-31B-it-DFlash")
    }

    func testRegistryMissesUnregisteredOrSnapshotNames() {
        // No drafter → DFlash does not engage (request decodes normally).
        XCTAssertNil(DFlashRegistry.draftId(forModel: "gemma-4-12B-it-8bit"))
        XCTAssertNil(DFlashRegistry.draftId(forModel: "Qwen3.5-27B-4bit-mtp"))
        XCTAssertNil(DFlashRegistry.draftId(forModel: "Llama-3.2-1B-Instruct-4bit"))
        // An HF snapshot SHA dir name does NOT match — the registry keys on
        // the store name, so the model must be loaded under its real name.
        XCTAssertNil(
            DFlashRegistry.draftId(
                forModel: "0d17175ead577037577f24963de2f0f5fafb72a5"))
    }

    func testDflashEnabledParsesFromToml() throws {
        func cfg(_ extra: String) throws -> AthenaConfig {
            try AthenaConfig.parse(
                toml: "listen_host = \"127.0.0.1\"\nlisten_port = 7447\n"
                    + "log_dir = \"/tmp\"\n\(extra)")
        }
        XCTAssertEqual(try cfg("dflash_enabled = true\n").dflashEnabled, true)
        XCTAssertEqual(try cfg("dflash_enabled = false\n").dflashEnabled, false)
        // Absent ⇒ nil ⇒ daemon default false.
        XCTAssertNil(try cfg("").dflashEnabled)
    }
}
