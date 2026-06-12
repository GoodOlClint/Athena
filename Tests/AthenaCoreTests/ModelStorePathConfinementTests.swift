import Foundation
import XCTest

@testable import AthenaCore
@testable import AthenaLLM

/// M65.4 — path confinement for the model store. These pin the
/// traversal guards so a crafted model id / name can never resolve,
/// read, or delete outside the store root.
final class ModelStorePathConfinementTests: XCTestCase {

    private func tmpDir() throws -> URL {
        let u = FileManager.default.temporaryDirectory
            .appendingPathComponent("athena-confine-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: u, withIntermediateDirectories: true)
        return u
    }

    // MARK: D6 / NC12 — shared resolver confinement

    func testLocalDirectoryResolvesABareChild() throws {
        let root = try tmpDir()
        defer { try? FileManager.default.removeItem(at: root) }
        let model = root.appendingPathComponent("good", isDirectory: true)
        try FileManager.default.createDirectory(
            at: model, withIntermediateDirectories: true)

        // Bare name and full HF id (same `modelStoreIdentity`) both resolve.
        XCTAssertEqual(
            ModelStoreLayout.localDirectory(for: "good", storeRoot: root)?
                .standardizedFileURL,
            model.standardizedFileURL)
        XCTAssertEqual(
            ModelStoreLayout.localDirectory(for: "org/good", storeRoot: root)?
                .standardizedFileURL,
            model.standardizedFileURL)
    }

    func testLocalDirectoryRejectsTraversalIdentity() throws {
        let root = try tmpDir()
        defer { try? FileManager.default.removeItem(at: root) }
        // A sibling dir next to the store root — what a `..` escape would
        // reach. It exists on disk, so only the guard prevents resolution.
        let sibling = root.deletingLastPathComponent()
            .appendingPathComponent("athena-confine-sibling")
        try? FileManager.default.createDirectory(
            at: sibling, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: sibling) }

        // `org/..` → identity `..` → would be root's PARENT. Refused.
        XCTAssertNil(
            ModelStoreLayout.localDirectory(for: "org/..", storeRoot: root))
        XCTAssertNil(
            ModelStoreLayout.localDirectory(for: "..", storeRoot: root))
        XCTAssertNil(
            ModelStoreLayout.localDirectory(for: ".", storeRoot: root))
    }

    // MARK: isValidName

    func testIsValidNameRejectsEscapes() {
        for bad in ["..", ".", "", "a/b", "org/name", "/abs"] {
            XCTAssertFalse(
                ModelStoreOps.isValidName(bad), "should reject \(bad)")
        }
        for ok in ["model-4bit", "gemma-3-27b-it", "a_b.c"] {
            XCTAssertTrue(
                ModelStoreOps.isValidName(ok), "should accept \(ok)")
        }
    }

    // MARK: C7 — copy confines the SOURCE, not just the destination

    func testCopyRejectsTraversalSource() throws {
        let root = try tmpDir()
        defer { try? FileManager.default.removeItem(at: root) }
        XCTAssertThrowsError(
            try ModelStoreOps.copy(
                root: root, src: "../secret", dst: "clone",
                deepCopy: false, force: false)
        ) { err in
            guard case ModelStoreOps.OpError.invalidName = err else {
                return XCTFail("expected invalidName, got \(err)")
            }
        }
    }
}
