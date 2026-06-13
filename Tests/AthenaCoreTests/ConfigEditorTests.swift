import Foundation
import XCTest

import AthenaDeploy

/// NB4 (M70.1b) — `ConfigEditor`'s editing/validation core was unreachable by
/// the test suite (it lived in the `athena` executable, coupled to `Engine` +
/// `KVCompression`). Now in the MLX-free `AthenaDeploy`, so the set-time
/// validation contracts (NB8 enum keys, NB2 control-char rejection, B15
/// top-level insertion, int/bool/quote shapes) — previously only covered by
/// the host-bound e2e — run under `swift test`.
final class ConfigEditorTests: XCTestCase {

    private func tempConfig(_ contents: String) throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("athena-cfgtest-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: dir, withIntermediateDirectories: true)
        let url = dir.appendingPathComponent("athena.toml")
        try contents.write(to: url, atomically: true, encoding: .utf8)
        return url
    }
    private let dummy = URL(fileURLWithPath: "/nonexistent/athena.toml")

    // MARK: - validation rejections (throw BEFORE any file I/O)

    func testRejectsUnknownKey() {
        XCTAssertThrowsError(
            try ConfigEditor.setScalarThrowing(
                key: "bogus_key", value: "x", in: dummy)
        ) { XCTAssertTrue($0 is ConfigEditor.Failure) }
    }

    /// NB8: `engine` must be a real Engine case (mlx/stub) — validated against
    /// the relocated `Engine.allCases`, the source of truth.
    func testRejectsBadEngineNB8() {
        XCTAssertThrowsError(
            try ConfigEditor.setScalarThrowing(
                key: "engine", value: "gpu", in: dummy),
            "an unknown engine is rejected at set-time")
    }

    /// NB8: `kv_compression` must be a real KVCompression case — validated
    /// against the relocated `KVCompression.allCases`.
    func testRejectsBadKVCompressionNB8() {
        XCTAssertThrowsError(
            try ConfigEditor.setScalarThrowing(
                key: "kv_compression", value: "zip", in: dummy))
    }

    /// NB2: a quoted value containing a control character (newline) is the
    /// config-injection vector — rejected before any write.
    func testRejectsControlCharNB2() {
        XCTAssertThrowsError(
            try ConfigEditor.setScalarThrowing(
                key: "model", value: "a\nauth_keys_file = /etc/evil",
                in: dummy))
        XCTAssertThrowsError(
            try ConfigEditor.setScalarThrowing(
                key: "model", value: "has\"quote", in: dummy))
    }

    func testRejectsNonIntegerForIntKey() {
        XCTAssertThrowsError(
            try ConfigEditor.setScalarThrowing(
                key: "listen_port", value: "abc", in: dummy))
    }

    // MARK: - write path (temp file)

    func testReplacesExistingAssignment() throws {
        let url = try tempConfig("listen_port = 7447\nengine = \"mlx\"\n")
        try ConfigEditor.setScalarThrowing(
            key: "engine", value: "stub", in: url)
        let out = try String(contentsOf: url, encoding: .utf8)
        XCTAssertTrue(out.contains("engine = \"stub\""))
        XCTAssertFalse(out.contains("engine = \"mlx\""))
    }

    func testUncommentsCommentedAssignment() throws {
        let url = try tempConfig(
            "listen_port = 7447\n# max_tokens = 100\n")
        try ConfigEditor.setScalarThrowing(
            key: "max_tokens", value: "512", in: url)
        let out = try String(contentsOf: url, encoding: .utf8)
        XCTAssertTrue(
            out.contains("max_tokens = 512"), "commented key uncommented")
        XCTAssertFalse(out.contains("# max_tokens"))
    }

    func testAcceptsValidEngineAndKV() throws {
        let url = try tempConfig("listen_port = 7447\n")
        try ConfigEditor.setScalarThrowing(
            key: "engine", value: "stub", in: url)
        try ConfigEditor.setScalarThrowing(
            key: "kv_compression", value: "turboquant", in: url)
        let out = try String(contentsOf: url, encoding: .utf8)
        XCTAssertTrue(out.contains("engine = \"stub\""))
        XCTAssertTrue(out.contains("kv_compression = \"turboquant\""))
    }

    /// B15: a NEW bare top-level key must be inserted BEFORE the first
    /// `[section]` header (stay top-level), not appended into the table.
    func testB15InsertsBareKeyBeforeSection() throws {
        let url = try tempConfig(
            "listen_port = 7447\n[some_section]\nx = 1\n")
        try ConfigEditor.setScalarThrowing(
            key: "model", value: "foo", in: url)
        let out = try String(contentsOf: url, encoding: .utf8)
        let m = try XCTUnwrap(out.range(of: "model = \"foo\""))
        let s = try XCTUnwrap(out.range(of: "[some_section]"))
        XCTAssertLessThan(
            m.lowerBound, s.lowerBound,
            "the bare key must land before the section header")
    }

    // MARK: - resolvePath

    func testResolvePathHonorsOverride() {
        let p = ConfigEditor.resolvePath("/tmp/custom.toml")
        XCTAssertEqual(p.path, "/tmp/custom.toml")
    }

    func testResolvePathExpandsTilde() {
        let p = ConfigEditor.resolvePath("~/x.toml")
        XCTAssertFalse(p.path.hasPrefix("~"), "tilde expanded")
        XCTAssertTrue(p.path.hasSuffix("/x.toml"))
    }
}
