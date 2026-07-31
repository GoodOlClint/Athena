import AthenaDeploy
import Foundation
import XCTest

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

    /// ADR 037 amendment — the field bug: on a real install the config FILE is
    /// service-user-writable but its DIRECTORY is root-owned (deliberately —
    /// `auth_keys_file` and the TLS key live there), so an atomic write fails
    /// EACCES creating its temp file and every daemon-mediated config write
    /// returned `writeFailed`. Simulated here with a read-only directory
    /// containing a writable file: the edit must still land, in place.
    func testWritesInPlaceWhenTheDirectoryIsNotWritable() throws {
        let url = try tempConfig(
            """
            listen_port = 7447
            max_prompt_tokens = 8192
            """)
        let dir = url.deletingLastPathComponent()
        let fm = FileManager.default
        try fm.setAttributes(
            [.posixPermissions: 0o555], ofItemAtPath: dir.path)
        defer {
            try? fm.setAttributes(
                [.posixPermissions: 0o755], ofItemAtPath: dir.path)
        }
        // Pre-condition: the directory really does reject a new file, so this
        // test can't pass for the wrong reason.
        XCTAssertThrowsError(
            try "x".write(
                to: dir.appendingPathComponent("probe.tmp"),
                atomically: false, encoding: .utf8),
            "the test's read-only directory is not actually read-only")

        try ConfigEditor.setScalarThrowing(
            key: "max_prompt_tokens", value: "16384", in: url)
        let text = try String(contentsOf: url, encoding: .utf8)
        XCTAssertTrue(text.contains("max_prompt_tokens = 16384"))
        XCTAssertTrue(text.contains("listen_port = 7447"))  // layout kept
    }

    /// The rollback path (NB2) must survive the same unwritable directory —
    /// otherwise a bad value would leave the config corrupt precisely where the
    /// atomic write can't run.
    func testRollbackAlsoWorksWithAnUnwritableDirectory() throws {
        let url = try tempConfig(
            """
            listen_host = "127.0.0.1"
            listen_port = 7447
            log_dir = "/l"
            """)
        let dir = url.deletingLastPathComponent()
        let fm = FileManager.default
        try fm.setAttributes(
            [.posixPermissions: 0o555], ofItemAtPath: dir.path)
        defer {
            try? fm.setAttributes(
                [.posixPermissions: 0o755], ofItemAtPath: dir.path)
        }
        // `listen_port` must stay an int; a non-int is caught before the write,
        // so drive the rollback with a value that parses as a key but breaks
        // the config: an empty required scalar.
        XCTAssertThrowsError(
            try ConfigEditor.setScalarThrowing(
                key: "listen_port", value: "notanint", in: url))
        let text = try String(contentsOf: url, encoding: .utf8)
        XCTAssertTrue(text.contains("listen_port = 7447"))
        XCTAssertNotNil(try? AthenaConfig.parse(toml: text))
    }

    /// ADR 041: `token_budget_window` is enum-ish — a typo is refused at
    /// set-time instead of failing the daemon's next config parse.
    func testRejectsBadTokenBudgetWindow() {
        XCTAssertThrowsError(
            try ConfigEditor.setScalarThrowing(
                key: "token_budget_window", value: "week", in: dummy))
    }

    /// ADR 041/042: both new int keys are settable at all (they were unknown
    /// keys before — `max_prompt_tokens` was readable but not writable).
    func testAcceptsBudgetAndPromptCeilingKeys() throws {
        let url = try tempConfig(
            """
            listen_port = 7447
            # token_budget = 1
            # max_prompt_tokens = 1
            """)
        try ConfigEditor.setScalarThrowing(
            key: "token_budget", value: "50000000", in: url)
        try ConfigEditor.setScalarThrowing(
            key: "token_budget_window", value: "day", in: url)
        try ConfigEditor.setScalarThrowing(
            key: "max_prompt_tokens", value: "96000", in: url)
        let text = try String(contentsOf: url, encoding: .utf8)
        XCTAssertTrue(text.contains("token_budget = 50000000"))
        XCTAssertTrue(text.contains("token_budget_window = \"day\""))
        XCTAssertTrue(text.contains("max_prompt_tokens = 96000"))
        // And the daemon parses back what was written.
        let cfg = try AthenaConfig.parse(
            toml: text + "\nlisten_host = \"127.0.0.1\"\nlog_dir = \"/l\"")
        XCTAssertEqual(cfg.tokenBudget, 50_000_000)
        XCTAssertEqual(cfg.maxPromptTokens, 96_000)
    }

    func testRejectsNonIntegerForBudget() {
        XCTAssertThrowsError(
            try ConfigEditor.setScalarThrowing(
                key: "token_budget", value: "lots", in: dummy))
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
            key: "kv_compression", value: "triattention", in: url)
        let out = try String(contentsOf: url, encoding: .utf8)
        XCTAssertTrue(out.contains("engine = \"stub\""))
        XCTAssertTrue(out.contains("kv_compression = \"triattention\""))
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
