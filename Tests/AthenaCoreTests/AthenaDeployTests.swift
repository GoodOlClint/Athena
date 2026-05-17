import Foundation
import XCTest

@testable import AthenaDeploy

final class AthenaConfigTests: XCTestCase {

    func testParsesDefaultsWithCommentedOptionals() throws {
        let toml = """
            # comment line
            listen_host = "127.0.0.1"
            listen_port = 7447
            # budget_bytes = 36000000000
            engine = "mlx"
            # model = "Qwen3.5-27B-4bit-mtp"
            log_dir = "/usr/local/var/log/athena"
            """
        let c = try AthenaConfig.parse(toml: toml)
        XCTAssertEqual(c.listenHost, "127.0.0.1")
        XCTAssertEqual(c.listenPort, 7447)
        XCTAssertNil(c.budgetBytes)
        XCTAssertEqual(c.engine, "mlx")
        XCTAssertNil(c.model)
        XCTAssertEqual(c.logDir, "/usr/local/var/log/athena")
    }

    func testInlineCommentsAndQuotesStripped() throws {
        let toml = """
            listen_host = "0.0.0.0"   # bind all
            listen_port = 7447
            budget_bytes = 36000000000  # 36 GB
            model = "Qwen3.6-27B-8bit-mtp"
            log_dir = "/var/log/athena"
            """
        let c = try AthenaConfig.parse(toml: toml)
        XCTAssertEqual(c.listenHost, "0.0.0.0")
        XCTAssertEqual(c.budgetBytes, 36_000_000_000)
        XCTAssertEqual(c.model, "Qwen3.6-27B-8bit-mtp")
    }

    func testKeyPrefixIsNotMistaken() throws {
        // `listen_port` must not be satisfied by some `listen_portx`.
        let toml = """
            listen_host = "h"
            listen_portx = 1
            listen_port = 7447
            log_dir = "/l"
            """
        XCTAssertEqual(try AthenaConfig.parse(toml: toml).listenPort, 7447)
    }

    func testMissingRequiredKeyThrows() {
        let toml = "listen_host = \"h\"\nlog_dir = \"/l\""
        XCTAssertThrowsError(try AthenaConfig.parse(toml: toml)) {
            XCTAssertEqual(
                $0 as? AthenaConfig.ParseError,
                .missingRequiredKey("listen_port"))
        }
    }

    func testInvalidIntThrows() {
        let toml = "listen_host=\"h\"\nlisten_port=\"abc\"\nlog_dir=\"/l\""
        XCTAssertThrowsError(try AthenaConfig.parse(toml: toml)) {
            XCTAssertEqual(
                $0 as? AthenaConfig.ParseError,
                .invalidInt(key: "listen_port", value: "abc"))
        }
    }

    func testCommittedDeployConfigParses() throws {
        // Guards deploy/athena.toml against schema drift.
        let repoRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()  // AthenaCoreTests
            .deletingLastPathComponent()  // Tests
            .deletingLastPathComponent()  // repo root
        let cfg = try AthenaConfig.parse(
            file: repoRoot.appendingPathComponent("deploy/athena.toml"))
        XCTAssertEqual(cfg.listenPort, 7447)
        XCTAssertEqual(cfg.engine, "mlx")
        XCTAssertNil(cfg.budgetBytes)
    }
}

final class LaunchdPlistTests: XCTestCase {

    private func cfg(
        budget: Int? = nil, model: String? = nil, dataDir: String? = nil,
        logLevel: String? = nil, syslogRemote: String? = nil
    ) -> AthenaConfig {
        AthenaConfig(
            listenHost: "127.0.0.1", listenPort: 7447, budgetBytes: budget,
            engine: "mlx", model: model, dataDir: dataDir,
            logLevel: logLevel, syslogRemote: syslogRemote,
            logDir: "/var/log/athena")
    }

    func testDefaultProgramArguments() {
        let d = LaunchdPlist.dictionary(
            label: "me.goodolclint.athena",
            executablePath: "/usr/local/libexec/athena/athena",
            user: "svc", workingDirectory: "/usr/local/var/athena",
            config: cfg())
        XCTAssertEqual(
            d["ProgramArguments"] as? [String],
            [
                "/usr/local/libexec/athena/athena", "load",
                "--host", "127.0.0.1", "--port", "7447", "--engine", "mlx",
            ])
        XCTAssertEqual(d["RunAtLoad"] as? Bool, true)
        XCTAssertEqual(d["Label"] as? String, "me.goodolclint.athena")
    }

    func testFullProgramArgumentsOrder() {
        let d = LaunchdPlist.dictionary(
            label: "l", executablePath: "/bin/athena", user: "svc",
            workingDirectory: "/w",
            config: cfg(
                budget: 36_000_000_000, model: "M", dataDir: "/srv/a",
                logLevel: "debug",
                syslogRemote: "udp://10.0.0.5:514"))
        XCTAssertEqual(
            d["ProgramArguments"] as? [String],
            [
                "/bin/athena", "load", "--host", "127.0.0.1",
                "--port", "7447", "--budget-bytes", "36000000000",
                "--engine", "mlx", "--model", "M",
                "--data-dir", "/srv/a", "--log-level", "debug",
                "--syslog-remote", "udp://10.0.0.5:514",
            ])
    }

    func testXmlDataRoundTripsAsValidPlist() throws {
        let data = try LaunchdPlist.xmlData(
            label: "l", executablePath: "/bin/athena", user: "svc",
            workingDirectory: "/w", config: cfg(budget: 1))
        let back = try PropertyListSerialization.propertyList(
            from: data, options: [], format: nil) as? [String: Any]
        XCTAssertEqual(back?["Label"] as? String, "l")
        XCTAssertEqual(
            (back?["ProgramArguments"] as? [String])?.contains("--budget-bytes"),
            true)
    }
}

final class InstallPlanTests: XCTestCase {

    func testPathsUnderPrefix() {
        let plan = InstallPlan(
            sourceDir: URL(fileURLWithPath: "/src"),
            prefix: URL(fileURLWithPath: "/opt/athena"),
            label: "me.goodolclint.athena")
        XCTAssertEqual(
            plan.installedBinary.path, "/opt/athena/libexec/athena/athena")
        XCTAssertEqual(plan.binSymlink.path, "/opt/athena/bin/athena")
        XCTAssertEqual(
            plan.installedConfig.path, "/opt/athena/etc/athena/athena.toml")
        XCTAssertEqual(
            plan.plistPath.path,
            "/Library/LaunchDaemons/me.goodolclint.athena.plist")
    }

    func testArtifactNamesAreBinaryPlusBundles() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("athena-plan-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        try Data().write(to: dir.appendingPathComponent("athena"))
        try Data().write(to: dir.appendingPathComponent("athena.o"))
        for b in ["mlx-swift_Cmlx.bundle", "swift-nio_NIOPosix.bundle"] {
            try FileManager.default.createDirectory(
                at: dir.appendingPathComponent(b),
                withIntermediateDirectories: true)
        }
        let plan = InstallPlan(
            sourceDir: dir, prefix: URL(fileURLWithPath: "/usr/local"),
            label: "l")
        XCTAssertEqual(
            plan.artifactNames(),
            ["athena", "mlx-swift_Cmlx.bundle", "swift-nio_NIOPosix.bundle"])
    }
}
