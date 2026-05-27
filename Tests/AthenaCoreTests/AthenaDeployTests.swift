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

    func testNetworkProxyKeys() throws {
        let toml = """
            listen_host = "127.0.0.1"
            listen_port = 7447
            log_dir = "/l"
            https_proxy = "http://proxy.corp:3128"
            # http_proxy = "http://nope:1"
            all_proxy = "socks5://s:1080"
            no_proxy = "127.0.0.1,localhost,.internal"
            """
        let c = try AthenaConfig.parse(toml: toml)
        XCTAssertEqual(c.httpsProxy, "http://proxy.corp:3128")
        XCTAssertNil(c.httpProxy)  // commented ⇒ nil
        XCTAssertEqual(c.allProxy, "socks5://s:1080")
        XCTAssertEqual(c.noProxy, "127.0.0.1,localhost,.internal")
    }

    func testTLSKeysParse() throws {
        let toml = """
            listen_host = "127.0.0.1"
            listen_port = 7447
            log_dir = "/l"
            tls_cert = "/etc/athena/tls/fullchain.pem"
            tls_key = "/etc/athena/tls/privkey.pem"
            """
        let c = try AthenaConfig.parse(toml: toml)
        XCTAssertEqual(c.tlsCert, "/etc/athena/tls/fullchain.pem")
        XCTAssertEqual(c.tlsKey, "/etc/athena/tls/privkey.pem")
    }

    func testTLSKeysAbsentAreNil() throws {
        let toml = """
            listen_host = "127.0.0.1"
            listen_port = 7447
            log_dir = "/l"
            # tls_cert = "/etc/athena/tls/fullchain.pem"
            """
        let c = try AthenaConfig.parse(toml: toml)
        XCTAssertNil(c.tlsCert)
        XCTAssertNil(c.tlsKey)
    }

    func testRateLimitKeysParse() throws {
        let toml = """
            listen_host = "127.0.0.1"
            listen_port = 7447
            log_dir = "/l"
            rate_limit = 10
            rate_burst = 20
            """
        let c = try AthenaConfig.parse(toml: toml)
        XCTAssertEqual(c.rateLimit, "10")
        XCTAssertEqual(c.rateBurst, 20)
    }

    func testRateLimitKeysAbsentAreNil() throws {
        let toml = """
            listen_host = "127.0.0.1"
            listen_port = 7447
            log_dir = "/l"
            """
        let c = try AthenaConfig.parse(toml: toml)
        XCTAssertNil(c.rateLimit)
        XCTAssertNil(c.rateBurst)
    }

    func testConcurrencyKeysParse() throws {
        let toml = """
            listen_host = "127.0.0.1"
            listen_port = 7447
            log_dir = "/l"
            max_concurrency = 8
            max_concurrency_per_principal = 2
            """
        let c = try AthenaConfig.parse(toml: toml)
        XCTAssertEqual(c.maxConcurrency, 8)
        XCTAssertEqual(c.maxConcurrencyPerPrincipal, 2)
    }

    func testAuditRetentionKeyParse() throws {
        let toml = """
            listen_host = "127.0.0.1"
            listen_port = 7447
            log_dir = "/l"
            audit_retention_days = 365
            """
        let c = try AthenaConfig.parse(toml: toml)
        XCTAssertEqual(c.auditRetentionDays, 365)
    }

    func testAuditRetentionKeyAbsentIsNil() throws {
        let toml = """
            listen_host = "127.0.0.1"
            listen_port = 7447
            log_dir = "/l"
            """
        let c = try AthenaConfig.parse(toml: toml)
        XCTAssertNil(c.auditRetentionDays)
    }

    func testTokenMaxAgeKeyParse() throws {
        let toml = """
            listen_host = "127.0.0.1"
            listen_port = 7447
            log_dir = "/l"
            token_max_age_days = 90
            """
        let c = try AthenaConfig.parse(toml: toml)
        XCTAssertEqual(c.tokenMaxAgeDays, 90)
    }

    func testTokenMaxAgeKeyAbsentIsNil() throws {
        let toml = """
            listen_host = "127.0.0.1"
            listen_port = 7447
            log_dir = "/l"
            """
        let c = try AthenaConfig.parse(toml: toml)
        XCTAssertNil(c.tokenMaxAgeDays)
    }

    func testConcurrencyKeysAbsentAreNil() throws {
        let toml = """
            listen_host = "127.0.0.1"
            listen_port = 7447
            log_dir = "/l"
            """
        let c = try AthenaConfig.parse(toml: toml)
        XCTAssertNil(c.maxConcurrency)
        XCTAssertNil(c.maxConcurrencyPerPrincipal)
    }

    func testRequestTimeoutKeyParse() throws {
        let toml = """
            listen_host = "127.0.0.1"
            listen_port = 7447
            log_dir = "/l"
            request_timeout_secs = 120
            """
        let c = try AthenaConfig.parse(toml: toml)
        XCTAssertEqual(c.requestTimeoutSecs, 120)
    }

    func testRequestTimeoutKeyAbsentIsNil() throws {
        let toml = """
            listen_host = "127.0.0.1"
            listen_port = 7447
            log_dir = "/l"
            """
        let c = try AthenaConfig.parse(toml: toml)
        XCTAssertNil(c.requestTimeoutSecs)
    }

    func testPreloadKeyParse() throws {
        let toml = """
            listen_host = "127.0.0.1"
            listen_port = 7447
            log_dir = "/l"
            preload = true
            """
        let c = try AthenaConfig.parse(toml: toml)
        XCTAssertEqual(c.preload, true)
    }

    func testPreloadKeyAbsentIsNil() throws {
        let toml = """
            listen_host = "127.0.0.1"
            listen_port = 7447
            log_dir = "/l"
            """
        let c = try AthenaConfig.parse(toml: toml)
        XCTAssertNil(c.preload)
    }

    func testQueueRetentionKeysParse() throws {
        let toml = """
            listen_host = "127.0.0.1"
            listen_port = 7447
            log_dir = "/l"
            queue_result_ttl_secs = 604800
            queue_max_rows = 10000
            """
        let c = try AthenaConfig.parse(toml: toml)
        XCTAssertEqual(c.queueResultTtlSecs, 604_800)
        XCTAssertEqual(c.queueMaxRows, 10_000)
    }

    func testQueueRetentionKeysAbsentAreNil() throws {
        let toml = """
            listen_host = "127.0.0.1"
            listen_port = 7447
            log_dir = "/l"
            """
        let c = try AthenaConfig.parse(toml: toml)
        XCTAssertNil(c.queueResultTtlSecs)
        XCTAssertNil(c.queueMaxRows)
    }

    func testVectorTtlAndContentOptOutKeysParse() throws {
        let toml = """
            listen_host = "127.0.0.1"
            listen_port = 7447
            log_dir = "/l"
            vector_ttl_secs = 2592000
            drop_request_content = true
            """
        let c = try AthenaConfig.parse(toml: toml)
        XCTAssertEqual(c.vectorTtlSecs, 2_592_000)
        XCTAssertEqual(c.dropRequestContent, true)
    }

    func testVectorTtlAndContentOptOutKeysAbsentAreNil() throws {
        let toml = """
            listen_host = "127.0.0.1"
            listen_port = 7447
            log_dir = "/l"
            """
        let c = try AthenaConfig.parse(toml: toml)
        XCTAssertNil(c.vectorTtlSecs)
        XCTAssertNil(c.dropRequestContent)
    }

    func testEncryptStoreKeyParse() throws {
        let toml = """
            listen_host = "127.0.0.1"
            listen_port = 7447
            log_dir = "/l"
            encrypt_store = true
            """
        let c = try AthenaConfig.parse(toml: toml)
        XCTAssertEqual(c.encryptStore, true)
    }

    func testEncryptStoreKeyAbsentIsNil() throws {
        let toml = """
            listen_host = "127.0.0.1"
            listen_port = 7447
            log_dir = "/l"
            """
        let c = try AthenaConfig.parse(toml: toml)
        XCTAssertNil(c.encryptStore)
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

final class DefaultConfigTests: XCTestCase {

    func testSynthesizedConfigRoundTrips() throws {
        // The synthesizer's whole contract: its output must parse, with
        // the supplied required keys present and the optional keys left
        // commented (⇒ nil ⇒ the daemon's built-in defaults apply).
        let toml = DefaultConfig.toml(
            listenPort: 7447, logDir: "/usr/local/var/log/athena")
        let c = try AthenaConfig.parse(toml: toml)
        XCTAssertEqual(c.listenHost, "127.0.0.1")
        XCTAssertEqual(c.listenPort, 7447)
        XCTAssertEqual(c.engine, "mlx")
        XCTAssertEqual(c.logDir, "/usr/local/var/log/athena")
        // Optionals stay commented ⇒ built-in defaults at runtime.
        XCTAssertNil(c.budgetBytes)
        XCTAssertNil(c.model)
        XCTAssertNil(c.modelStore)
        XCTAssertNil(c.dataDir)
        XCTAssertNil(c.maxTokens)
        XCTAssertNil(c.temperature)
        XCTAssertNil(c.speculative)
        XCTAssertNil(c.kvCompression)
        XCTAssertNil(c.authKeysFile)
        XCTAssertNil(c.httpsProxy)
    }

    func testSynthesizedConfigHonorsParameters() throws {
        let toml = DefaultConfig.toml(
            listenHost: "0.0.0.0", listenPort: 9000, engine: "stub",
            logDir: "/var/log/x")
        let c = try AthenaConfig.parse(toml: toml)
        XCTAssertEqual(c.listenHost, "0.0.0.0")
        XCTAssertEqual(c.listenPort, 9000)
        XCTAssertEqual(c.engine, "stub")
        XCTAssertEqual(c.logDir, "/var/log/x")
    }
}

final class LaunchdPlistTests: XCTestCase {

    private func cfg(
        budget: Int? = nil, model: String? = nil,
        modelStore: String? = nil, dataDir: String? = nil,
        logLevel: String? = nil,
        maxTokens: Int? = nil, temperature: String? = nil,
        speculative: Bool? = nil, vectorCapBytes: Int? = nil,
        authKeysFile: String? = nil,
        tlsCert: String? = nil, tlsKey: String? = nil,
        rateLimit: String? = nil, rateBurst: Int? = nil,
        maxConcurrency: Int? = nil,
        maxConcurrencyPerPrincipal: Int? = nil,
        auditRetentionDays: Int? = nil,
        tokenMaxAgeDays: Int? = nil,
        requestTimeoutSecs: Int? = nil,
        preload: Bool? = nil,
        queueResultTtlSecs: Int? = nil,
        queueMaxRows: Int? = nil,
        vectorTtlSecs: Int? = nil,
        dropRequestContent: Bool? = nil,
        encryptStore: Bool? = nil
    ) -> AthenaConfig {
        AthenaConfig(
            listenHost: "127.0.0.1", listenPort: 7447, budgetBytes: budget,
            engine: "mlx", model: model, modelStore: modelStore,
            dataDir: dataDir, logLevel: logLevel,
            maxTokens: maxTokens,
            temperature: temperature, speculative: speculative,
            vectorCapBytes: vectorCapBytes,
            authKeysFile: authKeysFile,
            tlsCert: tlsCert, tlsKey: tlsKey,
            rateLimit: rateLimit, rateBurst: rateBurst,
            maxConcurrency: maxConcurrency,
            maxConcurrencyPerPrincipal: maxConcurrencyPerPrincipal,
            auditRetentionDays: auditRetentionDays,
            tokenMaxAgeDays: tokenMaxAgeDays,
            requestTimeoutSecs: requestTimeoutSecs,
            preload: preload,
            queueResultTtlSecs: queueResultTtlSecs,
            queueMaxRows: queueMaxRows,
            vectorTtlSecs: vectorTtlSecs,
            dropRequestContent: dropRequestContent,
            encryptStore: encryptStore,
            logDir: "/var/log/athena")
    }

    func testDefaultProgramArguments() {
        let d = LaunchdPlist.dictionary(
            label: "me.goodolclint.athena",
            executablePath: "/usr/local/libexec/athena/athenad",
            user: "svc", workingDirectory: "/usr/local/var/athena",
            config: cfg())
        XCTAssertEqual(
            d["ProgramArguments"] as? [String],
            [
                "/usr/local/libexec/athena/athenad",
                "load",
                "--background",
                "--host", "127.0.0.1", "--port", "7447",
                "--engine", "mlx",
            ])
        XCTAssertEqual(d["RunAtLoad"] as? Bool, true)
        XCTAssertEqual(d["Label"] as? String, "me.goodolclint.athena")
    }

    func testFullProgramArgumentsOrder() {
        let d = LaunchdPlist.dictionary(
            label: "l", executablePath: "/bin/athena", user: "svc",
            workingDirectory: "/w",
            config: cfg(
                budget: 36_000_000_000, model: "M",
                modelStore: "/srv/models", dataDir: "/srv/a",
                logLevel: "debug",
                maxTokens: 2048, temperature: "0.2",
                speculative: true, vectorCapBytes: 1_000_000,
                authKeysFile: "/etc/athena/auth.keys",
                tlsCert: "/etc/athena/tls/fullchain.pem",
                tlsKey: "/etc/athena/tls/privkey.pem",
                rateLimit: "10", rateBurst: 20,
                maxConcurrency: 8,
                maxConcurrencyPerPrincipal: 2,
                auditRetentionDays: 365,
                tokenMaxAgeDays: 90,
                requestTimeoutSecs: 120,
                preload: true,
                queueResultTtlSecs: 604_800,
                queueMaxRows: 10_000,
                vectorTtlSecs: 2_592_000,
                dropRequestContent: true,
                encryptStore: true))
        XCTAssertEqual(
            d["ProgramArguments"] as? [String],
            [
                "/bin/athena", "load", "--background",
                "--host", "127.0.0.1",
                "--port", "7447", "--budget-bytes", "36000000000",
                "--engine", "mlx", "--model", "M",
                "--model-store", "/srv/models",
                "--data-dir", "/srv/a", "--log-level", "debug",
                "--max-tokens", "2048", "--temperature", "0.2",
                "--speculative", "--vector-cap-bytes", "1000000",
                "--auth-keys-file", "/etc/athena/auth.keys",
                "--tls-cert", "/etc/athena/tls/fullchain.pem",
                "--tls-key", "/etc/athena/tls/privkey.pem",
                "--rate-limit", "10", "--rate-burst", "20",
                "--max-concurrency", "8",
                "--max-concurrency-per-principal", "2",
                "--audit-retention-days", "365",
                "--token-max-age-days", "90",
                "--request-timeout-secs", "120",
                "--preload",
                "--queue-result-ttl-secs", "604800",
                "--queue-max-rows", "10000",
                "--vector-ttl-secs", "2592000",
                "--drop-request-content",
                "--encrypt-store",
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
        XCTAssertEqual(
            plan.installedDaemon.path,
            "/opt/athena/libexec/athena/athenad")
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
            [
                "athena", "athenad", "mlx-swift_Cmlx.bundle",
                "swift-nio_NIOPosix.bundle",
            ])
    }
}
