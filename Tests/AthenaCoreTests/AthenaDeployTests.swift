import AthenaCore
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

    // ADR 041 A2 — token budget + window.

    func testTokenBudgetKeysParse() throws {
        let toml = """
            listen_host = "127.0.0.1"
            listen_port = 7447
            log_dir = "/l"
            token_budget = 50000000
            token_budget_window = "day"
            """
        let c = try AthenaConfig.parse(toml: toml)
        XCTAssertEqual(c.tokenBudget, 50_000_000)
        XCTAssertEqual(c.tokenBudgetWindow, "day")
        XCTAssertEqual(QuotaWindow.parse(c.tokenBudgetWindow), .day)
    }

    func testTokenBudgetKeysAbsentAreNil() throws {
        let toml = """
            listen_host = "127.0.0.1"
            listen_port = 7447
            log_dir = "/l"
            """
        let c = try AthenaConfig.parse(toml: toml)
        XCTAssertNil(c.tokenBudget)
        XCTAssertNil(c.tokenBudgetWindow)
        // Absent window still resolves to the documented default.
        XCTAssertEqual(QuotaWindow.parse(c.tokenBudgetWindow), .month)
    }

    /// An unrecognized window fails the parse LOUDLY (ADR 041 §2) — the daemon
    /// must not boot enforcing a window the operator did not choose.
    func testInvalidTokenBudgetWindowFailsParse() {
        let toml = """
            listen_host = "127.0.0.1"
            listen_port = 7447
            log_dir = "/l"
            token_budget = 100
            token_budget_window = "week"
            """
        XCTAssertThrowsError(try AthenaConfig.parse(toml: toml)) { err in
            guard
                case AthenaConfig.ParseError.invalidEnum(let key, let value, _) = err
            else { return XCTFail("expected invalidEnum, got \(err)") }
            XCTAssertEqual(key, "token_budget_window")
            XCTAssertEqual(value, "week")
        }
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

    func testUploadCapKeysParse() throws {
        // ADR 017 — both upload caps parse as positive ints.
        let toml = """
            listen_host = "127.0.0.1"
            listen_port = 7447
            log_dir = "/l"
            max_audio_upload_bytes = 104857600
            max_video_upload_bytes = 1073741824
            max_request_body_bytes = 4194304
            """
        let c = try AthenaConfig.parse(toml: toml)
        XCTAssertEqual(c.maxAudioUploadBytes, 104_857_600)
        XCTAssertEqual(c.maxVideoUploadBytes, 1_073_741_824)  // ADR 022
        XCTAssertEqual(c.maxRequestBodyBytes, 4_194_304)
    }

    func testUploadCapKeysAbsentAreNil() throws {
        let toml = """
            listen_host = "127.0.0.1"
            listen_port = 7447
            log_dir = "/l"
            """
        let c = try AthenaConfig.parse(toml: toml)
        XCTAssertNil(c.maxAudioUploadBytes)
        XCTAssertNil(c.maxVideoUploadBytes)
        XCTAssertNil(c.maxRequestBodyBytes)
    }

    func testUploadCapZeroIsRejected() throws {
        // ADR 017 — 0 (and negative) is a parse error, NOT "unlimited".
        for raw in ["0", "-1"] {
            let toml = """
                listen_host = "127.0.0.1"
                listen_port = 7447
                log_dir = "/l"
                max_audio_upload_bytes = \(raw)
                """
            XCTAssertThrowsError(
                try AthenaConfig.parse(toml: toml),
                "max_audio_upload_bytes=\(raw) must be rejected"
            ) { error in
                guard
                    case AthenaConfig.ParseError.invalidInt(let key, _) =
                        error
                else {
                    return XCTFail("expected invalidInt, got \(error)")
                }
                XCTAssertEqual(key, "max_audio_upload_bytes")
            }
        }
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

    // ADR 015 — cold_load_wait_secs parse + plist forwarding.
    func testColdLoadWaitKeyParse() throws {
        let toml = """
            listen_host = "127.0.0.1"
            listen_port = 7447
            log_dir = "/l"
            cold_load_wait_secs = 90
            """
        let c = try AthenaConfig.parse(toml: toml)
        XCTAssertEqual(c.coldLoadWaitSecs, 90)
    }

    func testColdLoadWaitKeyAbsentIsNil() throws {
        let toml = """
            listen_host = "127.0.0.1"
            listen_port = 7447
            log_dir = "/l"
            """
        let c = try AthenaConfig.parse(toml: toml)
        XCTAssertNil(c.coldLoadWaitSecs)
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

    // ADR 025 S2 — the queue (and its queue_result_ttl_secs / queue_max_rows
    // / drop_request_content retention keys) was removed, so those config
    // parse tests are gone with it.

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

    func testBoolKeysTruthyAndStrict() throws {
        // J1 (M66.4): 1/yes/on (any case) ⇒ true; 0/no/off ⇒ false.
        let toml = """
            listen_host = "h"
            listen_port = 7447
            log_dir = "/l"
            preload = 1
            encrypt_store = YES
            speculative = on
            """
        let c = try AthenaConfig.parse(toml: toml)
        XCTAssertEqual(c.preload, true)
        XCTAssertEqual(c.encryptStore, true)
        XCTAssertEqual(c.speculative, true)
    }

    func testBoolKeyInvalidValueThrows() {
        // J1: an unrecognized bool value is a ParseError, not silent false.
        let toml = """
            listen_host = "h"
            listen_port = 7447
            log_dir = "/l"
            encrypt_store = maybe
            """
        XCTAssertThrowsError(try AthenaConfig.parse(toml: toml)) {
            XCTAssertEqual(
                $0 as? AthenaConfig.ParseError,
                .invalidBool(key: "encrypt_store", value: "maybe"))
        }
    }

    func testCRLFLineEndingsParse() throws {
        // NJ1 (M66.4): CRLF must not leave a trailing \r on values.
        let toml =
            "listen_host = \"127.0.0.1\"\r\n"
            + "listen_port = 7447\r\n"
            + "log_dir = \"/l\"\r\n"
            + "encrypt_store = true\r\n"
        let c = try AthenaConfig.parse(toml: toml)
        XCTAssertEqual(c.listenHost, "127.0.0.1")
        XCTAssertEqual(c.listenPort, 7447)  // Int("7447\r") would be nil
        XCTAssertEqual(c.encryptStore, true)  // "true\r" != "true"
    }

    func testQuotedHashIsLiteralUnquotedIsComment() throws {
        // J2 (M66.4): `#` inside quotes is literal; unquoted starts a
        // comment.
        let toml = """
            listen_host = "127.0.0.1"
            listen_port = 7447
            log_dir = "/l"
            model = "weird#name"
            log_level = info  # inline comment
            """
        let c = try AthenaConfig.parse(toml: toml)
        XCTAssertEqual(c.model, "weird#name")
        XCTAssertEqual(c.logLevel, "info")
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

    // MARK: - M70.3 NJ4 — kv_compression parse coverage

    /// NJ4 — kv_compression is load-bearing (ConfigEditor reads it to drive
    /// operator-visible behavior) but only its nil-default was tested, which
    /// would pass even if `parse()` ignored the key. Feed it and assert the
    /// parsed field, so a mis-wired scalar or a renamed key fails.
    func testKvCompressionKeyParses() throws {
        let toml = """
            listen_host = "127.0.0.1"
            listen_port = 7447
            engine = "mlx"
            log_dir = "/var/log/athena"
            kv_compression = "triattention"
            """
        let c = try AthenaConfig.parse(toml: toml)
        XCTAssertEqual(c.kvCompression, "triattention")
    }

    /// The contrast case: an absent key stays nil (the built-in default applies
    /// at runtime), so the positive test isn't masking a parser that silently
    /// ignores the key.
    func testKvCompressionKeyAbsentIsNil() throws {
        let c = try AthenaConfig.parse(
            toml: """
                listen_host = "127.0.0.1"
                listen_port = 7447
                engine = "mlx"
                log_dir = "/var/log/athena"
                """)
        XCTAssertNil(c.kvCompression)
    }
}

final class LaunchdPlistTests: XCTestCase {

    private func cfg(
        budget: Int? = nil, model: String? = nil,
        modelStore: String? = nil, dataDir: String? = nil,
        logLevel: String? = nil,
        maxTokens: Int? = nil, temperature: String? = nil,
        speculative: Bool? = nil,
        authKeysFile: String? = nil,
        tlsCert: String? = nil, tlsKey: String? = nil,
        rateLimit: String? = nil, rateBurst: Int? = nil,
        maxConcurrency: Int? = nil,
        maxConcurrencyPerPrincipal: Int? = nil,
        auditRetentionDays: Int? = nil,
        tokenMaxAgeDays: Int? = nil,
        requestTimeoutSecs: Int? = nil,
        coldLoadWaitSecs: Int? = nil,
        preload: Bool? = nil,
        encryptStore: Bool? = nil
    ) -> AthenaConfig {
        AthenaConfig(
            listenHost: "127.0.0.1", listenPort: 7447, budgetBytes: budget,
            engine: "mlx", model: model, modelStore: modelStore,
            dataDir: dataDir, logLevel: logLevel,
            maxTokens: maxTokens,
            temperature: temperature, speculative: speculative,
            authKeysFile: authKeysFile,
            tlsCert: tlsCert, tlsKey: tlsKey,
            rateLimit: rateLimit, rateBurst: rateBurst,
            maxConcurrency: maxConcurrency,
            maxConcurrencyPerPrincipal: maxConcurrencyPerPrincipal,
            auditRetentionDays: auditRetentionDays,
            tokenMaxAgeDays: tokenMaxAgeDays,
            requestTimeoutSecs: requestTimeoutSecs,
            coldLoadWaitSecs: coldLoadWaitSecs,
            preload: preload,
            encryptStore: encryptStore,
            logDir: "/var/log/athena")
    }

    // ADR 037 slice 1 — the plist is STATIC: only the invariant exec line, no
    // config flags. The daemon reads the full TOML at boot via ATHENA_CONFIG.
    func testStaticProgramArguments() {
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
            ])
        XCTAssertEqual(d["RunAtLoad"] as? Bool, true)
        XCTAssertEqual(d["KeepAlive"] as? Bool, true)
        XCTAssertEqual(d["Label"] as? String, "me.goodolclint.athena")
        // NJ2: no configPath ⇒ no EnvironmentVariables key.
        XCTAssertNil(d["EnvironmentVariables"])
    }

    func testConfigPathExportsAthenaConfigEnv() {
        // NJ2 (M66.4): the prefix-correct config path is exported as
        // ATHENA_CONFIG so the daemon's TOML re-reads resolve to it.
        let d = LaunchdPlist.dictionary(
            label: "l", executablePath: "/bin/athena", user: "svc",
            workingDirectory: "/w", config: cfg(),
            configPath: "/opt/athena/etc/athena/athena.toml")
        let env = d["EnvironmentVariables"] as? [String: String]
        XCTAssertEqual(
            env?["ATHENA_CONFIG"], "/opt/athena/etc/athena/athena.toml")
    }

    // ADR 037 slice 1 — even a fully-populated config produces the SAME static
    // args: no config value is frozen into the plist anymore (that was the
    // sudo-requiring freeze). Every one of these values reaches the daemon via
    // the TOML at boot, not the plist.
    func testFullConfigStillYieldsStaticArgs() {
        let d = LaunchdPlist.dictionary(
            label: "l", executablePath: "/bin/athena", user: "svc",
            workingDirectory: "/w",
            config: cfg(
                budget: 36_000_000_000, model: "M",
                modelStore: "/srv/models", dataDir: "/srv/a",
                logLevel: "debug",
                maxTokens: 2048, temperature: "0.2",
                speculative: true,
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
                encryptStore: true))
        XCTAssertEqual(
            d["ProgramArguments"] as? [String],
            ["/bin/athena", "load", "--background"])
        // None of the frozen flags leak into the plist.
        let args = d["ProgramArguments"] as? [String] ?? []
        for flag in [
            "--budget-bytes", "--model", "--max-tokens", "--auth-keys-file",
            "--tls-cert", "--rate-limit", "--encrypt-store", "--speculative",
            "--cold-load-wait-secs", "--host", "--port", "--engine",
        ] {
            XCTAssertFalse(args.contains(flag), "frozen flag leaked: \(flag)")
        }
    }

    func testXmlDataRoundTripsAsValidPlist() throws {
        let data = try LaunchdPlist.xmlData(
            label: "l", executablePath: "/bin/athena", user: "svc",
            workingDirectory: "/w", config: cfg(budget: 1))
        let back =
            try PropertyListSerialization.propertyList(
                from: data, options: [], format: nil) as? [String: Any]
        XCTAssertEqual(back?["Label"] as? String, "l")
        // Static args survive the XML round-trip; no --budget-bytes freeze.
        XCTAssertEqual(
            back?["ProgramArguments"] as? [String],
            ["/bin/athena", "load", "--background"])
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
        // M43.3 removed the athenad shim; the test's installedDaemon
        // assertion was dropped here so swift test compiles cleanly
        // (caught while wiring M46.1's InferenceDeadlineTests additions).
        XCTAssertEqual(plan.binLauncher.path, "/opt/athena/bin/athena")
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
            // M43.3 — athenad was dropped from the artifact set when
            // the launchd path stopped going through the shim; the
            // production list is now just the binary + sorted bundles.
            [
                "athena", "mlx-swift_Cmlx.bundle",
                "swift-nio_NIOPosix.bundle",
            ])
    }
}
