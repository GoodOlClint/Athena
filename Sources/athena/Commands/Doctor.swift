import ArgumentParser
import AthenaClient
import AthenaCore
import AthenaDeploy
import AthenaLLM
import AthenaStore
import Foundation
import NIOSSL

/// `athena doctor` — read-only environment preflight. Surfaces the
/// failure modes this appliance keeps hitting: SSD model store not
/// mounted, a `swift build` binary with no Metal lib (MLX aborts at
/// first inference), unparseable config, unresolvable default model,
/// unwritable data/log dirs. Exit non-zero if any check FAILs. M9.6a.
struct Doctor: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "doctor",
        abstract: "Diagnose the runtime environment (read-only).")

    @Option(help: "Config file (overrides auto-resolution).")
    var config: String?
    @Option(help: "Model store root. Default: ~/.athena/models.")
    var modelStore: String?

    private enum Level: String { case ok = "ok  ", warn = "warn", fail = "FAIL" }

    func run() async throws {
        var fails = 0
        func say(_ l: Level, _ msg: String) {
            if l == .fail { fails += 1 }
            print("\(l.rawValue)  \(msg)")
        }
        let fm = FileManager.default

        // 1. Metal library next to THIS binary — absent ⇒ a `swift
        //    build` binary that aborts at first MLX inference.
        let binDir = Bundle.main.executableURL!
            .resolvingSymlinksInPath().deletingLastPathComponent()
        let metallib = InstallPlan(
            sourceDir: binDir,
            prefix: URL(fileURLWithPath: "/usr/local"),
            label: "x"
        ).sourceMetallib
        if fm.fileExists(atPath: metallib.path) {
            say(.ok, "Metal library present (MLX can run)")
        } else {
            say(
                .fail,
                "no Metal library at \(metallib.path) — this binary "
                    + "cannot run MLX (build via deploy/build.sh)")
        }

        // 2. xcodebuild — only needed to build/convert-from-source,
        //    not to run an installed binary.
        if fm.isExecutableFile(atPath: "/usr/bin/xcodebuild") {
            say(.ok, "xcodebuild present")
        } else {
            say(.warn, "xcodebuild not found (needed only to rebuild)")
        }

        // 3. Model store root — the recurring SSD-detach failure.
        let root =
            modelStore.map { URL(fileURLWithPath: $0, isDirectory: true) }
            ?? ModelStore.defaultRoot
        var isDir: ObjCBool = false
        if fm.fileExists(atPath: root.path, isDirectory: &isDir),
            isDir.boolValue
        {
            say(.ok, "model store present: \(root.path)")
        } else {
            say(
                .fail,
                "model store missing: \(root.path) — pull/convert a "
                    + "model, set model_store, or mount the volume "
                    + "if it lives on one")
        }

        // 4. Config resolve + parse.
        let cfgURL = ConfigEditor.resolvePath(config)
        var parsed: AthenaConfig?
        if !fm.fileExists(atPath: cfgURL.path) {
            say(.warn, "no config at \(cfgURL.path) (using defaults)")
        } else {
            do {
                parsed = try AthenaConfig.parse(file: cfgURL)
                say(.ok, "config parses: \(cfgURL.path)")
            } catch {
                say(.fail, "config unparseable (\(cfgURL.path)): \(error)")
            }
        }

        // 5. Default model resolvable + healthy.
        let store = ModelStore(rootDirectory: root)
        let modelURL = store.resolve(parsed?.model)
        if !fm.fileExists(atPath: modelURL.path) {
            say(
                .warn,
                "default model not present: \(modelURL.path) "
                    + "(pull/convert before serving)")
        } else {
            let problems = ModelHealth.check(modelURL)
            if problems.isEmpty {
                say(.ok, "default model ok: \(modelURL.lastPathComponent)")
            } else {
                say(
                    .fail,
                    "default model unhealthy "
                        + "(\(modelURL.lastPathComponent)): "
                        + problems.joined(separator: "; "))
            }
        }

        // 5b. Dangling / broken store entries (M69 operability). A
        //     `pull`-created symlink whose HF-cache target was pruned (or any
        //     entry missing config.json / *.safetensors) sits in the store but
        //     500s only at REQUEST time — never at startup, `ls`, or `show`.
        //     Surface it here so an operator can re-pull or remove it.
        if isDir.boolValue {
            let broken = ModelStoreOps.brokenEntries(root: root)
            if broken.isEmpty {
                say(.ok, "store entries all resolve (no dangling symlinks)")
            } else {
                for b in broken {
                    say(
                        .warn,
                        "store entry '\(b.name)' won't load: "
                            + b.problems.joined(separator: "; ")
                            + " — re-pull or `athena rm \(b.name)`")
                }
            }
        }

        // 6. Data dir writable (store + queue live here).
        let dataDir =
            parsed?.dataDir.map {
                URL(
                    fileURLWithPath:
                        ($0 as NSString).expandingTildeInPath,
                    isDirectory: true)
            }
            ?? AthenaEnv.userHome()
                .appendingPathComponent(".athena", isDirectory: true)
        if writable(dataDir) {
            say(.ok, "data dir writable: \(dataDir.path)")
        } else {
            say(.fail, "data dir not writable: \(dataDir.path)")
        }

        // 7. Log dir (config-driven; advisory). M45.1: this dir now
        //    holds ONLY the launchd StandardErrorPath crash-dump
        //    (athena.err.log) — not a diagnostic log. The operator
        //    diagnostic surface is the macOS unified log (`athena
        //    logs` / `log show --predicate 'subsystem == "athena"'`;
        //    see docs/logging.md).
        if let log = parsed?.logDir {
            if fm.fileExists(atPath: log) {
                say(.ok,
                    "log dir present: \(log) (crash-dump capture only; "
                        + "diagnostic surface is the macOS unified log "
                        + "— `athena logs`)")
            } else {
                say(.warn,
                    "log dir missing: \(log) (created on install; "
                        + "crash-dump capture only)")
            }
        }

        // 8. Daemon reachability (informational). Probe HTTPS when TLS
        //    is configured so the scheme matches what the daemon serves.
        let host = parsed?.listenHost ?? "127.0.0.1"
        let port = parsed?.listenPort ?? GovernorConfig.defaultPort
        let tlsOn = parsed?.tlsCert != nil && parsed?.tlsKey != nil
        let scheme = tlsOn ? "https" : "http"
        var req = URLRequest(
            url: URL(string: "\(scheme)://\(host):\(port)/healthz")!)
        req.timeoutInterval = 2
        if (try? await URLSession.shared.data(for: req)) != nil {
            say(.ok, "daemon responding at \(scheme)://\(host):\(port)")
        } else {
            say(
                .warn,
                "daemon not responding at \(scheme)://\(host):\(port)"
                    + (tlsOn
                        ? " (or its TLS cert isn't system-trusted)" : ""))
        }

        // 9. Auth posture (mirrors the daemon's fail-safe gate so
        //    this predicts startup).
        let env = ProcessInfo.processInfo.environment
        let envKeys =
            !(env["ATHENA_ADMIN_KEYS"] ?? "").isEmpty
            || !(env["ATHENA_INFERENCE_KEYS"] ?? "").isEmpty
        let fileKeys =
            (parsed?.authKeysFile).map {
                fm.fileExists(
                    atPath: ($0 as NSString).expandingTildeInPath)
            } ?? false
        var nTok = 0
        var nUsr = 0
        var nAdmins = 0
        // ADR 025 S4 — only OPEN the store if it already exists on disk; a
        // stateless-loopback daemon writes none, and doctor must not be the
        // thing that creates a phantom `athena.sqlite` (`SQLITE_OPEN_CREATE`).
        let dctStoreFile = dataDir.appendingPathComponent("athena.sqlite")
        if fm.fileExists(atPath: dctStoreFile.path),
            let db = try? AthenaStore(
                path: dctStoreFile,
                key: StoreKey.resolve(trustEnv: geteuid() != 0))
        {
            nTok = await db.tokenCount()
            nUsr = await db.userCount()
            nAdmins = (try? await db.usersWithRole("admin"))?.count ?? 0
        }
        let anyCreds = envKeys || fileKeys || nTok > 0 || nUsr > 0
        let loopback: Set<String> = [
            "127.0.0.1", "::1", "localhost",
        ]
        if anyCreds {
            var src: [String] = []
            if nTok > 0 { src.append("\(nTok) token(s)") }
            if nUsr > 0 {
                src.append(
                    "\(nUsr) user(s)/\(nAdmins) admin")
            }
            if envKeys { src.append("env(admin)") }
            if fileKeys { src.append("file") }
            say(
                .ok,
                "auth: enabled (RBAC) — "
                    + src.joined(separator: ", "))
            // A DB with users but no admin role and no env/file
            // admin key ⇒ no one can administer the appliance.
            if nUsr > 0, nAdmins == 0, !envKeys, !fileKeys {
                say(
                    .warn,
                    "RBAC: users exist but NO admin — grant the "
                        + "admin role (`athena auth role grant "
                        + "<user> admin`)")
            }
        } else if loopback.contains(host) {
            say(
                .warn,
                "auth: disabled — loopback open (dev). Add "
                    + "credentials before exposing.")
        } else {
            say(
                .fail,
                "auth: disabled and listen=\(host) is non-loopback "
                    + "— daemon will REFUSE to start")
        }

        // 10. TLS posture (predicts the daemon's HTTPS / fail-closed
        //     behavior; mirrors AthenaServer.serverBuilder).
        let tlsCert = parsed?.tlsCert.map {
            ($0 as NSString).expandingTildeInPath
        }
        let tlsKey = parsed?.tlsKey.map {
            ($0 as NSString).expandingTildeInPath
        }
        switch (tlsCert, tlsKey) {
        case (nil, nil):
            if loopback.contains(host) {
                say(
                    .ok,
                    "TLS: disabled — loopback bind (plaintext stays "
                        + "on this host)")
            } else {
                say(
                    .warn,
                    "TLS: disabled on non-loopback \(host) — bearer "
                        + "tokens + the WebUI cookie travel in "
                        + "plaintext. Set tls_cert/tls_key or front "
                        + "the daemon with a TLS reverse proxy "
                        + "(docs/reverse-proxy.md).")
            }
        case (.some, nil), (nil, .some):
            say(
                .fail,
                "TLS: only one of tls_cert/tls_key set — the daemon "
                    + "will REFUSE to start (set both, or neither)")
        case (let cert?, let key?):
            var tlsOK = true
            if !fm.isReadableFile(atPath: cert) {
                say(.fail, "TLS: cert not readable: \(cert)")
                tlsOK = false
            }
            if !fm.isReadableFile(atPath: key) {
                say(.fail, "TLS: key not readable: \(key)")
                tlsOK = false
            } else if let mode = (try? fm.attributesOfItem(
                atPath: key))?[.posixPermissions] as? NSNumber,
                mode.intValue & 0o077 != 0
            {
                say(
                    .warn,
                    "TLS: private key \(key) is group/other-"
                        + "accessible (chmod 600 recommended)")
            }
            if tlsOK {
                if let leaf = try? NIOSSLCertificate.fromPEMFile(cert)
                    .first
                {
                    let expiry = Date(
                        timeIntervalSince1970:
                            TimeInterval(leaf.notValidAfter))
                    let days = Int(
                        expiry.timeIntervalSinceNow / 86400)
                    if days < 0 {
                        say(
                            .fail,
                            "TLS: certificate EXPIRED "
                                + "\(-days) day(s) ago — renew before "
                                + "serving")
                    } else if days < 14 {
                        say(
                            .warn,
                            "TLS: enabled — cert expires in "
                                + "\(days) day(s); renew soon")
                    } else {
                        say(
                            .ok,
                            "TLS: enabled — cert valid "
                                + "\(days) more day(s)")
                    }
                } else {
                    say(
                        .fail,
                        "TLS: \(cert) is not valid PEM — the daemon "
                            + "would refuse to start")
                }
            }
        }

        // 11. HF token posture (informational — only gated/private
        // model fetches need it; public repos work without).
        say(.ok, "hf token: \(HFAuth.source())")

        // 12. Abuse-protection posture (rate limit + concurrency caps;
        //     M29). Both are keyed by the auth principal and enforced
        //     ONLY when auth is enabled — so a config'd limit on an
        //     auth-off bind does nothing, and an authed non-loopback
        //     bind with NO limit can be flooded by a single key.
        let rate = parsed?.rateLimit.flatMap(Double.init) ?? 0
        let burst = parsed?.rateBurst ?? 0
        let gConc = parsed?.maxConcurrency ?? 0
        let pConc = parsed?.maxConcurrencyPerPrincipal ?? 0
        let rateOn = rate > 0
        let concOn = gConc > 0 || pConc > 0
        if rateOn {
            let b = burst > 0 ? "\(burst)" : "auto"
            say(
                .ok,
                "rate limiting: \(rate)/s per principal (burst \(b))")
        }
        if concOn {
            var parts: [String] = []
            if gConc > 0 { parts.append("global \(gConc)") }
            if pConc > 0 { parts.append("per-principal \(pConc)") }
            say(
                .ok,
                "concurrency caps: " + parts.joined(separator: ", "))
        }
        if rateOn || concOn, !anyCreds {
            say(
                .warn,
                "rate/concurrency limits are configured but auth is "
                    + "disabled — they are NOT enforced on an auth-off "
                    + "bind (only authenticated callers are throttled)")
        } else if !rateOn, !concOn {
            if anyCreds, !loopback.contains(host) {
                say(
                    .warn,
                    "abuse protection: no rate_limit or "
                        + "max_concurrency on non-loopback \(host) — one "
                        + "key can flood the sync path. Set rate_limit "
                        + "and/or max_concurrency.")
            } else {
                say(
                    .ok,
                    "abuse protection: none configured "
                        + "(rate/concurrency limits off)")
            }
        }

        // 12b. Prompt-prefix cache posture (M59). Off by default; when on,
        //      report the scope (the secure default is per-principal) and
        //      warn if it is enabled on an auth-off, non-loopback bind where
        //      every caller collapses to the "anon" principal and would
        //      therefore share cached prefixes.
        let promptCacheOn =
            (ProcessInfo.processInfo.environment["ATHENA_PROMPT_CACHE"]
                .map { $0 == "1" || $0.lowercased() == "true" })
            ?? (parsed?.promptCacheEnabled ?? false)
        if promptCacheOn {
            let scope = parsed?.promptCacheScope ?? "principal"
            let entries = parsed?.promptCacheMaxEntries ?? 4
            say(
                .ok,
                "prompt cache: ON (scope=\(scope), max_entries=\(entries)) "
                    + "— MTP path, bit-identical reuse")
            if !anyCreds, !loopback.contains(host) {
                say(
                    .warn,
                    "prompt cache is ON with auth disabled on non-loopback "
                        + "\(host): every caller is the same 'anon' principal, "
                        + "so cached prefixes are shared across callers. Enable "
                        + "auth, or bind loopback / behind a proxy.")
            }
        } else {
            say(.ok, "prompt cache: off")
        }

        // 13. Data-at-rest posture (M34/ADR 025): store-persistence mode +
        //     SQLite encryption (or the FileVault fallback). Passwords
        //     (PBKDF2), token hashes (SHA-256) and outbound secrets (Keychain)
        //     are already safe; after ADR 025 the store carries no request
        //     content (queue + vector tenants gone) — only auth/audit/usage.
        let storeFile = dataDir.appendingPathComponent("athena.sqlite")
        let storeExists = fm.fileExists(atPath: storeFile.path)

        // ADR 025 S4 — report whether the daemon persists the store at all.
        let storeMode = StoreMode.resolve(
            hasBootstrapKeys: envKeys || fileKeys,
            dbFileExists: storeExists,
            isLoopback: StoreMode.isLoopback(host),
            encryptStore: parsed?.encryptStore == true,
            persistOverride: parsed?.persistStore == true)
        if storeMode == .ephemeral {
            // Nothing on disk to encrypt or bound — the at-rest checks are
            // moot, but doctor still reports the audit/usage posture below.
            say(
                .ok,
                "store: STATELESS (loopback, no credentials) — the daemon "
                    + "creates no athena.sqlite; audit/usage live in memory "
                    + "only. Zero request-related data at rest.")
        } else {
            say(
                .ok,
                "store: persistent at \(storeFile.path)"
                    + (parsed?.persistStore == true
                        ? " (persist_store forces on-disk even in loopback)"
                        : ""))
            let onDiskEncrypted =
                storeExists && !AthenaStore.isPlaintextDatabase(at: storeFile)
            if parsed?.encryptStore == true {
                let keySrc = StoreKey.source(trustEnv: geteuid() != 0)
                if !storeExists {
                    say(
                        .ok,
                        "at-rest: encryption enabled (SQLCipher) — store not "
                            + "yet created; key from \(keySrc)")
                } else if onDiskEncrypted {
                    say(
                        .ok,
                        "at-rest: store encrypted (SQLCipher); key from "
                            + "\(keySrc)")
                } else {
                    say(
                        .warn,
                        "at-rest: encrypt_store is set but the store is still "
                            + "plaintext — it migrates on the next daemon "
                            + "start (key from \(keySrc))")
                }
                if keySrc == "none" {
                    say(
                        .warn,
                        "at-rest: no key resolvable yet — the daemon will mint "
                            + "one in the Keychain on start. Back up "
                            + "ATHENA_STORE_KEY / the Keychain item; without "
                            + "it an encrypted store is unrecoverable.")
                }
            } else if onDiskEncrypted {
                say(
                    .warn,
                    "at-rest: the store on disk is encrypted but encrypt_store "
                        + "is not set — set encrypt_store (or ATHENA_STORE_KEY) "
                        + "so the daemon can open it")
            } else {
                switch Self.fileVaultOn() {
                case .some(true):
                    say(
                        .ok,
                        "at-rest: store is plaintext but FileVault is ON "
                            + "(full-disk encrypted). Set encrypt_store for "
                            + "defense-in-depth.")
                case .some(false):
                    say(
                        .warn,
                        "at-rest: store is plaintext and FileVault is OFF — "
                            + "credential hashes and audit/usage metadata are "
                            + "readable on disk. Enable FileVault or set "
                            + "encrypt_store.")
                case .none:
                    say(
                        .warn,
                        "at-rest: store is plaintext; FileVault status unknown "
                            + "— ensure FileVault is on, or set encrypt_store.")
                }
            }
        }
        // ADR 025 S2 — the async queue (and its result-retention knobs)
        // was removed, so there is no request-content retention to report;
        // the store no longer persists any inference inputs/outputs.

        // 14. Audit-log posture (M30): RBAC/admin mutations (user/role/
        //     token CRUD, model.remove, default_set, daemon load/unload)
        //     append to an audit_log table. Report how many records exist
        //     and the retention policy. Unlike the queue/vector blobs
        //     above — where unbounded GROWTH is the risk — an audit trail
        //     is normally kept for compliance, so it's a day-based
        //     retention (which PRUNES the trail) that's worth surfacing.
        var nAudit = 0
        // ADR 025 S4 — don't create a phantom store just to count audit rows;
        // a stateless run has none on disk.
        if storeExists,
            let db = try? AthenaStore(
                path: storeFile,
                key: StoreKey.resolve(trustEnv: geteuid() != 0))
        {
            nAudit = await db.auditCount()
        }
        if let days = parsed?.auditRetentionDays, days > 0 {
            say(
                .ok,
                "audit: \(nAudit) record(s); retention \(days) day(s) "
                    + "— older RBAC/admin records are pruned. Set "
                    + "audit_retention_days = 0 to keep the trail "
                    + "indefinitely for compliance.")
        } else {
            say(
                .ok,
                "audit: \(nAudit) record(s); retention unbounded "
                    + "(records kept indefinitely)")
        }

        // 15. Token-expiry posture (M36): how many managed tokens carry
        //     a TTL, how many have already expired (rejected at
        //     validation but still on disk), and whether a global
        //     token_max_age_days cap bounds the never-expiring ones.
        var tokTotal = 0
        var tokExpiring = 0
        var tokExpired = 0
        if let db = try? AthenaStore(
            path: storeFile, key: StoreKey.resolve(trustEnv: geteuid() != 0))
        {
            let now = Date().timeIntervalSince1970
            for t in await db.listTokens() {
                tokTotal += 1
                if let e = t.expires {
                    tokExpiring += 1
                    if e <= now { tokExpired += 1 }
                }
            }
        }
        let tokCap = parsed?.tokenMaxAgeDays ?? 0
        if tokTotal == 0 {
            say(
                .ok,
                "tokens: no managed tokens (env/file keys or auth off)")
        } else {
            let capNote =
                tokCap > 0 ? "global cap \(tokCap)d" : "no global cap"
            say(
                .ok,
                "tokens: \(tokTotal) managed, \(tokExpiring) with a TTL, "
                    + capNote)
            if tokExpired > 0 {
                say(
                    .warn,
                    "tokens: \(tokExpired) already expired but still on "
                        + "disk — rejected (401) yet lingering; prune "
                        + "with `athena auth rm <prefix>`")
            }
            let neverExpire = tokTotal - tokExpiring
            if neverExpire > 0, tokCap == 0, anyCreds,
                !loopback.contains(host)
            {
                say(
                    .warn,
                    "tokens: \(neverExpire) never expire and no "
                        + "token_max_age_days cap on a non-loopback bind "
                        + "— long-lived secrets. Set token_max_age_days "
                        + "or mint with --ttl.")
            }
        }

        // 16. CLI-only options NOT reflected in the TOML (M43.4 #10).
        //     `--prompt-cache-cap-bytes` has no `athena.toml` key (the
        //     plist still encodes it at install time, so editing the TOML
        //     for "the same thing" is a silent no-op). The per-module
        //     `--*-model` flags now DO map to TOML keys (ADR 026:
        //     `model`/`embedding_model`/… via `athena default --module M`).
        let cliOnly: [(flag: String, note: String)] = [
            (
                "--prompt-cache-cap-bytes",
                "set on `athena load`; no TOML key (governor cache "
                    + "ceiling)"
            ),
        ]
        say(
            .ok,
            "TOML reflects every `athena load` flag EXCEPT the "
                + "following CLI-only knobs:")
        for entry in cliOnly {
            print("        \(entry.flag)")
            print("          \(entry.note)")
        }

        if fails > 0 {
            print("\n\(fails) critical issue(s).")
            throw ExitCode.failure
        }
        print("\nall critical checks passed.")
    }

    /// macOS FileVault state via `fdesetup status` (read-only, no sudo for
    /// status). nil ⇒ couldn't determine. Backs the at-rest posture check.
    private static func fileVaultOn() -> Bool? {
        let p = Process()
        p.executableURL = URL(fileURLWithPath: "/usr/bin/fdesetup")
        p.arguments = ["status"]
        let out = Pipe()
        p.standardOutput = out
        p.standardError = FileHandle.nullDevice
        do { try p.run() } catch { return nil }
        p.waitUntilExit()
        let data = out.fileHandleForReading.readDataToEndOfFile()
        guard let s = String(data: data, encoding: .utf8) else { return nil }
        if s.contains("FileVault is On") { return true }
        if s.contains("FileVault is Off") { return false }
        return nil
    }

    /// Writable = exists & we can create a probe file, or the parent
    /// exists & is writable (dir will be created on first use).
    private func writable(_ dir: URL) -> Bool {
        let fm = FileManager.default
        if fm.fileExists(atPath: dir.path) {
            let probe = dir.appendingPathComponent(".athena-doctor")
            if (try? Data().write(to: probe)) != nil {
                try? fm.removeItem(at: probe)
                return true
            }
            return false
        }
        return fm.isWritableFile(
            atPath: dir.deletingLastPathComponent().path)
    }
}
