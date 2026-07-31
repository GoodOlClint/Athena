import AthenaCore
import Foundation

/// Shared flat-TOML config editing used by `athena config` (M9.4b) and
/// `athena default` (M9.5d). Keeps the in-place scalar rewrite (which
/// preserves comments/layout) in exactly one place.
///
/// NB4 (M70.1b): the editing CORE (key sets, path resolution, the in-place
/// validate+rewrite `setScalarThrowing`, and `Failure`) lives here in the
/// MLX-free `AthenaDeploy` so it is unit-testable under `swift test` (ADR 008
/// follow-on). The thin CLI conveniences that `FailableExit.die` (`read`,
/// `value`, `setScalar`) stay in the `athena` executable as an extension —
/// `FailableExit` lives in the Linux-clean `AthenaClient`, which the
/// server-side `AthenaDeploy` must not depend on.
public enum ConfigEditor {
    /// String-valued keys are quoted; these two are bare ints.
    public static let intKeys: Set<String> = [
        "listen_port", "budget_bytes", "max_tokens",
        // ADR 042 — `max_prompt_tokens` was readable via `athena config get`
        // but not settable (absent from these sets ⇒ unknownKey), which is
        // exactly the knob B2 now publishes and the operator has to tune.
        "max_prompt_tokens",
        // ADR 041 — per-period token budget (0 ⇒ unlimited).
        "token_budget",
        "rate_burst",
        "max_concurrency", "max_concurrency_per_principal",
        "audit_retention_days", "token_max_age_days",
        "request_timeout_secs", "cold_load_wait_secs",
        "max_audio_upload_bytes", "max_request_body_bytes",
        "mlx_cache_limit_bytes",
    ]
    /// Written bare (unquoted), like ints: floats and bools.
    public static let rawKeys: Set<String> = [
        "temperature", "speculative", "rate_limit", "preload",
        "encrypt_store", "persist_store",
        "deny_debugger_attach",
    ]
    public static let knownKeys: Set<String> = [
        "listen_host", "listen_port", "budget_bytes", "engine",
        "model", "model_store", "data_dir", "log_level",
        // ADR 026 — per-module default model keys (the LLM default is `model`).
        "embedding_model", "transcription_model", "diarization_model",
        "speaker_embedding_model",
        "log_dir", "max_tokens", "max_prompt_tokens", "temperature",
        "speculative", "auth_keys_file",
        "tls_cert", "tls_key", "rate_limit", "rate_burst",
        // ADR 041 — quotas. NOT deny-listed under ADR 037 (operability knobs,
        // not a path to daemon takeover), so both are settable via the config
        // API like `rate_limit`.
        "token_budget", "token_budget_window",
        "max_concurrency", "max_concurrency_per_principal",
        "audit_retention_days", "token_max_age_days",
        "request_timeout_secs", "cold_load_wait_secs", "preload",
        "max_audio_upload_bytes", "max_request_body_bytes",
        "mlx_cache_limit_bytes",
        // ADR 023 G2 — admission accounting mode (footprint | estimate).
        "governor_admission_mode",
        "encrypt_store", "persist_store",
        // ADR 024 T2 — opt-in debugger-attach denial (bool; see rawKeys).
        "deny_debugger_attach",
        "https_proxy", "http_proxy", "all_proxy", "no_proxy",
        "kv_compression",
    ]

    /// `--config` wins; else `$ATHENA_CONFIG`; else the installed file at
    /// the default prefix; else the in-repo dev copy.
    ///
    /// NJ2/NB9 (M66.4): the launchd plist exports `ATHENA_CONFIG` =
    /// `<prefix>/etc/athena/athena.toml`, so the daemon's TOML-only
    /// re-reads (kv_compression, the egress-proxy keys — none forwarded
    /// as plist args) resolve to the PREFIX-CORRECT file
    /// instead of the hard-coded `/usr/local`. A non-default `--prefix`
    /// install no longer silently drops those keys. Operators editing on a
    /// non-default prefix set `ATHENA_CONFIG` (or pass `--config`) so the
    /// CLI edits the same file the daemon reads.
    public static func resolvePath(_ override: String?) -> URL {
        if let override {
            return URL(
                fileURLWithPath: (override as NSString).expandingTildeInPath)
        }
        if let env = ProcessInfo.processInfo.environment["ATHENA_CONFIG"],
            !env.isEmpty
        {
            return URL(
                fileURLWithPath: (env as NSString).expandingTildeInPath)
        }
        let installed = URL(
            fileURLWithPath: "/usr/local/etc/athena/athena.toml")
        if FileManager.default.fileExists(atPath: installed.path) {
            return installed
        }
        return URL(fileURLWithPath: "deploy/athena.toml")
    }

    /// An active (uncommented) `key = …` assignment on this line?
    private static func isAssignment(
        _ line: Substring, _ key: String
    ) -> Bool {
        let t = line.drop(while: { $0 == " " || $0 == "\t" })
        guard !t.hasPrefix("#"), t.hasPrefix(key) else { return false }
        let rest = t.dropFirst(key.count)
            .drop(while: { $0 == " " || $0 == "\t" })
        return rest.first == "="
    }

    /// A commented `# key = …` line (so we can uncomment in place)?
    private static func isCommented(
        _ line: Substring, _ key: String
    ) -> Bool {
        var t = line.drop(while: { $0 == " " || $0 == "\t" })
        guard t.first == "#" else { return false }
        t = t.dropFirst().drop(while: { $0 == " " || $0 == "\t" })
        guard t.hasPrefix(key) else { return false }
        let rest = t.dropFirst(key.count)
            .drop(while: { $0 == " " || $0 == "\t" })
        return rest.first == "="
    }

    public enum Failure: Error, CustomStringConvertible {
        case unknownKey(String)
        case notAnInteger(String)
        case badValue(String, String)
        case noConfig(URL)
        case writeFailed(URL, String)
        public var description: String {
            switch self {
            case .unknownKey(let k):
                return
                    "unknown key '\(k)' (allowed: "
                    + ConfigEditor.knownKeys.sorted()
                    .joined(separator: ", ") + ")"
            case .notAnInteger(let k):
                return "\(k) must be an integer"
            case .badValue(let k, let want):
                return "\(k) must be \(want)"
            case .noConfig(let u): return "no config at \(u.path)"
            case .writeFailed(let u, let e):
                return "cannot write \(u.path): \(e)"
            }
        }
    }

    /// Write the config, surviving BOTH ownership shapes an installed
    /// appliance presents (ADR 037 amendment, 2026-07-25).
    ///
    /// The installed TOML is chowned to the SERVICE USER so the daemon can
    /// rewrite it (`PUT /api/config`, `athena config set`, the WebUI editor) —
    /// but its DIRECTORY deliberately stays root-owned, because
    /// `auth_keys_file` and the TLS key live in that same directory by default
    /// and a service-writable directory would let a compromised daemon replace
    /// them (unlink+create needs only directory write), silently defeating the
    /// ADR 037 deny-list that stops those same keys being *repointed*.
    ///
    /// An atomic write creates a temp file in that directory, so as the daemon
    /// it fails EACCES — the observed field bug: every daemon-mediated config
    /// write on a real install returned `writeFailed`, which is precisely what
    /// ADR 037 set out to remove. So: try atomic (root/dev-tree case), and on
    /// failure write IN PLACE, which needs only the file's own write bit — the
    /// bit the install already grants. Symmetrically, an atomic write replaces
    /// the inode and would leave the file ROOT-owned after one `sudo athena
    /// config set`, permanently breaking the daemon's own writes; restoring the
    /// prior uid/gid keeps a sudo edit from re-rooting a service-owned config.
    private static func writePreservingOwnership(
        _ text: String, to url: URL
    ) throws {
        let fm = FileManager.default
        let prior = try? fm.attributesOfItem(atPath: url.path)
        let uid = prior?[.ownerAccountID] as? NSNumber
        let gid = prior?[.groupOwnerAccountID] as? NSNumber
        do {
            try text.write(to: url, atomically: true, encoding: .utf8)
            if let uid, let gid {
                // Best-effort: only root can chown, and only root could have
                // taken this branch on a service-owned file anyway.
                try? fm.setAttributes(
                    [.ownerAccountID: uid, .groupOwnerAccountID: gid],
                    ofItemAtPath: url.path)
            }
        } catch {
            try text.write(to: url, atomically: false, encoding: .utf8)
        }
    }

    /// Validate + rewrite one scalar in place (replacing an active
    /// line or uncommenting a `# key =` one), then sanity-parse.
    /// THROWS rather than exiting — safe to call from the server
    /// (`/ui/api/config`); a bad request must never kill the daemon.
    public static func setScalarThrowing(
        key: String, value: String, in url: URL
    ) throws {
        guard knownKeys.contains(key) else {
            throw Failure.unknownKey(key)
        }
        let formatted: String
        if intKeys.contains(key) {
            guard Int(value) != nil else {
                throw Failure.notAnInteger(key)
            }
            formatted = "\(key) = \(value)"
        } else if rawKeys.contains(key) {
            // Bare, unquoted. Validate the two raw keys' shapes.
            if key == "temperature", Double(value) == nil {
                throw Failure.badValue(key, "a number")
            }
            if key == "rate_limit", Double(value) == nil {
                throw Failure.badValue(key, "a number")
            }
            if key == "speculative",
                value != "true", value != "false"
            {
                throw Failure.badValue(key, "true or false")
            }
            if key == "preload",
                value != "true", value != "false"
            {
                throw Failure.badValue(key, "true or false")
            }
            if key == "encrypt_store",
                value != "true", value != "false"
            {
                throw Failure.badValue(key, "true or false")
            }
            if key == "persist_store",
                value != "true", value != "false"
            {
                throw Failure.badValue(key, "true or false")
            }
            formatted = "\(key) = \(value)"
        } else {
            // NB8 (M66.4): validate the two enum-ish string keys at
            // set-time against their source-of-truth case lists, so a typo
            // is rejected HERE instead of bricking the next daemon boot
            // (engine → ArgumentParser, kv_compression → fail-closed
            // KVCompression.resolve).
            if key == "engine",
                !Engine.allCases.map(\.rawValue).contains(value)
            {
                throw Failure.badValue(
                    key,
                    "one of "
                        + Engine.allCases.map(\.rawValue)
                        .joined(separator: ", "))
            }
            // ADR 041 — the budget window is enum-ish; a typo here would
            // otherwise fail the daemon's next config parse.
            if key == "token_budget_window",
                !QuotaWindow.allCases.map(\.rawValue).contains(value)
            {
                throw Failure.badValue(
                    key,
                    "one of "
                        + QuotaWindow.allCases.map(\.rawValue)
                        .joined(separator: ", "))
            }
            if key == "kv_compression",
                !KVCompression.allCases.map(\.rawValue).contains(value)
            {
                throw Failure.badValue(
                    key,
                    "one of "
                        + KVCompression.allCases.map(\.rawValue)
                        .joined(separator: ", "))
            }
            // ADR 023 G2 — admission mode is an enum-ish string; reject a typo
            // at set-time against the source-of-truth case list (mirrors
            // engine / kv_compression above) so a bad value can't silently fall
            // back to the default at the next boot.
            if key == "governor_admission_mode",
                !GovernorMemory.AdmissionMode.allCases.map(\.rawValue)
                    .contains(value)
            {
                throw Failure.badValue(
                    key,
                    "one of "
                        + GovernorMemory.AdmissionMode.allCases.map(\.rawValue)
                        .joined(separator: ", "))
            }
            // NB2 (M66.4): the value is written quoted (`key = "<value>"`).
            // Reject a value containing a quote, backslash, or any control
            // character (incl. CR/LF) — a newline would inject arbitrary
            // EXTRA config lines (e.g. a forged `auth_keys_file`/`tls_cert`)
            // and a quote/backslash would corrupt the file. Reachable over
            // the network via `/ui/api/config`. (`#` is now safe — the
            // reader treats it literally inside quotes per J2.)
            if value.contains("\"") || value.contains("\\")
                || value.unicodeScalars.contains(where: {
                    $0.value < 0x20 || $0.value == 0x7F
                })
            {
                throw Failure.badValue(
                    key,
                    "free of quotes, backslashes, and control "
                        + "characters")
            }
            formatted = "\(key) = \"\(value)\""
        }
        guard
            let contents = try? String(
                contentsOf: url, encoding: .utf8)
        else { throw Failure.noConfig(url) }

        // Whether the file parsed BEFORE this edit — so a post-edit parse
        // failure can be attributed to the edit (roll back) vs. a
        // pre-existing problem like a missing required key (keep + warn).
        let wasParseable =
            (try? AthenaConfig.parse(toml: contents)) != nil

        var lines = contents.split(
            separator: "\n", omittingEmptySubsequences: false)
        if let i = lines.firstIndex(where: { isAssignment($0, key) }) {
            lines[i] = Substring(formatted)
        } else if let i = lines.firstIndex(where: {
            isCommented($0, key)
        }) {
            lines[i] = Substring(formatted)
        } else if let s = lines.firstIndex(where: { line in
            // B15 (M66.4): a NEW bare top-level key appended at EOF would
            // land inside the last `[section]` table. Insert it just before
            // the first section header so it stays top-level.
            line.drop(while: { $0 == " " || $0 == "\t" }).first == "["
        }) {
            lines.insert(Substring(formatted), at: s)
        } else {
            if lines.last?.isEmpty == true { lines.removeLast() }
            lines.append(Substring(formatted))
        }
        let rewritten = lines.joined(separator: "\n") + "\n"
        do {
            try writePreservingOwnership(rewritten, to: url)
        } catch {
            throw Failure.writeFailed(url, "\(error)")
        }
        // NB2 (M66.4): if THIS edit made a previously-valid config
        // unparseable, roll back to the pre-edit contents and report —
        // never leave a corrupt file that bricks the next daemon start. A
        // config that was already unparseable (e.g. missing a required
        // key) keeps the edit and only warns, so it can still be repaired.
        if (try? AthenaConfig.parse(file: url)) == nil {
            if wasParseable {
                try? writePreservingOwnership(contents, to: url)
                throw Failure.badValue(
                    key, "a value that keeps the config parseable")
            }
            FileHandle.standardError.write(
                Data(
                    "warning: config still unparseable (pre-existing)\n"
                        .utf8))
        }
    }
}
