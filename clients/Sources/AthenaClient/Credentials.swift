import Foundation

#if canImport(Darwin)
    import Darwin
#elseif canImport(Glibc)
    import Glibc
#endif

/// One keyed secret store for the thin CLI (M13). A single login-
/// Keychain service (`athena`) with an arbitrary `account` per secret,
/// so the daemon bearer key (`<host>:<port>`) and the Hugging Face
/// token (`hf:token`) share one backend, one point of use.
///
/// Portable by design: the Keychain calls are `#if os(macOS)`-gated;
/// other platforms resolve via flag/env only (a future portable
/// client target compiles unchanged).
public enum Secrets {
    static let service = "athena"

    #if os(macOS)
        @discardableResult
        private static func security(_ args: [String]) -> (
            Int32, String
        ) {
            let p = Process()
            p.executableURL = URL(
                fileURLWithPath: "/usr/bin/security")
            p.arguments = args
            let out = Pipe()
            p.standardOutput = out
            p.standardError = FileHandle.nullDevice
            try? p.run()
            p.waitUntilExit()
            let data = out.fileHandleForReading.readDataToEndOfFile()
            return (
                p.terminationStatus,
                String(data: data, encoding: .utf8)?
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                    ?? ""
            )
        }

        public static func read(account: String) -> String? {
            let (rc, val) = security([
                "find-generic-password", "-s", service, "-a",
                account, "-w",
            ])
            return rc == 0 && !val.isEmpty ? val : nil
        }

        public static func store(
            _ value: String, account: String
        ) throws {
            // -U updates an existing item rather than erroring.
            let (rc, _) = security([
                "add-generic-password", "-U", "-s", service, "-a",
                account, "-w", value,
            ])
            guard rc == 0 else {
                throw CredentialError.keychain(
                    "security add-generic-password failed (\(rc))")
            }
        }

        @discardableResult
        public static func remove(account: String) -> Bool {
            security([
                "delete-generic-password", "-s", service, "-a",
                account,
            ]).0 == 0
        }

        /// Store a secret in the INVOKING USER's Keychain even when
        /// the current process is running as root via sudo (M45.6).
        /// `geteuid() == 0` + `SUDO_USER` set ⇒ drop priv via
        /// `sudo -u <user>` to invoke `security`; otherwise behaves
        /// like `store(_:account:)`. Operator-Keychain writes from
        /// `sudo athena install` would otherwise land in root's
        /// Keychain (useless to the operator's interactive
        /// session).
        ///
        /// Best-effort: throws if the security call fails (e.g.
        /// locked Keychain, no GUI session). Callers should catch
        /// and surface a one-line "couldn't auto-stash, run
        /// `athena auth login`" fallback.
        public static func storeAsInvokingOperator(
            _ value: String, account: String
        ) throws {
            let env = ProcessInfo.processInfo.environment
            guard geteuid() == 0,
                let sudoUser = env["SUDO_USER"],
                !sudoUser.isEmpty, sudoUser != "root"
            else {
                try store(value, account: account)
                return
            }
            let p = Process()
            p.executableURL = URL(fileURLWithPath: "/usr/bin/sudo")
            // `-u` drops to the invoking user; `-H` sets HOME so
            // `security` finds that user's login Keychain (not
            // root's). `-n` makes sudo non-interactive — root → any
            // user doesn't need a password, so this should succeed
            // without blocking on stdin.
            p.arguments = [
                "-u", sudoUser, "-H", "-n",
                "/usr/bin/security", "add-generic-password",
                "-U", "-s", service, "-a", account, "-w", value,
            ]
            p.standardOutput = FileHandle.nullDevice
            p.standardError = FileHandle.nullDevice
            try p.run()
            p.waitUntilExit()
            guard p.terminationStatus == 0 else {
                throw CredentialError.keychain(
                    "sudo -u \(sudoUser) security failed "
                        + "(\(p.terminationStatus))")
            }
        }
    #else
        public static func read(account: String) -> String? { nil }
        public static func store(_ value: String, account: String)
            throws
        {
            throw CredentialError.unsupported
        }
        @discardableResult
        public static func remove(account: String) -> Bool { false }
        public static func storeAsInvokingOperator(
            _ value: String, account: String
        ) throws {
            throw CredentialError.unsupported
        }
    #endif
}

/// Client-side bearer-credential resolution for the CLI commands that
/// talk to a (local or remote) daemon. Precedence: `--key` >
/// `ATHENA_KEY` env > the platform secret store > none. One Keychain
/// item per daemon endpoint, so local + remote daemons keep distinct
/// keys. M12.3 (now backed by the keyed `Secrets` store, M13).
public enum Credentials {
    private static func account(_ host: String, _ port: Int)
        -> String
    {
        "\(host):\(port)"
    }

    /// Resolve the bearer key for a daemon endpoint, or nil.
    public static func resolve(
        explicit: String?, host: String, port: Int
    ) -> String? {
        if let explicit, !explicit.isEmpty { return explicit }
        if let env = ProcessInfo.processInfo
            .environment["ATHENA_KEY"], !env.isEmpty
        {
            return env
        }
        return Secrets.read(account: account(host, port))
    }

    public static func keychainRead(host: String, port: Int)
        -> String?
    {
        Secrets.read(account: account(host, port))
    }

    public static func store(
        _ key: String, host: String, port: Int
    ) throws {
        try Secrets.store(key, account: account(host, port))
    }

    /// M45.6: install-time variant. When invoked under `sudo`, this
    /// writes to the INVOKING operator's Keychain (`$SUDO_USER`) so
    /// `athena <verb>` works from their interactive shell without
    /// `athena auth login`. Falls back to the regular `store` when
    /// not sudo.
    public static func storeAsInvokingOperator(
        _ key: String, host: String, port: Int
    ) throws {
        try Secrets.storeAsInvokingOperator(
            key, account: account(host, port))
    }

    @discardableResult
    public static func remove(host: String, port: Int) -> Bool {
        Secrets.remove(account: account(host, port))
    }
}

/// Hugging Face token resolution for gated/private model downloads
/// (M13). The `swift-huggingface` client auto-detects `HF_TOKEN`
/// (then `HUGGING_FACE_HUB_TOKEN`, then token files), so the whole
/// integration is: resolve a stored token and export it to the env
/// before any `#hubDownloader()` — exactly the `HF_HOME` convention
/// in `serve`. An operator-set token env is NEVER overridden.
public enum HFAuth {
    /// Keychain account under the shared `athena` service.
    static let account = "hf:token"

    private static func envToken() -> String? {
        let e = ProcessInfo.processInfo.environment
        if let t = e["HF_TOKEN"], !t.isEmpty { return t }
        if let t = e["HUGGING_FACE_HUB_TOKEN"], !t.isEmpty {
            return t
        }
        return nil
    }

    /// Precedence: explicit > HF_TOKEN / HUGGING_FACE_HUB_TOKEN env >
    /// Keychain. (Token files are left to the Hub client itself.)
    public static func resolve(explicit: String? = nil) -> String? {
        if let explicit, !explicit.isEmpty { return explicit }
        if let t = envToken() { return t }
        return Secrets.read(account: account)
    }

    /// If no token env is already set, export a Keychain-stored token
    /// as `HF_TOKEN` so the Hub client picks it up. Operator env wins
    /// (never overridden), mirroring the `HF_HOME` rule in `serve`.
    public static func exportToEnv() {
        if envToken() != nil { return }
        if let t = Secrets.read(account: account), !t.isEmpty {
            setenv("HF_TOKEN", t, 1)
        }
    }

    /// How `resolve()` would source the token, for `hf status`
    /// (no secret printed).
    public static func source() -> String {
        let e = ProcessInfo.processInfo.environment
        if let t = e["HF_TOKEN"], !t.isEmpty {
            return "HF_TOKEN env"
        }
        if let t = e["HUGGING_FACE_HUB_TOKEN"], !t.isEmpty {
            return "HUGGING_FACE_HUB_TOKEN env"
        }
        if Secrets.read(account: account) != nil { return "Keychain" }
        return "none (run `athena hf login`, or set HF_TOKEN)"
    }

    public static func store(_ token: String) throws {
        try Secrets.store(token, account: account)
    }

    @discardableResult
    public static func remove() -> Bool {
        Secrets.remove(account: account)
    }
}

/// Egress-proxy Basic-auth credentials (M13.2). One global proxy ⇒ a
/// single Keychain item (account `proxy:auth`, value `user:password`).
/// Stored via `athena proxy login`; never written to disk in
/// plaintext. Inline `user:pass@host` in the proxy URL is also
/// accepted, but a Keychain credential takes precedence.
public enum ProxyAuth {
    static let account = "proxy:auth"

    /// (user, pass) split on the FIRST `:`; usernames containing a
    /// colon are unsupported (documented).
    public static func read() -> (user: String, pass: String)? {
        guard let raw = Secrets.read(account: account),
            let i = raw.firstIndex(of: ":")
        else { return nil }
        let u = String(raw[..<i])
        let p = String(raw[raw.index(after: i)...])
        return u.isEmpty ? nil : (u, p)
    }

    public static func store(user: String, pass: String) throws {
        try Secrets.store("\(user):\(pass)", account: account)
    }

    @discardableResult
    public static func remove() -> Bool {
        Secrets.remove(account: account)
    }

    public static func source() -> String {
        read() != nil ? "Keychain" : "none (inline URL or unset)"
    }
}

public enum CredentialError: Error, CustomStringConvertible {
    case keychain(String)
    case unsupported
    public var description: String {
        switch self {
        case .keychain(let m): return m
        case .unsupported:
            return
                "secret store unavailable on this platform — use "
                + "env (ATHENA_KEY / HF_TOKEN) or the matching flag"
        }
    }
}
