import AthenaClient
import Crypto
import Foundation

/// At-rest encryption key resolution for the SQLite store (M34.3b).
///
/// Precedence: `ATHENA_STORE_KEY` env > Keychain (the shared `athena`
/// service, account `store:key`). The daemon mints + persists a random
/// 256-bit key on first run when `encrypt_store` is on; the CLI verbs
/// only READ it (to open an already-encrypted store). The key is never
/// written to the TOML or to disk in plaintext — only the daemon process
/// and same-user CLI invocations can resolve it.
enum StoreKey {
    static let account = "store:key"
    static let envVar = "ATHENA_STORE_KEY"

    /// Read-only resolution (no generation). `nil` ⇒ no key configured ⇒
    /// the store opens as a standard plaintext database.
    ///
    /// B8 (M66.3): `trustEnv: false` skips `ATHENA_STORE_KEY`. A privileged
    /// (root) `install`/`doctor` invoked via `sudo` inherits the INVOKER's
    /// environment, not the service user's, so the env var there is the
    /// wrong source (and a credential exposure); those callers pass
    /// `trustEnv: false` and rely on the Keychain (or, for a fresh install,
    /// fall through to a plaintext store the daemon encrypts on first boot
    /// as the service user).
    static func resolve(trustEnv: Bool = true) -> String? {
        if trustEnv {
            let env = ProcessInfo.processInfo.environment
            if let e = env[envVar], !e.isEmpty { return e }
        }
        return Secrets.read(account: account)
    }

    /// Resolve, or mint + persist a fresh random 256-bit key. Used by the
    /// daemon when `encrypt_store` is enabled. Throws (⇒ the daemon
    /// refuses to start, fail-closed) if no key exists and one cannot be
    /// stored — never serves an "encrypted" store it can't actually key.
    static func ensure() throws -> String {
        if let k = resolve() { return k }
        let hex = SymmetricKey(size: .bits256).withUnsafeBytes { buf in
            buf.map { String(format: "%02x", $0) }.joined()
        }
        do {
            try Secrets.store(hex, account: account)
        } catch {
            throw StoreKeyError.cannotPersist("\(error)")
        }
        return hex
    }

    /// How `resolve()` would source the key, for `athena doctor`
    /// (no secret is printed). `trustEnv: false` mirrors a privileged
    /// caller that ignores the sudo-inherited env (B8).
    static func source(trustEnv: Bool = true) -> String {
        if trustEnv {
            let env = ProcessInfo.processInfo.environment
            if let v = env[envVar], !v.isEmpty { return "\(envVar) env" }
        }
        if Secrets.read(account: account) != nil { return "Keychain" }
        return "none"
    }

    enum StoreKeyError: Error, CustomStringConvertible {
        case cannotPersist(String)
        var description: String {
            switch self {
            case .cannotPersist(let m):
                return
                    "cannot store the generated store encryption key in "
                    + "the Keychain (\(m)); set \(StoreKey.envVar) or fix "
                    + "Keychain access, then retry"
            }
        }
    }
}
