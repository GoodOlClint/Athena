import Foundation

/// Which config keys may be set over the daemon's `PUT /api/config` control
/// route (ADR 037). Config takeover ≈ daemon takeover, and loopback dev mode
/// has no auth (so the route is open to any local process there), so the keys
/// that would let a remote/local caller redirect auth, TLS, at-rest encryption,
/// the data directory, or the debugger posture are **denied** — they stay
/// TOML-plus-sudo. Pure and MLX-free (ADR 008/009), unit-pinned.
public enum ConfigApiPolicy {
    /// Keys refused by `PUT /api/config` (edit the TOML + sudo-restart instead).
    public static let deniedKeys: Set<String> = [
        "auth_keys_file",
        "tls_cert",
        "tls_key",
        "encrypt_store",
        "data_dir",
        "deny_debugger_attach",
    ]

    /// May this key be set via the API? (Existence/known-key validation stays
    /// with `ConfigEditor`; this is only the security deny-list.)
    public static func isSettable(_ key: String) -> Bool {
        !deniedKeys.contains(key)
    }

    /// The `400` message for a denied key — names why, points at the TOML path.
    public static func deniedMessage(_ key: String) -> String {
        "'\(key)' cannot be set via the API (it governs auth/TLS/encryption/"
            + "data-dir/debugger posture — config takeover would be daemon "
            + "takeover). Edit the config file and restart with sudo."
    }
}
