import Foundation

/// Whether the daemon persists its auth/audit/usage SQLite store on disk.
public enum StorePersistence: String, Equatable, Sendable {
    /// Open (creating if needed) `<data-dir>/athena.sqlite`.
    case persistent
    /// In-memory only — no file is ever written; audit/usage live for the
    /// process lifetime and vanish on exit (ADR 025 S4 stateless loopback).
    case ephemeral
}

/// ADR 025 S4 — decide whether the daemon's auth/audit/usage store is
/// persisted on disk or kept ephemeral (stateless loopback). MLX-free and
/// unit-pinned (ADR 008/009): the rule is the slice's behavioral contract.
///
/// After ADR 025 (queue + vector tenants removed) and ADR 026 (allowlist
/// retired), the only remaining store tenants are auth/audit/usage. None of
/// them needs durable state when there is nothing to authenticate — so a
/// loopback dev daemon with no credentials writes **no `athena.sqlite`** at
/// all, leaving zero request-related data at rest.
public enum StoreMode {
    private static let loopbackHosts: Set<String> = [
        "127.0.0.1", "::1", "localhost",
    ]

    public static func isLoopback(_ host: String) -> Bool {
        loopbackHosts.contains(host)
    }

    /// Resolve the store persistence mode. **Ephemeral** (no file) iff there
    /// is nothing that requires durable state:
    /// - no bootstrap (file/env) auth keys → nothing to authenticate against;
    /// - no existing `athena.sqlite` on disk → no prior state to respect;
    /// - a loopback bind → a non-loopback bind is auth-required (fail-closed
    ///   elsewhere) and so always persistent;
    /// - at-rest encryption not opted into → `encrypt_store` implies the
    ///   operator wants a durable, encrypted file.
    ///
    /// `persistOverride` (the `persist_store` config switch) forces persistent
    /// regardless — the ADR-025 "config switch controls persistence
    /// independent of mode" lever, for an operator who wants audit/usage kept
    /// in a loopback dev run.
    public static func resolve(
        hasBootstrapKeys: Bool,
        dbFileExists: Bool,
        isLoopback: Bool,
        encryptStore: Bool,
        persistOverride: Bool
    ) -> StorePersistence {
        if persistOverride { return .persistent }
        if hasBootstrapKeys { return .persistent }
        if dbFileExists { return .persistent }
        if encryptStore { return .persistent }
        if !isLoopback { return .persistent }
        return .ephemeral
    }
}
