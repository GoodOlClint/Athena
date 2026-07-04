import Foundation

#if canImport(FoundationNetworking)
    import FoundationNetworking
#endif

/// Daemon-mediated config + restart over the control plane (ADR 037). Lets
/// `athena config set` / `athena restart` run WITHOUT sudo when a daemon is
/// reachable — the daemon (running as the service user, owning its TOML) does
/// the write / self-restart. The sudo path stays as a fallback for the
/// daemon-down case.
public enum RemoteConfig {
    public enum SetOutcome {
        case ok
        /// The daemon rejected it (4xx) — surfaced verbatim; do NOT fall back.
        case rejected(Int, Data)
        /// Couldn't reach a daemon — the caller should fall back to the local
        /// (sudo/offline) path.
        case unreachable
    }

    /// `PUT /api/config {key,value}`. Reachable + 2xx ⇒ `.ok`; reachable + 4xx
    /// ⇒ `.rejected`; connection failure ⇒ `.unreachable`.
    public static func set(
        _ d: DaemonOptions, key: String, value: String
    ) async -> SetOutcome {
        guard
            let body = try? JSONSerialization.data(withJSONObject: [
                "key": key, "value": value,
            ])
        else { return .unreachable }
        do {
            let (code, data) = try await HTTPClient.send(
                "PUT", d.base + "/api/config", body: body, key: d.authKey)
            if code < 300 { return .ok }
            return .rejected(code, data)
        } catch {
            return .unreachable
        }
    }

    /// `POST /api/admin/restart`. Returns true when the daemon acknowledged the
    /// restart (2xx); false when unreachable/refused (caller falls back).
    public static func restart(_ d: DaemonOptions) async -> Bool {
        do {
            let (code, _) = try await HTTPClient.send(
                "POST", d.base + "/api/admin/restart", body: Data("{}".utf8),
                key: d.authKey)
            return code < 300
        } catch {
            return false
        }
    }
}
