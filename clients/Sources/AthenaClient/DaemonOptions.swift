import ArgumentParser
import Foundation

#if canImport(FoundationNetworking)
    import FoundationNetworking
#endif

/// Shared host/port options for CLI subcommands that talk to a running
/// daemon over HTTP. (Portable client surface — M14.1 extraction.)
public struct DaemonOptions: ParsableArguments {
    @Option(help: "Daemon host.")
    public var host: String = "127.0.0.1"

    @Option(help: "Daemon port.")
    public var port: Int = athenaDefaultPort

    @Option(help: "Bearer key (else ATHENA_KEY env, else Keychain).")
    public var key: String?

    public init() {}

    public var base: String { "http://\(host):\(port)" }

    /// Resolved bearer key for this endpoint (flag > env > Keychain).
    public var authKey: String? {
        Credentials.resolve(explicit: key, host: host, port: port)
    }

    /// True when `--host` points OFF-BOX. The macOS full `athena`
    /// overloads its local model/RBAC verbs: a loopback host keeps the
    /// pre-existing LOCAL behavior (operate on this box's store/DB
    /// directly); an off-box host routes the SAME verb to that
    /// daemon's `/api/*` over HTTP. The portable client has no local
    /// path, so its command structs always go remote regardless of
    /// this. Loopback = `127.0.0.1` / `localhost` / `::1` (the address
    /// `athena start` binds by default).
    public var isRemote: Bool {
        !Self.loopback.contains(host.lowercased())
    }

    private static let loopback: Set<String> = [
        "127.0.0.1", "localhost", "::1", "[::1]",
    ]
}

/// Minimal JSON HTTP helpers for the thin-client subcommands.
public enum HTTPClient {
    /// Bounded-retry HTTP call (M19). Idempotency-safe: see
    /// `RetryPolicy`. Backoff sleeps between attempts; the final
    /// outcome (status or thrown `URLError`) is returned/rethrown so
    /// callers behave exactly as before once retries are exhausted.
    public static func send(
        _ method: String, _ url: String, body: Data? = nil,
        key: String? = nil
    ) async throws -> (Int, Data) {
        let policy = RetryPolicy.fromEnvironment()
        var attempt = 0
        while true {
            var req = URLRequest(url: URL(string: url)!)
            req.httpMethod = method
            if let key, !key.isEmpty {
                req.setValue(
                    "Bearer \(key)",
                    forHTTPHeaderField: "Authorization")
            }
            if let body {
                req.httpBody = body
                req.setValue(
                    "application/json",
                    forHTTPHeaderField: "Content-Type")
            }
            do {
                let (data, resp) =
                    try await URLSession.shared.data(for: req)
                let http = resp as? HTTPURLResponse
                let code = http?.statusCode ?? 0
                if let d = policy.delay(
                    attempt: attempt, method: method,
                    outcome: .status(code),
                    retryAfter: Self.retryAfter(http))
                {
                    try? await Task.sleep(
                        nanoseconds: UInt64(d * 1_000_000_000))
                    attempt += 1
                    continue
                }
                return (code, data)
            } catch let e as URLError {
                if let d = policy.delay(
                    attempt: attempt, method: method,
                    outcome: .transport(e.code))
                {
                    try? await Task.sleep(
                        nanoseconds: UInt64(d * 1_000_000_000))
                    attempt += 1
                    continue
                }
                throw e
            }
        }
    }

    /// `Retry-After` in delta-seconds form (HTTP-date form ignored;
    /// the cap in `RetryPolicy` bounds it regardless).
    private static func retryAfter(_ r: HTTPURLResponse?)
        -> TimeInterval?
    {
        guard
            let v = r?.value(forHTTPHeaderField: "Retry-After"),
            let s = TimeInterval(
                v.trimmingCharacters(in: .whitespaces)), s >= 0
        else { return nil }
        return s
    }

    /// Pretty-print JSON `data`, or raw text on failure.
    public static func printJSON(_ data: Data) {
        if let obj = try? JSONSerialization.jsonObject(with: data),
            let pretty = try? JSONSerialization.data(
                withJSONObject: obj,
                options: [.prettyPrinted, .sortedKeys]),
            let s = String(data: pretty, encoding: .utf8)
        {
            print(s)
            renderHint(obj)
        } else {
            print(String(data: data, encoding: .utf8) ?? "<binary>")
        }
    }

    /// M43.4 #5 — operator-facing remediation lines for error
    /// envelopes that carry `error.hint`. Daemon adds these to auth
    /// denies (`ATHENA_KEY`, `/ui/login`, role-grant guidance) and
    /// classified inference errors. Printed to stderr so structured
    /// JSON consumers on stdout aren't disturbed; humans see both.
    private static func renderHint(_ obj: Any) {
        guard let root = obj as? [String: Any],
            let err = root["error"] as? [String: Any],
            let hint = err["hint"] as? String,
            !hint.isEmpty
        else { return }
        FileHandle.standardError.write(
            Data(("hint: " + hint + "\n").utf8))
    }

    public static func noDaemon(_ d: DaemonOptions, _ e: Error)
        -> Never
    {
        FailableExit.die(
            "no running athena daemon at \(d.host):\(d.port) "
                + "(\(e.localizedDescription))")
    }
}

public enum FailableExit {
    public static func die(_ msg: String) -> Never {
        FileHandle.standardError.write(Data((msg + "\n").utf8))
        exit(1)
    }
}
