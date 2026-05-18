import ArgumentParser
import AthenaCore
import Foundation

/// Shared host/port options for CLI subcommands that talk to a running
/// daemon over HTTP. (Portable client surface — M14.1 extraction.)
public struct DaemonOptions: ParsableArguments {
    @Option(help: "Daemon host.")
    public var host: String = "127.0.0.1"

    @Option(help: "Daemon port.")
    public var port: Int = GovernorConfig.defaultPort

    @Option(help: "Bearer key (else ATHENA_KEY env, else Keychain).")
    public var key: String?

    public init() {}

    public var base: String { "http://\(host):\(port)" }

    /// Resolved bearer key for this endpoint (flag > env > Keychain).
    public var authKey: String? {
        Credentials.resolve(explicit: key, host: host, port: port)
    }
}

/// Minimal JSON HTTP helpers for the thin-client subcommands.
public enum HTTPClient {
    public static func send(
        _ method: String, _ url: String, body: Data? = nil,
        key: String? = nil
    ) async throws -> (Int, Data) {
        var req = URLRequest(url: URL(string: url)!)
        req.httpMethod = method
        if let key, !key.isEmpty {
            req.setValue(
                "Bearer \(key)", forHTTPHeaderField: "Authorization")
        }
        if let body {
            req.httpBody = body
            req.setValue(
                "application/json", forHTTPHeaderField: "Content-Type")
        }
        let (data, resp) = try await URLSession.shared.data(for: req)
        let code = (resp as? HTTPURLResponse)?.statusCode ?? 0
        return (code, data)
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
        } else {
            print(String(data: data, encoding: .utf8) ?? "<binary>")
        }
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
