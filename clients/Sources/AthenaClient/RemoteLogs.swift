import ArgumentParser
import Foundation

#if canImport(FoundationNetworking)
    import FoundationNetworking
#endif

// M45.5 — remote daemon-log oversight. The macOS `athena logs` and the
// portable client both route through here when the operator is talking
// to a daemon (local loopback OR off-box). One-shot via `/api/logs`;
// SSE follow via `/api/logs/stream`. Pull only — the passive oracle
// never pushes log entries out.
public enum RemoteLogs {
    /// Projection of one `log show --style ndjson` entry (the
    /// daemon's compact LogEntryDTO).
    public struct Entry: Decodable {
        public let ts: String
        public let level: String
        public let category: String
        public let message: String
        public init(
            ts: String, level: String, category: String, message: String
        ) {
            self.ts = ts
            self.level = level
            self.category = category
            self.message = message
        }
    }
    private struct Report: Decodable {
        let logs: [Entry]
        let truncated: Bool?
    }

    /// `/api/logs` query string from the filters (categories are
    /// repeatable so each appears as a separate `category=` pair).
    static func query(
        since: String?, categories: [String], debug: Bool, limit: Int?
    ) -> String {
        func enc(_ s: String) -> String {
            s.addingPercentEncoding(
                withAllowedCharacters: .alphanumerics) ?? s
        }
        var parts: [String] = []
        if let s = since, !s.isEmpty { parts.append("since=\(enc(s))") }
        for c in categories where !c.isEmpty {
            parts.append("category=\(enc(c))")
        }
        if debug { parts.append("debug=1") }
        if let l = limit { parts.append("limit=\(l)") }
        return parts.isEmpty ? "" : "?" + parts.joined(separator: "&")
    }

    public static func show(
        _ d: DaemonOptions, since: String? = nil,
        categories: [String] = [], debug: Bool = false,
        limit: Int? = nil
    ) async throws {
        let q = query(
            since: since, categories: categories,
            debug: debug, limit: limit)
        let (code, data): (Int, Data)
        do {
            (code, data) = try await HTTPClient.send(
                "GET", d.base + "/api/logs" + q, key: d.authKey)
        } catch { HTTPClient.noDaemon(d, error) }
        guard code < 400,
            let r = try? JSONDecoder().decode(Report.self, from: data)
        else {
            HTTPClient.printJSON(data)
            if code >= 400 { throw ExitCode.failure }
            return
        }
        render(r.logs)
        if r.truncated == true {
            FileHandle.standardError.write(
                Data(
                    "(truncated — older entries dropped; narrow --since, raise --limit, or use --follow)\n"
                        .utf8))
        }
    }

    /// Stream `/api/logs/stream` SSE indefinitely (until Ctrl-C or
    /// the daemon's 10-min cap). Each event line on the wire is
    /// `data: {<Entry JSON>}\n\n`; print as a one-line summary so
    /// the operator can `grep` it.
    public static func stream(
        _ d: DaemonOptions, categories: [String] = [],
        debug: Bool = false
    ) async throws {
        let q = query(
            since: nil, categories: categories,
            debug: debug, limit: nil)
        guard let url = URL(string: d.base + "/api/logs/stream" + q)
        else {
            FailableExit.die(
                "error: bad URL \(d.base)/api/logs/stream")
        }
        var req = URLRequest(url: url)
        req.httpMethod = "GET"
        if let key = d.authKey, !key.isEmpty {
            req.setValue(
                "Bearer \(key)",
                forHTTPHeaderField: "Authorization")
        }
        req.setValue("text/event-stream", forHTTPHeaderField: "Accept")
        let (bytes, resp): (URLSession.AsyncBytes, URLResponse)
        do {
            (bytes, resp) = try await URLSession.shared.bytes(for: req)
        } catch { HTTPClient.noDaemon(d, error) }
        if let http = resp as? HTTPURLResponse, http.statusCode >= 400 {
            FailableExit.die(
                "error: /api/logs/stream → \(http.statusCode) "
                    + "(check auth — admin-only)")
        }
        for try await line in bytes.lines {
            guard line.hasPrefix("data: ") else { continue }
            let json = String(line.dropFirst(6))
            guard
                let data = json.data(using: .utf8),
                let entry = try? JSONDecoder().decode(
                    Entry.self, from: data)
            else { continue }
            print(renderLine(entry))
        }
    }

    /// Render the historical table — same compact one-line format
    /// as the stream so historical + live views are visually
    /// consistent.
    public static func render(_ entries: [Entry]) {
        guard !entries.isEmpty else {
            print("no log entries")
            return
        }
        for e in entries { print(renderLine(e)) }
    }

    private static func renderLine(_ e: Entry) -> String {
        "\(e.ts) \(e.level) \(e.category): \(e.message)"
    }
}

// MARK: - Portable command struct (remote-only)

/// `athena logs` on the portable client (Linux/Windows) — always
/// remote. On macOS the local-overloading verb in
/// Sources/athena/Commands/Logs.swift defaults to this same remote
/// path (via DaemonOptions.isRemote check) and exposes `--offline`
/// to fall back to direct `/usr/bin/log` shell-out when the daemon
/// is down.
public struct LogsCmd: AsyncParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "logs",
        abstract:
            "Show / stream the daemon's unified-log entries via /api/logs."
    )
    @OptionGroup public var daemon: DaemonOptions
    @Flag(name: .shortAndLong, help: "Stream new entries via SSE (live).")
    public var follow = false
    @Option(help: "How far back to look (default 1h). E.g. 5m, 1h.")
    public var since: String = "1h"
    @Option(
        name: .customLong("category"),
        parsing: .singleValue,
        help: "Filter by category (repeatable). E.g. daemon, audit, model.llm."
    )
    public var categories: [String] = []
    @Flag(
        help:
            "Include info+debug entries (memory-only by default — only persists if `sudo log config --mode \"level:debug\" --subsystem athena` was set)."
    )
    public var debug = false
    @Option(help: "Max entries returned by /api/logs (default 200, capped 5000).")
    public var limit: Int?
    public init() {}
    public func run() async throws {
        if follow {
            try await RemoteLogs.stream(
                daemon, categories: categories, debug: debug)
        } else {
            try await RemoteLogs.show(
                daemon, since: since, categories: categories,
                debug: debug, limit: limit)
        }
    }
}
