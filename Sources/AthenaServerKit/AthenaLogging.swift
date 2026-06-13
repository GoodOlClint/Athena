import AthenaCore
import Darwin
import Foundation
import Logging
import os

// Centralized logging (M10 → M45). Athena's daemon routes ALL output
// through swift-log. Under launchd (`athena load --background`), the
// SOLE sink is the macOS unified log (`os.Logger`, subsystem
// "athena"); operator queries via `log show` / `log stream`, and
// `sudo log config --mode "level:debug" --subsystem athena` is the
// live verbosity gate. Foreground invocation (interactive `athena
// load` or `athena start` without install) adds a stdout
// `TerminalLogHandler` so the operator sees logs in their terminal in
// real time. No file sinks, no UDP. M45.1 dropped the opt-in syslog
// UDP shipper — operators wanting off-box shipping run FluentBit /
// vector.dev / syslog-ng against `subsystem == "athena"`.

/// Per-request log scope set by `AuthMiddleware` once the principal
/// is resolved. The swift-log `MetadataProvider` installed at
/// bootstrap reads this TaskLocal and surfaces `req=…` / `principal=…`
/// on every log line emitted from within the request task hierarchy,
/// across `await` boundaries.
///
/// Outside any request — startup, queue worker, lifecycle — the
/// TaskLocal is unbound and no req/principal keys appear, which is
/// correct: those lines have no request to correlate against.
///
/// `principal` is nil for unauthenticated requests (auth-off
/// loopback mode); the req-id still applies so the line is
/// correlatable.
public struct LogScope: Sendable {
    public let req: String
    public let principal: String?

    @TaskLocal
    public static var current: LogScope? = nil

    public init(req: String = UUID().uuidString, principal: String?) {
        self.req = req
        self.principal = principal
    }
}

public enum AthenaLog {
    /// Unified-logging subsystem. Simple, not reverse-DNS (Apple
    /// allows any string; this matches the binary/command name).
    static let subsystem = "athena"

    /// swift-log label for daemon/server logs. The label convention
    /// itself lives in AthenaCore (`AthenaLogLabel`, shared with the
    /// governor seam, dependency-free); the os.Logger `category` is
    /// derived by stripping the `athena.` prefix — anything else
    /// (Hummingbird's serverName "athena", NIO, etc.) ⇒ `daemon`.
    public static let daemonLabel = AthenaLogLabel.daemon

    static func category(forLabel label: String) -> String {
        let p = "athena."
        guard label.hasPrefix(p), label.count > p.count else {
            return "daemon"
        }
        return String(label.dropFirst(p.count))
    }

    /// Parse a log-level string (case-insensitive). nil/invalid ⇒ nil
    /// so the caller can decide the default (and warn on invalid).
    public static func level(_ s: String?) -> Logging.Logger.Level? {
        s.flatMap {
            Logging.Logger.Level(rawValue: $0.lowercased())
        }
    }

    /// Bootstrap once, before any `Logger` is created.
    ///
    /// - `background=true` (launchd-spawned): the SOLE sink is the
    ///   unified-log handler at `.trace`, so the macOS `log config`
    ///   mode is the live verbosity gate (otherwise the swift-side
    ///   filter would drop `.debug`/`.info` before `os.Logger` ever
    ///   sees them).
    /// - `background=false` (foreground / interactive): adds a
    ///   `TerminalLogHandler` alongside the unified-log handler so
    ///   the operator sees logs in their terminal. `terminalLevel`
    ///   gates ONLY the terminal sink; the unified log stays
    ///   `.trace`-permissive so historical `log show` queries
    ///   remain truthful regardless of what the operator picked at
    ///   the CLI.
    public static func bootstrap(
        background: Bool = false,
        terminalLevel: Logging.Logger.Level = .info
    ) {
        // M45.3: surface req/principal from the LogScope TaskLocal on
        // every log line. The provider closure is invoked at each
        // log() call; if no scope is bound (startup, worker, etc.)
        // it returns empty metadata and the keys are simply absent.
        let provider = Logging.Logger.MetadataProvider {
            guard let scope = LogScope.current else { return [:] }
            var meta: Logging.Logger.Metadata = [
                "req": "\(scope.req)"
            ]
            if let principal = scope.principal {
                meta["principal"] = "\(principal)"
            }
            return meta
        }
        LoggingSystem.bootstrap(
            { label, metadataProvider in
                var unified = OSUnifiedLogHandler(label: label)
                unified.logLevel = .trace
                unified.metadataProvider = metadataProvider
                if background {
                    return unified
                }
                var terminal = TerminalLogHandler(
                    category: category(forLabel: label))
                terminal.logLevel = terminalLevel
                terminal.metadataProvider = metadataProvider
                return MultiplexLogHandler([terminal, unified])
            },
            metadataProvider: provider)
    }
}

/// swift-log `LogHandler` for foreground terminal output. ISO 8601
/// UTC with millisecond precision so a redirected file (e.g.
/// `athena load 2> log` or `athena start 2>&1 | tee log`) sorts
/// cleanly against unified-log timestamps and against other
/// machine-collected logs. Format:
///
///     2026-05-27T21:30:45.123Z notice daemon: athena daemon up
///
/// Writes to STDERR (standard Unix logging convention): keeps stdout
/// reserved for non-log output (`athena start`'s "started…" line, or
/// whatever the daemon command itself prints), so operators can
/// `athena start | jq` without contaminating the pipe with log noise,
/// and `2>` works as expected to silence/redirect just the logs.
///
/// Metadata is appended as space-separated `key=value` pairs after
/// the message (M45.3 plumbs req/principal/function through the
/// swift-log `MetadataProvider`).
struct TerminalLogHandler: LogHandler {
    let category: String
    var logLevel: Logging.Logger.Level = .info
    var metadata: Logging.Logger.Metadata = [:]
    var metadataProvider: Logging.Logger.MetadataProvider?

    // ISO8601DateFormatter is documented thread-safe for shared
    // read-only string-from-Date use.
    nonisolated(unsafe) private static let ts: ISO8601DateFormatter = {
        let f = ISO8601DateFormatter()
        f.formatOptions = [
            .withInternetDateTime, .withFractionalSeconds,
        ]
        return f
    }()

    subscript(metadataKey key: String) -> Logging.Logger.Metadata.Value? {
        get { metadata[key] }
        set { metadata[key] = newValue }
    }

    func log(
        level: Logging.Logger.Level,
        message: Logging.Logger.Message,
        metadata explicit: Logging.Logger.Metadata?,
        source: String,
        file: String,
        function: String,
        line: UInt
    ) {
        var merged = self.metadata
        if let provided = metadataProvider?.get() {
            merged.merge(provided) { _, new in new }
        }
        if let explicit { merged.merge(explicit) { _, new in new } }
        // M45.3: per-emission `function=` field, sourced from
        // swift-log's call-site capture. Always present, sorted with
        // the rest so the in-merged-view filter (`function=loadModel`)
        // works regardless of whether req/principal are bound.
        merged["function"] = "\(function)"
        var text = message.description
        if !merged.isEmpty {
            text +=
                " "
                + merged.sorted { $0.key < $1.key }
                .map { "\($0.key)=\($0.value)" }
                .joined(separator: " ")
        }
        let stamp = Self.ts.string(from: Date())
        let out =
            "\(stamp) \(level.rawValue) \(category): \(text)\n"
        FileHandle.standardError.write(Data(out.utf8))
    }
}

/// swift-log `LogHandler` that forwards to a per-category
/// `os.Logger`, so messages land in the macOS unified log.
struct OSUnifiedLogHandler: LogHandler {
    private let osLogger: os.Logger
    var logLevel: Logging.Logger.Level = .info
    var metadata: Logging.Logger.Metadata = [:]
    var metadataProvider: Logging.Logger.MetadataProvider?

    init(label: String) {
        self.osLogger = os.Logger(
            subsystem: AthenaLog.subsystem,
            category: AthenaLog.category(forLabel: label))
    }

    subscript(metadataKey key: String) -> Logging.Logger.Metadata.Value? {
        get { metadata[key] }
        set { metadata[key] = newValue }
    }

    func log(
        level: Logging.Logger.Level,
        message: Logging.Logger.Message,
        metadata explicit: Logging.Logger.Metadata?,
        source: String,
        file: String,
        function: String,
        line: UInt
    ) {
        var merged = self.metadata
        if let provided = metadataProvider?.get() {
            merged.merge(provided) { _, new in new }
        }
        if let explicit { merged.merge(explicit) { _, new in new } }
        // M45.3: same per-emission `function=` field as
        // TerminalLogHandler so a merged `log show` view filters
        // identically by function across both sinks.
        merged["function"] = "\(function)"

        var text = message.description
        if !merged.isEmpty {
            let pairs = merged.sorted { $0.key < $1.key }
                .map { "\($0.key)=\($0.value)" }
                .joined(separator: " ")
            text += " {\(pairs)}"
        }
        // Server logs are kept readable in Console rather than
        // redacted as `<private>`. DO NOT interpolate user prompt
        // content, bearer-token strings, or Keychain secrets into the
        // message field — once a secret reaches os_log under
        // `.public` privacy it WILL appear in plaintext in Console
        // and sysdiagnose. The audit trail keeps `target=u:foo` /
        // `t:hash[:8]` and never the raw token.
        osLogger.log(
            level: Self.osType(level), "\(text, privacy: .public)")
    }

    /// swift-log → OSLogType. `.default` is the unified-log "notice"
    /// tier — persisted to disk, unlike `.info`/`.debug` which are
    /// memory-only (won't appear in `log show` after the fact). So
    /// `.notice`/`.warning` map to `.default` (persisted, survives a
    /// later `log show`) without inflating `.error` metrics.
    private static func osType(_ l: Logging.Logger.Level) -> OSLogType {
        switch l {
        case .trace, .debug: return .debug
        case .info: return .info
        case .notice, .warning: return .default
        case .error: return .error
        case .critical: return .fault
        }
    }
}
