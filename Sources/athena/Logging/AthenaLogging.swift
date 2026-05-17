import AthenaCore
import Darwin
import Foundation
import Logging
import os

// Centralized logging (M10). Athena's logs go to BOTH stdout (so
// launchd/`athena start` file capture and `athena logs` keep working)
// AND the macOS unified logging system (Console.app / `log stream` /
// `log show`), split by category so the daemon and each model can be
// filtered apart. Dependency-free: a small swift-log → os.Logger
// bridge rather than a third-party backend.

enum AthenaLog {
    /// Unified-logging subsystem. Simple, not reverse-DNS (Apple
    /// allows any string; this matches the binary/command name).
    static let subsystem = "athena"

    /// swift-log label for daemon/server logs. The label convention
    /// itself lives in AthenaCore (`AthenaLogLabel`, shared with the
    /// governor seam, dependency-free); the os.Logger `category` is
    /// derived by stripping the `athena.` prefix — anything else
    /// (Hummingbird's serverName "athena", NIO, etc.) ⇒ `daemon`.
    static let daemonLabel = AthenaLogLabel.daemon

    static func category(forLabel label: String) -> String {
        let p = "athena."
        guard label.hasPrefix(p), label.count > p.count else {
            return "daemon"
        }
        return String(label.dropFirst(p.count))
    }

    /// Parse a log-level string (case-insensitive). nil/invalid ⇒ nil
    /// so the caller can decide the default (and warn on invalid).
    static func level(_ s: String?) -> Logging.Logger.Level? {
        s.flatMap {
            Logging.Logger.Level(rawValue: $0.lowercased())
        }
    }

    /// Bootstrap once, before any `Logger` is created. Multiplexes the
    /// stdout handler with the unified-logging bridge; if `syslogRemote`
    /// is set (the opt-in passive-oracle exception — logs only), a
    /// third UDP RFC5424 sink is added. `level` gates ALL sinks —
    /// `debug`/`trace` = max.
    static func bootstrap(
        level: Logging.Logger.Level = .info,
        syslogRemote: String? = nil
    ) {
        // One shared UDP socket reused across every per-label handler.
        let sender = syslogRemote.flatMap { SyslogSender(endpoint: $0) }
        if syslogRemote != nil, sender == nil {
            let warn =
                "warning: ignoring --syslog-remote (only "
                + "udp://host[:port] is supported)\n"
            FileHandle.standardError.write(Data(warn.utf8))
        }
        LoggingSystem.bootstrap { label in
            var stream = StreamLogHandler.standardOutput(label: label)
            stream.logLevel = level
            var unified = OSUnifiedLogHandler(label: label)
            unified.logLevel = level
            var handlers: [any LogHandler] = [stream, unified]
            if let sender {
                var sys = SyslogLogHandler(
                    sender: sender,
                    category: category(forLabel: label))
                sys.logLevel = level
                handlers.append(sys)
            }
            return MultiplexLogHandler(handlers)
        }
    }
}

/// Owns one connectionless UDP socket to a remote syslog collector.
/// Fire-and-forget: a send failure is silently dropped — logging must
/// never block or crash the passive oracle. Shared across all
/// per-label handlers. This is THE single sanctioned outbound path
/// (logs only); see the passive-oracle logging carve-out.
final class SyslogSender: @unchecked Sendable {
    private let fd: Int32
    private var addr: sockaddr_storage
    private let addrLen: socklen_t

    /// Accepts `udp://host[:port]` or bare `host[:port]` (default
    /// 514). `tcp://`/anything else ⇒ nil (UDP-only for now).
    init?(endpoint raw: String) {
        var s = raw
        if s.hasPrefix("udp://") { s = String(s.dropFirst(6)) }
        else if s.contains("://") { return nil }
        let host: String
        let port: String
        if let colon = s.lastIndex(of: ":"), !s.hasSuffix(":") {
            host = String(s[s.startIndex..<colon])
            port = String(s[s.index(after: colon)...])
        } else {
            host = s
            port = "514"
        }
        guard !host.isEmpty else { return nil }

        var hints = addrinfo()
        hints.ai_family = AF_UNSPEC
        hints.ai_socktype = SOCK_DGRAM
        var res: UnsafeMutablePointer<addrinfo>?
        guard getaddrinfo(host, port, &hints, &res) == 0,
            let info = res
        else { return nil }
        defer { freeaddrinfo(res) }
        let sock = socket(
            info.pointee.ai_family, info.pointee.ai_socktype,
            info.pointee.ai_protocol)
        guard sock >= 0 else { return nil }
        var storage = sockaddr_storage()
        memcpy(
            &storage, info.pointee.ai_addr,
            Int(info.pointee.ai_addrlen))
        self.fd = sock
        self.addr = storage
        self.addrLen = info.pointee.ai_addrlen
    }

    deinit { close(fd) }

    func send(_ datagram: String) {
        let bytes = Array(datagram.utf8)
        _ = withUnsafePointer(to: &addr) {
            $0.withMemoryRebound(
                to: sockaddr.self, capacity: 1
            ) { sa in
                bytes.withUnsafeBytes { buf in
                    sendto(
                        fd, buf.baseAddress, buf.count, 0, sa,
                        addrLen)
                }
            }
        }
    }
}

/// swift-log `LogHandler` that emits RFC5424 over the shared UDP
/// `SyslogSender`. facility = local0 (16); the os-log category
/// becomes the syslog MSGID for server-side filtering.
struct SyslogLogHandler: LogHandler {
    let sender: SyslogSender
    let category: String
    var logLevel: Logging.Logger.Level = .info
    var metadata: Logging.Logger.Metadata = [:]
    var metadataProvider: Logging.Logger.MetadataProvider?

    private static let host = ProcessInfo.processInfo.hostName
    private static let pid = String(
        ProcessInfo.processInfo.processIdentifier)
    // ISO8601DateFormatter.string(from:) is documented thread-safe;
    // shared read-only formatting use only.
    nonisolated(unsafe) private static let ts: ISO8601DateFormatter = {
        let f = ISO8601DateFormatter()
        f.formatOptions = [
            .withInternetDateTime, .withFractionalSeconds,
        ]
        return f
    }()

    subscript(metadataKey key: String) -> Logging.Logger.Metadata.Value?
    {
        get { metadata[key] }
        set { metadata[key] = newValue }
    }

    /// RFC5424 severity (0..7); facility local0 (16) ⇒ PRI = 128 + sev.
    private static func severity(_ l: Logging.Logger.Level) -> Int {
        switch l {
        case .critical: return 2
        case .error: return 3
        case .warning: return 4
        case .notice: return 5
        case .info: return 6
        case .trace, .debug: return 7
        }
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
        var text = message.description
        if !merged.isEmpty {
            text +=
                " "
                + merged.sorted { $0.key < $1.key }
                .map { "\($0.key)=\($0.value)" }
                .joined(separator: " ")
        }
        let pri = 128 + Self.severity(level)  // local0
        let stamp = Self.ts.string(from: Date())
        let msgid = category.isEmpty ? "-" : category
        let line =
            "<\(pri)>1 \(stamp) \(Self.host) athena \(Self.pid) "
            + "\(msgid) - \(text)"
        sender.send(line)
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

        var text = message.description
        if !merged.isEmpty {
            let pairs = merged.sorted { $0.key < $1.key }
                .map { "\($0.key)=\($0.value)" }
                .joined(separator: " ")
            text += " {\(pairs)}"
        }
        // Server logs are not sensitive — keep them readable in
        // Console rather than redacted as <private>.
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
