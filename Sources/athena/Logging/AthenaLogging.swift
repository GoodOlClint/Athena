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

    /// swift-log labels we tag loggers with. The os.Logger `category`
    /// is derived by stripping the `athena.` prefix; anything else
    /// (Hummingbird's serverName "athena", NIO, etc.) ⇒ `daemon`.
    static let daemonLabel = "athena.daemon"
    static func modelLabel(_ module: String) -> String {
        "athena.model.\(module)"
    }

    static func category(forLabel label: String) -> String {
        let p = "athena."
        guard label.hasPrefix(p), label.count > p.count else {
            return "daemon"
        }
        return String(label.dropFirst(p.count))
    }

    /// Bootstrap once, before any `Logger` is created. Multiplexes the
    /// standard stdout handler with the unified-logging bridge.
    static func bootstrap() {
        LoggingSystem.bootstrap { label in
            MultiplexLogHandler([
                StreamLogHandler.standardOutput(label: label),
                OSUnifiedLogHandler(label: label),
            ])
        }
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
