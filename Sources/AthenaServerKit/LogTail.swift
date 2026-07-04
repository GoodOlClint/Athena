import Foundation

/// `athena logs` tail-N decision algebra (usability audit 2026-07-02 §1).
///
/// `/usr/bin/log show` emits **oldest-first**. The old `collectLogEntries`
/// stopped reading at `limit` entries, so a busy hour returned the *oldest*
/// `limit` and silently dropped the tail — the operator saw log entries that
/// were ~30 min stale. The fix is to drain the whole window and keep the
/// **newest** `limit`, flagging when anything was dropped.
///
/// This is the pure, MLX-free logic (ADR 008/009): a fixed-capacity buffer
/// that consumes an oldest-first stream and retains the last `limit` elements
/// in arrival order. Kept generic (element-agnostic) so it is unit-testable
/// under `swift test` without the daemon's `LogEntryDTO`.
public struct LogTail<Element> {
    private var buffer: [Element] = []
    private let limit: Int
    /// True once at least one element has been dropped off the front — i.e.
    /// the window held more than `limit` entries.
    public private(set) var truncated = false

    public init(limit: Int) {
        self.limit = max(1, limit)
        buffer.reserveCapacity(self.limit)
    }

    /// Append one entry, evicting the oldest if over capacity.
    public mutating func append(_ e: Element) {
        buffer.append(e)
        if buffer.count > limit {
            // ponytail: naive front-shift; a real ring buffer only if log
            // volume grows 100× (51k/window today shifts fine at limit≤5000).
            buffer.removeFirst(buffer.count - limit)
            truncated = true
        }
    }

    /// The retained newest `limit` entries, still oldest-first.
    public var entries: [Element] { buffer }
}
