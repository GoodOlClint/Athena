import Foundation

/// ADR 030 Part 2 (WP2) — the process-global "last recognized MLX allocation
/// fault" slot.
///
/// A `[metal::malloc]` / "maximum allowed buffer size" fault fires on MLX's own
/// `default-qos.cooperative` worker thread and reaches only the global
/// `MLX.setErrorHandler` (never a Swift `catch`). Historically that handler
/// re-`fatalError`ed, killing the daemon for every in-flight tenant. WP2 instead
/// **records** a recognized allocation fault here and returns from the handler,
/// keeping the process alive; the offending tenant's gated execution span
/// (`InferenceGate.withExclusiveExecution`) observes the latch on exit and
/// converts it to a classified `metalOutOfMemory` (→ 503), so one oversized
/// request degrades instead of aborting the daemon.
///
/// Why the InferenceGate is the correct consumer: ADR 029 guarantees exactly one
/// Metal-executing tenant at a time, so a fault recorded during a gated span is
/// unambiguously attributable to that span. The gate clears the latch on entry
/// (fresh slate) and takes it on exit, so a fault never leaks across requests.
///
/// MLX-free (a plain lock-guarded string) + unit-pinned (ADR 008/009): the
/// record/take/clear algebra is tested without a Metal device.
public final class MetalFaultLatch: @unchecked Sendable {
    public static let shared = MetalFaultLatch()

    private let lock = NSLock()
    private var message: String?

    public init() {}

    /// Record a recognized allocation fault. First writer wins — a cascade of
    /// follow-on faults (from continuing to touch the invalid arrays before the
    /// decode loop breaks) must not clobber the original cause.
    public func record(_ message: String) {
        lock.withLock { if self.message == nil { self.message = message } }
    }

    /// True iff a fault is currently latched. Polled by the decode loops (via
    /// `DecodeLoopControl`) so they stop submitting work on a faulted device.
    public var isSet: Bool { lock.withLock { message != nil } }

    /// Consume and clear the latched fault, if any.
    public func take() -> String? {
        lock.withLock {
            let m = message
            message = nil
            return m
        }
    }

    /// Clear without reading — called on entry to a gated span for a fresh slate.
    public func clear() { lock.withLock { message = nil } }
}

/// ADR 030 Part 2 (WP2) — default-on revert knob for the degrade behavior. When
/// `false`, the global error handler re-`fatalError`s on ANY fault (the pre-WP2
/// behavior). Write-once at daemon boot (a plain global, like
/// `InferenceGate.enabled` / `GovernorMemory.serveCacheBounded`).
public enum MetalFaultDegrade {
    public nonisolated(unsafe) static var enabled = true
}
