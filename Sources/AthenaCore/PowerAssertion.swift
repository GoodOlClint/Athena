import Foundation

#if canImport(IOKit)
import IOKit.pwr_mgt
#endif

/// M60.2 — hold a macOS power assertion for the lifetime of the serving
/// daemon so an unattended Mac never enters system **idle sleep** and
/// SUSPENDS in-flight inference.
///
/// ## Why this exists (the root cause it fixes)
/// Investigated 2026-06-02: a downstream client saw the LLM "crater"/"freeze" on a
/// second back-to-back eval run left unattended overnight. It was NOT a
/// daemon bug, thermal throttle, or memory leak. Athena held **no** power
/// assertion, so when the laptop was left alone the display slept, `powerd`
/// dropped its built-in *"prevent sleep while display is on"* assertion, and
/// ~minutes later the Mac hit the **system-sleep idle timer** and suspended
/// the whole process mid-generation. Reproduced deterministically with
/// `pmset displaysleepnow`: decode held ~31 tok/s for ~3 min, then BOTH a 2 s
/// `/healthz` poller and an independent `powermetrics` logger showed an
/// identical **50 s gap** (full suspend), decode → 0, snapping back only when
/// the machine was woken. "Restart fixes it" was a red herring — restarting
/// meant *touching the machine*, which woke it.
///
/// ## Why `PreventUserIdleSystemSleep` specifically
/// This type lets the **display** sleep (the panel powers off, saving energy)
/// while keeping the **system** — and therefore the GPU and the decode loop —
/// fully alive. It blocks only the *idle* sleep timer, which is exactly the
/// unattended-batch case. It deliberately does NOT override an explicit
/// lid-close or an Apple-menu ▸ Sleep; those remain the operator's call. It is
/// the in-process equivalent of `caffeinate -s -i`.
///
/// Lock-guarded `@unchecked Sendable` (mirrors the codebase's `GDNRollback` /
/// rate-limit idiom) so `acquire`/`release` are safe from the serve lifecycle
/// and the `/healthz` reader. No-op (returns `false`) where IOKit is absent.
public final class PowerAssertion: @unchecked Sendable {
    private let lock = NSLock()
    private var assertionID: UInt32 = 0
    private var held = false
    /// Human-readable reason shown in `pmset -g assertions`.
    public let reason: String

    public init(reason: String) { self.reason = reason }

    /// Whether the assertion is currently held (surfaced on `/healthz`).
    public var isHeld: Bool {
        lock.lock(); defer { lock.unlock() }
        return held
    }

    /// Create the assertion. Idempotent; returns whether it is now held.
    /// Failure (rare — e.g. sandboxed without the entitlement) is non-fatal:
    /// serving continues, just without sleep protection, and `/healthz`
    /// reports `powerAssertionHeld: false` so an operator can see it.
    @discardableResult
    public func acquire() -> Bool {
        lock.lock(); defer { lock.unlock() }
        guard !held else { return true }
        #if canImport(IOKit)
        var aid: IOPMAssertionID = 0
        let rc = IOPMAssertionCreateWithName(
            kIOPMAssertionTypePreventUserIdleSystemSleep as CFString,
            IOPMAssertionLevel(kIOPMAssertionLevelOn),
            reason as CFString,
            &aid)
        if rc == kIOReturnSuccess {
            assertionID = aid
            held = true
        }
        return held
        #else
        return false
        #endif
    }

    /// Release the assertion. Idempotent.
    public func release() {
        lock.lock(); defer { lock.unlock() }
        guard held else { return }
        #if canImport(IOKit)
        IOPMAssertionRelease(assertionID)
        #endif
        held = false
        assertionID = 0
    }

    deinit { release() }
}
