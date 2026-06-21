#if canImport(Darwin)
import Darwin
// `ptrace` is intentionally not surfaced by Swift's Darwin overlay; bind the
// libsystem C symbol directly so we can issue PT_DENY_ATTACH.
@_silgen_name("ptrace")
private func c_ptrace(
    _ request: Int32, _ pid: pid_t, _ addr: UnsafeMutableRawPointer?,
    _ data: Int32
) -> Int32
#endif
import Foundation

/// ADR 024 Tier 2 — side-channel hygiene applied at daemon startup. Shrinks the
/// plaintext footprint and closes cheap side channels around the resident
/// weights / KV cache / decoded request audio. MLX-free; pure libsystem.
///
/// Honesty boundary (ADR 024, binding): none of this defends the *live* working
/// set against a kernel/SIP-off adversary, and zeroize is best-effort (the page
/// may be reclaimed/handed back by an allocator before we touch it). These raise
/// the bar on the co-resident-software adversary on top of the Tier-1 process
/// lockdown; they are not a guarantee.
public enum ProcessHardening {

    /// `ptrace(PT_DENY_ATTACH)` request value (sys/ptrace.h); the C macro is not
    /// imported into Swift, so it is named here.
    private static let ptDenyAttach: Int32 = 31

    public struct Applied: Sendable {
        public var coreDumpsDisabled: Bool
        public var debuggerAttachDenied: Bool
    }

    /// Apply the always-on hygiene (no core dumps) plus the opt-in debugger
    /// denial. Returns what was applied so the caller can log it. Idempotent.
    @discardableResult
    public static func applyAtStartup(denyDebuggerAttach: Bool) -> Applied {
        let core = disableCoreDumps()
        let deny = denyDebuggerAttach ? denyDebuggerAttachNow() : false
        return Applied(coreDumpsDisabled: core, debuggerAttachDenied: deny)
    }

    /// `setrlimit(RLIMIT_CORE, 0)` — a crash can never dump the address space
    /// (resident KV cache / weights / decrypted secrets) to a core file. Always
    /// safe to apply; returns whether it took effect.
    @discardableResult
    public static func disableCoreDumps() -> Bool {
        #if canImport(Darwin)
        var lim = rlimit(rlim_cur: 0, rlim_max: 0)
        return setrlimit(RLIMIT_CORE, &lim) == 0
        #else
        return false
        #endif
    }

    /// `ptrace(PT_DENY_ATTACH)` — refuse debugger attach (and SIGKILL on a
    /// forced attach). Defense-in-depth atop the Tier-1 Hardened Runtime / no
    /// `get-task-allow` lockdown (which already denies the task port);
    /// kernel-bypassable, hence opt-in. Returns whether the call succeeded.
    @discardableResult
    public static func denyDebuggerAttachNow() -> Bool {
        #if canImport(Darwin)
        return c_ptrace(ptDenyAttach, 0, nil, 0) == 0
        #else
        return false
        #endif
    }

    /// Best-effort secure-zero of a buffer we own (e.g. decoded request PCM),
    /// using `memset_s` so the write is not optimized away. Honesty boundary
    /// (binding): best-effort, NOT a guarantee. It does not protect the buffer
    /// while it is in live use, nor pages an allocator already reclaimed; and
    /// because Swift arrays are copy-on-write, if the array is still **shared**
    /// at the call (e.g. boxed for an in-flight task) `withUnsafeMutableBytes`
    /// makes a unique copy first, so only the local copy is cleared — a no-op on
    /// the shared bytes. Effective in the common single-owner case (the unit
    /// test pins that), which is why callers zero the decoded PCM once it is no
    /// longer shared.
    public static func secureZero(_ array: inout [Float]) {
        guard !array.isEmpty else { return }
        array.withUnsafeMutableBytes { raw in
            guard let base = raw.baseAddress, raw.count > 0 else { return }
            #if canImport(Darwin)
            _ = memset_s(base, rsize_t(raw.count), 0, rsize_t(raw.count))
            #else
            memset(base, 0, raw.count)
            #endif
        }
    }

    /// Best-effort secure-zero of a `Data` buffer (e.g. an in-memory upload).
    public static func secureZero(_ data: inout Data) {
        guard !data.isEmpty else { return }
        data.withUnsafeMutableBytes { raw in
            guard let base = raw.baseAddress, raw.count > 0 else { return }
            #if canImport(Darwin)
            _ = memset_s(base, rsize_t(raw.count), 0, rsize_t(raw.count))
            #else
            memset(base, 0, raw.count)
            #endif
        }
    }
}
