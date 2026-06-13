import Foundation

/// Pure CLI input parsing/validation helpers, relocated from the `athena`
/// executable (`Commands/AuthCmd.swift`, `Commands/DaemonLifecycle.swift`)
/// into the MLX-free `AthenaDeploy` so they are unit-testable under
/// `swift test` (NB4 / M70.1b — ADR 008 follow-on). Both are pure functions
/// over their input; the executable call sites are unchanged (it imports
/// `AthenaDeploy`).

/// Parse a human lifetime — `30d` / `12h` / `90m` / `3600s`, or a bare
/// integer (seconds) — into seconds. Returns nil on empty / non-positive /
/// malformed / overflowing input. Used by `auth token` `--ttl`.
public func parseTTLSeconds(_ s: String) -> Int? {
    let t = s.trimmingCharacters(in: .whitespaces)
    guard !t.isEmpty else { return nil }
    let mult: Int
    let numPart: Substring
    switch t.last {
    case "s": mult = 1; numPart = t.dropLast()
    case "m": mult = 60; numPart = t.dropLast()
    case "h": mult = 3600; numPart = t.dropLast()
    case "d": mult = 86400; numPart = t.dropLast()
    default: mult = 1; numPart = t[...]  // bare integer ⇒ seconds
    }
    guard let n = Int(numPart), n > 0 else { return nil }
    let (secs, overflow) = n.multipliedReportingOverflow(by: mult)
    guard !overflow else { return nil }
    return secs
}

/// A safe launchd/daemon `--label`: non-empty, ≤255 chars, and only
/// `[A-Za-z0-9._-]`. NB1 (M66.3) validates this UP FRONT in `Start.run`
/// before the euid branch, so a malformed label can't skip the M43.1
/// root-daemon hard-fail.
public func isValidLabel(_ s: String) -> Bool {
    guard !s.isEmpty, s.count <= 255 else { return false }
    let allowed = Set(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-_")
    return s.allSatisfy { allowed.contains($0) }
}
