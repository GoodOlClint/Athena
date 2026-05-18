import Foundation

#if canImport(Darwin)
    import Darwin
#endif

/// Tiny in-place stderr progress line for long model downloads.
/// Redraws on a TTY (carriage return); stays silent on a pipe/log
/// (the command prints its own start/end lines, so machine output
/// isn't polluted with `\r` spam). Calls arrive serialized on the
/// MainActor (the HF progress handler is `@MainActor`-dispatched),
/// so `@unchecked Sendable` is safe here.
final class ProgressBar: @unchecked Sendable {
    private let label: String
    private let isTTY: Bool
    private var lastPct = -1

    init(_ label: String) {
        self.label = label
        self.isTTY = isatty(2) != 0  // fd 2 = stderr
    }

    /// Bar string for a clamped percent — pure, unit-tested.
    static func bar(pct: Int) -> String {
        let p = max(0, min(100, pct))
        let filled = p / 5  // 20 cells, 5% each
        return String(repeating: "#", count: filled)
            + String(repeating: "-", count: 20 - filled)
    }

    func update(_ fraction: Double) {
        guard isTTY, fraction.isFinite else { return }
        let pct = max(0, min(100, Int(fraction * 100)))
        guard pct != lastPct else { return }  // throttle: 1%/redraw
        lastPct = pct
        FileHandle.standardError.write(
            Data("\r\(label) [\(Self.bar(pct: pct))] \(pct)%".utf8))
    }

    /// Terminate the in-place line so the next print starts clean.
    func finish() {
        guard isTTY, lastPct >= 0 else { return }
        FileHandle.standardError.write(Data("\n".utf8))
    }
}
