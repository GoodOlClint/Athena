import Foundation

#if canImport(Glibc)
    import Glibc
#endif

/// Ollama-style multi-row progress for `athena pull`/`convert` (usability audit
/// 2026-07-02 §2/§3). Foundation-only and compiled into both the portable and
/// macOS packages, so local (in-process) and remote (SSE) pulls render
/// identically. The pure state + line-builder (`ModelOpState`) is unit-pinned
/// (ADR 008/009); the `ModelOpRenderer` wrapper adds isatty-gated ANSI redraw
/// with a plain non-TTY fallback (no control chars).

/// A horizontal bar, e.g. `[####----]`, `width` cells.
public func renderBar(_ fraction: Double, width: Int = 20) -> String {
    let f = max(0, min(1, fraction))
    let filled = Int((Double(width) * f).rounded())
    return "[" + String(repeating: "#", count: filled)
        + String(repeating: "-", count: width - filled) + "]"
}

/// Pure render state — accumulates progress events and produces the lines to
/// draw. No I/O, so it is trivially unit-testable.
public struct ModelOpState {
    public struct FileRow: Equatable {
        public var name: String
        public var bytes: Int64
        public var total: Int64
        public var done: Bool
    }

    public let label: String
    public private(set) var order: [String] = []  // file names, first-seen order
    public private(set) var files: [String: FileRow] = [:]
    public private(set) var phase: String?
    public private(set) var aggFraction: Double = 0
    public private(set) var aggBytes: Int64 = 0
    public private(set) var aggTotal: Int64 = 0
    public private(set) var quantIndex: Int = 0
    public private(set) var quantCount: Int = 0

    public init(label: String) { self.label = label }

    public mutating func file(
        name: String, bytes: Int64, total: Int64, done: Bool
    ) {
        if files[name] == nil { order.append(name) }
        files[name] = FileRow(
            name: name, bytes: bytes, total: total, done: done)
    }
    public mutating func download(fraction: Double, bytes: Int64, total: Int64) {
        aggFraction = fraction
        aggBytes = bytes
        aggTotal = total
    }
    public mutating func phase(_ name: String) { phase = name }
    public mutating func quantize(index: Int, count: Int) {
        quantIndex = index
        quantCount = count
        phase = "quantize"
    }

    /// The lines to draw. In-progress file rows first (capped at `maxRows`, the
    /// rest collapsed into a "+N more" line), then a summary/total row. Pure.
    public func lines(maxRows: Int = 8) -> [String] {
        var out: [String] = []
        let active = order.compactMap { files[$0] }.filter { !$0.done }
        let doneCount = files.values.filter { $0.done }.count
        for row in active.prefix(maxRows) {
            let frac = row.total > 0 ? Double(row.bytes) / Double(row.total) : 0
            out.append(
                "  " + pad(row.name, 30) + " " + renderBar(frac, width: 16)
                    + " " + humanBytes(Int(row.bytes)) + "/"
                    + humanBytes(Int(row.total)))
        }
        if active.count > maxRows {
            out.append("  … +\(active.count - maxRows) more downloading")
        }
        // Summary / total row.
        if quantCount > 0 {
            let frac = Double(quantIndex) / Double(quantCount)
            out.append(
                "  " + pad("quantize", 30) + " " + renderBar(frac, width: 16)
                    + " \(quantIndex)/\(quantCount)")
        } else if !files.isEmpty || aggTotal > 0 {
            let label = phase ?? "total"
            var line =
                "  " + pad(label, 30) + " " + renderBar(aggFraction, width: 16)
                + " " + String(format: "%.0f%%", aggFraction * 100)
            if doneCount > 0 { line += "  (\(doneCount) done)" }
            out.append(line)
        } else if let phase {
            out.append("  \(phase) …")
        }
        return out
    }

    private func pad(_ s: String, _ n: Int) -> String {
        s.count >= n
            ? String(s.prefix(n - 1)) + "…"
            : s + String(repeating: " ", count: n - s.count)
    }
}

/// isatty-gated multi-row renderer. On a TTY it redraws in place (ANSI
/// cursor-up); off a TTY it prints a plain summary line on meaningful change
/// only (no control chars, grep-safe). Not Sendable — drive it from one task.
public final class ModelOpRenderer: @unchecked Sendable {
    private var state: ModelOpState
    private let isTTY: Bool
    private var drawnLines = 0
    private var lastPlain = ""
    private let lock = NSLock()

    public init(label: String, isTTY: Bool? = nil) {
        self.state = ModelOpState(label: label)
        self.isTTY = isTTY ?? (isatty(fileno(stdout)) == 1)
    }

    public func file(
        name: String, index: Int, count: Int, bytes: Int64, total: Int64,
        done: Bool
    ) {
        lock.lock()
        defer { lock.unlock() }
        state.file(name: name, bytes: bytes, total: total, done: done)
        draw()
    }
    public func download(fraction: Double, bytes: Int64, total: Int64) {
        lock.lock()
        defer { lock.unlock() }
        state.download(fraction: fraction, bytes: bytes, total: total)
        draw()
    }
    public func phase(_ name: String) {
        lock.lock()
        defer { lock.unlock() }
        state.phase(name)
        draw()
    }
    public func quantize(index: Int, count: Int) {
        lock.lock()
        defer { lock.unlock() }
        state.quantize(index: index, count: count)
        draw()
    }

    /// Caller holds `lock`.
    private func draw() {
        let lines = state.lines()
        if isTTY {
            var out = ""
            if drawnLines > 0 { out += "\u{1B}[\(drawnLines)A" }  // cursor up
            for l in lines {
                out += "\u{1B}[2K" + l + "\n"  // clear line + content
            }
            FileHandle.standardError.write(Data(out.utf8))
            drawnLines = lines.count
        } else {
            // Non-TTY: emit the last (summary) line only when it changes.
            guard let summary = lines.last, summary != lastPlain else { return }
            lastPlain = summary
            FileHandle.standardError.write(Data((summary + "\n").utf8))
        }
    }

    /// Final flush — leaves the completed rows on screen.
    public func finish() {
        lock.lock()
        defer { lock.unlock() }
        draw()
    }
}
