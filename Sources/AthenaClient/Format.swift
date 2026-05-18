import Foundation

/// Human-readable byte size (e.g. `4.4 GB`). Lives here so the
/// portable client's `ps` can format governor stats without dragging
/// in the daemon's model-store commands. (`athenad`'s `ListModels`
/// keeps its own copy — a 5-line pure formatter is cheaper duplicated
/// than coupled across the client/daemon boundary.)
public func humanBytes(_ n: Int) -> String {
    let u = ["B", "KB", "MB", "GB", "TB"]
    var v = Double(n)
    var i = 0
    while v >= 1024, i < u.count - 1 {
        v /= 1024
        i += 1
    }
    return String(format: i == 0 ? "%.0f %@" : "%.1f %@", v, u[i])
}
