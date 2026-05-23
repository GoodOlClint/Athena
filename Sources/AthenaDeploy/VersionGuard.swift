import Foundation

/// Install-time version comparison (M38). Pure + dependency-free so it's
/// unit-testable; `athena install` does the marker-file I/O around it and
/// surfaces the transition (fresh / reinstall / upgrade / DOWNGRADE) so an
/// operator can't silently roll the appliance back to an older build.
public enum VersionGuard {
    public enum Transition: String, Sendable {
        case fresh  // no prior install recorded
        case reinstall  // same version
        case upgrade  // newer than the installed one
        case downgrade  // OLDER than installed — can reintroduce bugs
    }

    /// Compare dotted numeric versions (e.g. "0.10.23"). Missing or
    /// non-numeric components count as 0. Returns -1 / 0 / 1.
    public static func compare(_ a: String, _ b: String) -> Int {
        let pa = a.split(separator: ".").map { Int($0) ?? 0 }
        let pb = b.split(separator: ".").map { Int($0) ?? 0 }
        for i in 0..<Swift.max(pa.count, pb.count) {
            let x = i < pa.count ? pa[i] : 0
            let y = i < pb.count ? pb[i] : 0
            if x != y { return x < y ? -1 : 1 }
        }
        return 0
    }

    public static func classify(
        from previous: String?, to current: String
    ) -> Transition {
        guard let previous,
            !previous.trimmingCharacters(in: .whitespacesAndNewlines)
                .isEmpty
        else { return .fresh }
        switch compare(previous, current) {
        case 0: return .reinstall
        case let n where n < 0: return .upgrade  // previous < current
        default: return .downgrade
        }
    }

    /// One-line human summary for `athena install` output.
    public static func summary(
        from previous: String?, to current: String
    ) -> String {
        switch classify(from: previous, to: current) {
        case .fresh:
            return "version: installing \(current) (fresh install)"
        case .reinstall:
            return "version: reinstalling \(current) (unchanged)"
        case .upgrade:
            return "version: upgrading \(previous ?? "?") -> \(current)"
        case .downgrade:
            return "version: DOWNGRADING \(previous ?? "?") -> \(current)"
        }
    }
}
