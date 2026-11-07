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

    /// SemVer 2.0 precedence. The numeric core compares first (missing or
    /// non-numeric components count as 0); a prerelease sorts below the same
    /// core without one; prerelease identifiers compare part by part on `.`
    /// (numeric as numbers, otherwise as ASCII, numeric below text, a
    /// shorter prefix below a longer one). Build metadata (`+...`) is
    /// ignored. Returns -1 / 0 / 1.
    public static func compare(_ a: String, _ b: String) -> Int {
        let (coreA, preA) = split(a)
        let (coreB, preB) = split(b)
        for i in 0 ..< Swift.max(coreA.count, coreB.count) {
            let x = i < coreA.count ? coreA[i] : 0
            let y = i < coreB.count ? coreB[i] : 0
            if x != y { return x < y ? -1 : 1 }
        }
        switch (preA, preB) {
        case (nil, nil): return 0
        case (nil, _): return 1
        case (_, nil): return -1
        case let (pa?, pb?):
            for i in 0 ..< Swift.min(pa.count, pb.count) {
                let r = compareIdentifier(pa[i], pb[i])
                if r != 0 { return r }
            }
            if pa.count == pb.count { return 0 }
            return pa.count < pb.count ? -1 : 1
        }
    }

    private static func split(_ v: String) -> (core: [Int], pre: [Substring]?) {
        let noBuild = v.split(separator: "+", maxSplits: 1, omittingEmptySubsequences: false)[0]
        let parts = noBuild.split(separator: "-", maxSplits: 1, omittingEmptySubsequences: false)
        let core = parts[0].split(separator: ".").map { Int($0) ?? 0 }
        let pre = parts.count > 1 ? parts[1].split(separator: ".", omittingEmptySubsequences: false) : nil
        return (core, pre)
    }

    private static func compareIdentifier(_ a: Substring, _ b: Substring) -> Int {
        switch (Int(a), Int(b)) {
        case let (x?, y?): return x == y ? 0 : (x < y ? -1 : 1)
        case (_?, nil): return -1
        case (nil, _?): return 1
        case (nil, nil): return a == b ? 0 : (a < b ? -1 : 1)
        }
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
