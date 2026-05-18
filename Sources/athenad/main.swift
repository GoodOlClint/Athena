import Foundation

#if canImport(Darwin)
    import Darwin
#elseif canImport(Glibc)
    import Glibc
#endif

// `athenad` — the Athena daemon process (macOS-only). Users never type
// it: `athena start`/launchd spawn it. It is a thin launcher that
// `exec`s the sibling `athena` binary as `athena load <args>` —
// `execv` replaces this image in place, so the PID is preserved
// (launchd KeepAlive/RunAtLoad keep working) without duplicating the
// server bootstrap. M14.2d. (A real in-binary daemon awaits the
// Server library extraction around M16.)

let forwarded = Array(CommandLine.arguments.dropFirst())

/// The `athena` binary that carries the actual `load` server. Prefer
/// the sibling next to this `athenad` (the install/build layout);
/// fall back to PATH.
func siblingAthena() -> String? {
    let me = CommandLine.arguments.first ?? "athenad"
    let dir = (me as NSString).deletingLastPathComponent
    guard !dir.isEmpty else { return nil }
    let cand = dir + "/athena"
    return FileManager.default.isExecutableFile(atPath: cand)
        ? cand : nil
}

let argv = ["athena", "load"] + forwarded
var cargs: [UnsafeMutablePointer<CChar>?] = argv.map { strdup($0) }
cargs.append(nil)

if let path = siblingAthena() {
    execv(path, &cargs)  // returns only on failure
}
execvp("athena", &cargs)  // PATH fallback; returns only on failure
perror("athenad: could not exec the 'athena' daemon binary")
exit(127)
