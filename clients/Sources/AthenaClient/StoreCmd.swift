import ArgumentParser
import Foundation

#if canImport(FoundationNetworking)
    import FoundationNetworking
#endif

/// `athena store …` — admin the shared SQLite store (M9.3). `export`
/// and `stats` talk to a running daemon; `import` is deliberately
/// offline-only (swapping the DB file under a live daemon corrupts it).
public struct Store: AsyncParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "store",
        abstract: "Export, import, and inspect the shared store.",
        subcommands: [
            StoreExport.self, StoreImport.self, StoreStats.self,
        ])
    public init() {}
}

/// Default store path mirrors `serve` (`<data-dir>/athena.sqlite`,
/// data-dir defaulting to `~/.athena`).
private func storeFile(_ dataDir: String?) -> URL {
    let root =
        dataDir.map {
            URL(
                fileURLWithPath: ($0 as NSString)
                    .expandingTildeInPath,
                isDirectory: true)
        }
        ?? FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".athena", isDirectory: true)
    return root.appendingPathComponent("athena.sqlite")
}

/// Result of the pre-import liveness probe (NK1). `reachable` = a daemon
/// answered (running — refuse); `absent` = the connection was actively
/// refused (nothing listening — safe to import); `unknown` = a timeout or
/// other ambiguous error where we CANNOT tell whether a (possibly busy)
/// daemon holds the DB — treat as refuse unless `--force`.
private enum DaemonProbe { case reachable, absent, unknown }

/// A fast liveness probe so `import` can refuse while a daemon holds the
/// DB open. Short timeout — absence is the common, fast path. NK1: a
/// timeout no longer reads as "absent" (which would let `import` clobber
/// the DB under a live-but-slow daemon); only an actively-refused
/// connection counts as absent.
private func probeDaemon(_ d: DaemonOptions) async -> DaemonProbe {
    var req = URLRequest(
        url: URL(string: d.base + "/v1/store/stats")!)
    req.timeoutInterval = 1.5
    do {
        _ = try await URLSession.shared.data(for: req)
        return .reachable  // any HTTP response ⇒ a daemon answered
    } catch let e as URLError {
        switch e.code {
        case .cannotConnectToHost, .cannotFindHost,
            .networkConnectionLost, .dnsLookupFailed:
            return .absent  // nothing is listening
        default:
            return .unknown  // timedOut et al. — can't tell
        }
    } catch {
        return .unknown
    }
}

public struct StoreExport: AsyncParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "export",
        abstract: "Snapshot the live store (VACUUM INTO) to a path.")
    @OptionGroup public var daemon: DaemonOptions
    @Argument(help: "Destination file (written by the daemon host).")
    public var path: String
    public init() {}

    public func run() async throws {
        guard
            let body = try? JSONSerialization.data(
                withJSONObject: ["path": path])
        else { FailableExit.die("error: could not encode request") }
        let (code, data): (Int, Data)
        do {
            (code, data) = try await HTTPClient.send(
                "POST", daemon.base + "/v1/store/export",
                body: body, key: daemon.authKey)
        } catch { HTTPClient.noDaemon(daemon, error) }
        HTTPClient.printJSON(data)
        if code >= 400 { throw ExitCode.failure }
    }
}

public struct StoreImport: AsyncParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "import",
        abstract:
            "Replace the local store from a snapshot (offline).")
    @OptionGroup public var daemon: DaemonOptions
    @Option(help: "Store data dir (default ~/.athena).")
    public var dataDir: String?
    @Flag(
        name: .long,
        help: "Import even if the daemon's liveness can't be determined (a probe timeout). Confirm no daemon is running first.")
    public var force = false
    @Argument(help: "Snapshot file to import.") public var path:
        String
    public init() {}

    public func run() async throws {
        // NK1: refuse against a running daemon; on an INDETERMINATE probe
        // (timeout) refuse unless --force, rather than assuming "absent"
        // and clobbering the DB under a live-but-slow daemon.
        switch await probeDaemon(daemon) {
        case .reachable:
            FailableExit.die(
                "error: a daemon is running at "
                    + "\(daemon.host):\(daemon.port) — stop it "
                    + "before import (swapping the DB under a live "
                    + "daemon corrupts it)")
        case .unknown where !force:
            FailableExit.die(
                "error: could not determine whether a daemon is running "
                    + "at \(daemon.host):\(daemon.port) (probe timed "
                    + "out). Stop any daemon and retry, or pass --force "
                    + "if you are certain none is running.")
        case .unknown, .absent:
            break
        }
        let src = URL(
            fileURLWithPath: (path as NSString)
                .expandingTildeInPath)
        let fm = FileManager.default
        guard fm.fileExists(atPath: src.path) else {
            FailableExit.die("error: no such file: \(src.path)")
        }
        let dest = storeFile(dataDir)
        // NK1: copy the snapshot to a temp file in the dest dir FIRST, then
        // atomically swap it in — the existing DB is never destroyed before
        // the copy fully succeeds. On a swap failure the original is
        // restored. Old WAL/SHM are removed only after the new DB is in
        // place (they belonged to the replaced database).
        let tmp = dest.appendingPathExtension("import-tmp")
        let bak = dest.appendingPathExtension("import-bak")
        do {
            try fm.createDirectory(
                at: dest.deletingLastPathComponent(),
                withIntermediateDirectories: true)
            try? fm.removeItem(at: tmp)
            try? fm.removeItem(at: bak)
            try fm.copyItem(at: src, to: tmp)  // original DB untouched here
            let destExists = fm.fileExists(atPath: dest.path)
            if destExists { try fm.moveItem(at: dest, to: bak) }
            do {
                try fm.moveItem(at: tmp, to: dest)
            } catch {
                if destExists { try? fm.moveItem(at: bak, to: dest) }
                try? fm.removeItem(at: tmp)
                throw error
            }
            for ext in ["wal", "shm"] {
                try? fm.removeItem(at: dest.appendingPathExtension(ext))
            }
            if destExists { try? fm.removeItem(at: bak) }
        } catch {
            FailableExit.die("error: import failed: \(error)")
        }
        print("imported \(src.path) → \(dest.path)")
    }
}

public struct StoreStats: AsyncParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "stats",
        abstract: "Vector + job counts and on-disk size.")
    @OptionGroup public var daemon: DaemonOptions
    public init() {}

    public func run() async throws {
        let (code, data): (Int, Data)
        do {
            (code, data) = try await HTTPClient.send(
                "GET", daemon.base + "/v1/store/stats",
                key: daemon.authKey)
        } catch { HTTPClient.noDaemon(daemon, error) }
        HTTPClient.printJSON(data)
        if code >= 400 { throw ExitCode.failure }
    }
}
