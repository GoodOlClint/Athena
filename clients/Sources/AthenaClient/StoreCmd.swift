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

/// A fast liveness probe so `import` can refuse while a daemon holds
/// the DB open. Short timeout — absence is the common, fast path.
private func daemonReachable(_ d: DaemonOptions) async -> Bool {
    var req = URLRequest(
        url: URL(string: d.base + "/v1/store/stats")!)
    req.timeoutInterval = 1.5
    return (try? await URLSession.shared.data(for: req)) != nil
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
    @Argument(help: "Snapshot file to import.") public var path:
        String
    public init() {}

    public func run() async throws {
        if await daemonReachable(daemon) {
            FailableExit.die(
                "error: a daemon is running at "
                    + "\(daemon.host):\(daemon.port) — stop it "
                    + "before import (swapping the DB under a live "
                    + "daemon corrupts it)")
        }
        let src = URL(
            fileURLWithPath: (path as NSString)
                .expandingTildeInPath)
        let fm = FileManager.default
        guard fm.fileExists(atPath: src.path) else {
            FailableExit.die("error: no such file: \(src.path)")
        }
        let dest = storeFile(dataDir)
        do {
            try fm.createDirectory(
                at: dest.deletingLastPathComponent(),
                withIntermediateDirectories: true)
            for ext in ["", "wal", "shm"] {
                let u = ext.isEmpty
                    ? dest : dest.appendingPathExtension(ext)
                if fm.fileExists(atPath: u.path) {
                    try fm.removeItem(at: u)
                }
            }
            try fm.copyItem(at: src, to: dest)
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
