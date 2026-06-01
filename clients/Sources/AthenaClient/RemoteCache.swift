import ArgumentParser
import Foundation

#if canImport(FoundationNetworking)
    import FoundationNetworking
#endif

// M59.4 — operator view of the cross-request prompt-prefix KV pool. The
// pool is in-process state on the daemon (no local store to read), so this
// ALWAYS goes over HTTP — to a loopback daemon or an off-box one alike. The
// endpoint (`/api/cache/prompt`) is admin-only (daemon.admin). `stats` is a
// pull-only read; `flush` drops every cached prefix not held by an in-flight
// generation and is audited daemon-side. Pull only — the passive oracle
// never pushes the pool out.
public enum RemoteCache {
    struct Stats: Decodable {
        let enabled: Bool
        let entries: Int
        let bytes: Int
        let hits: Int
        let misses: Int
        let evictions: Int
        let max_entries: Int
        let max_bytes: Int
    }
    struct Flush: Decodable {
        let flushed: Int
        let entries: Int
        let bytes: Int
    }

    static func gib(_ b: Int) -> String {
        b >= 1_000_000_000
            ? String(format: "%.2f GB", Double(b) / 1e9)
            : String(format: "%.0f MB", Double(b) / 1e6)
    }

    public static func stats(_ d: DaemonOptions) async throws {
        let (code, data): (Int, Data)
        do {
            (code, data) = try await HTTPClient.send(
                "GET", d.base + "/api/cache/prompt", key: d.authKey)
        } catch { HTTPClient.noDaemon(d, error) }
        guard code < 400,
            let s = try? JSONDecoder().decode(Stats.self, from: data)
        else {
            HTTPClient.printJSON(data)
            if code >= 400 { throw ExitCode.failure }
            return
        }
        guard s.enabled else {
            print("prompt cache: disabled ([prompt_cache] off)")
            return
        }
        let total = s.hits + s.misses
        let rate =
            total > 0
            ? String(format: "%.0f%%", Double(s.hits) / Double(total) * 100)
            : "n/a"
        let capBytes = s.max_bytes > 0 ? gib(s.max_bytes) : "governor"
        print("prompt cache: enabled")
        print(
            "  entries   \(s.entries) / \(s.max_entries)   "
                + "bytes \(gib(s.bytes)) / \(capBytes)")
        print(
            "  hits \(s.hits)  misses \(s.misses)  "
                + "hit-rate \(rate)  evictions \(s.evictions)")
    }

    public static func flush(_ d: DaemonOptions) async throws {
        let (code, data): (Int, Data)
        do {
            (code, data) = try await HTTPClient.send(
                "DELETE", d.base + "/api/cache/prompt", key: d.authKey)
        } catch { HTTPClient.noDaemon(d, error) }
        guard code < 400,
            let f = try? JSONDecoder().decode(Flush.self, from: data)
        else {
            HTTPClient.printJSON(data)
            if code >= 400 { throw ExitCode.failure }
            return
        }
        print(
            "flushed \(f.flushed) prefix entr\(f.flushed == 1 ? "y" : "ies"); "
                + "\(f.entries) still resident (\(gib(f.bytes)))")
    }
}

// MARK: - Command

/// `athena cache prompt` — inspect or flush the prompt-prefix KV pool.
/// Remote-capable (and loopback-capable) like the other unified verbs;
/// always over HTTP since the pool is in-daemon state.
public struct CacheCmd: AsyncParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "cache",
        abstract: "Inspect or flush the prompt-prefix KV cache.",
        subcommands: [PromptStats.self, PromptFlush.self],
        defaultSubcommand: PromptStats.self)
    public init() {}

    public struct PromptStats: AsyncParsableCommand {
        public static let configuration = CommandConfiguration(
            commandName: "prompt",
            abstract: "Show prompt-prefix cache stats (entries, bytes, hits).")
        @OptionGroup public var daemon: DaemonOptions
        public init() {}
        public func run() async throws { try await RemoteCache.stats(daemon) }
    }

    public struct PromptFlush: AsyncParsableCommand {
        public static let configuration = CommandConfiguration(
            commandName: "flush",
            abstract: "Flush the prompt-prefix cache (admin-only, audited).")
        @OptionGroup public var daemon: DaemonOptions
        public init() {}
        public func run() async throws { try await RemoteCache.flush(daemon) }
    }
}
