import ArgumentParser
import Foundation

#if canImport(FoundationNetworking)
    import FoundationNetworking
#endif

// M42.2 — remote driver for the persistent /api/models/allow surface.
// The on-daemon SQLite table is the source of truth; this thin client
// just maps each subcommand to one HTTP call. The portable `athena`
// (Linux/Windows) and the macOS unified `athena` both wire these in;
// the off-box behavior is identical (model.read / model.write are the
// server-side gates).

public enum RemoteAllowlist {
    private struct Entry: Decodable {
        let module: String
        let id: String
        let `default`: Bool
        let declared: Double
    }
    private struct ListResp: Decodable { let allowlist: [Entry] }
    private struct MutateResp: Decodable {
        let module: String
        let id: String
        let status: String
    }

    private static func enc(_ s: String) -> String {
        s.addingPercentEncoding(
            withAllowedCharacters: .urlQueryAllowed) ?? s
    }

    private static func fail(_ code: Int, _ data: Data) throws {
        HTTPClient.printJSON(data)
        if code >= 400 { throw ExitCode.failure }
    }

    public static func list(
        _ d: DaemonOptions, module: String?
    ) async throws {
        var url = d.base + "/api/models/allow"
        if let m = module, !m.isEmpty {
            url += "?module=" + enc(m)
        }
        let (code, data): (Int, Data)
        do {
            (code, data) = try await HTTPClient.send(
                "GET", url, key: d.authKey)
        } catch { HTTPClient.noDaemon(d, error) }
        guard code < 400,
            let r = try? JSONDecoder().decode(
                ListResp.self, from: data)
        else { return try fail(code, data) }
        renderList(
            r.allowlist.map {
                ($0.module, $0.id, $0.default)
            })
    }

    /// Shared table renderer so the HTTP (`list`) and offline
    /// (`--data-dir`) paths print byte-identical output (#11 / M43.5).
    public static func renderList(
        _ rows: [(module: String, id: String, isDefault: Bool)]
    ) {
        guard !rows.isEmpty else {
            print("no allowlist entries")
            return
        }
        print(
            "MODULE".padding(toLength: 20, withPad: " ", startingAt: 0)
                + "DEFAULT".padding(
                    toLength: 9, withPad: " ", startingAt: 0)
                + "ID")
        for e in rows {
            print(
                e.module.padding(
                    toLength: 20, withPad: " ", startingAt: 0)
                    + (e.isDefault ? "*" : "").padding(
                        toLength: 9, withPad: " ", startingAt: 0)
                    + e.id)
        }
    }

    public static func add(
        _ d: DaemonOptions, module: String, id: String,
        asDefault: Bool
    ) async throws {
        let body = try JSONSerialization.data(withJSONObject: [
            "module": module, "id": id, "default": asDefault,
        ])
        let (code, data): (Int, Data)
        do {
            (code, data) = try await HTTPClient.send(
                "POST", d.base + "/api/models/allow", body: body,
                key: d.authKey)
        } catch { HTTPClient.noDaemon(d, error) }
        guard code < 400,
            let r = try? JSONDecoder().decode(
                MutateResp.self, from: data)
        else { return try fail(code, data) }
        print("\(r.status) \(r.module):\(r.id)")
        // M43.2 usability #4 — warn when an LLM id isn't in the local
        // model store. Without this the operator's next inference would
        // surface as 503 module_loading (the M43.2 cold-load 503 is the
        // accurate signal, but only AFTER a request was attempted).
        // Best-effort: a 401/network failure is silent — the add itself
        // already succeeded.
        if module == "llm" {
            await warnIfMissingFromStore(d, id: id)
        }
    }

    private static func warnIfMissingFromStore(
        _ d: DaemonOptions, id: String
    ) async {
        struct Entry: Decodable { let name: String }
        struct Resp: Decodable { let models: [Entry] }
        let (code, data): (Int, Data)
        do {
            (code, data) = try await HTTPClient.send(
                "GET", d.base + "/api/models", key: d.authKey)
        } catch { return }
        guard code < 400,
            let r = try? JSONDecoder().decode(Resp.self, from: data)
        else { return }
        if r.models.contains(where: { $0.name == id }) { return }
        FileHandle.standardError.write(
            Data(
                ("warning: '\(id)' is not in the local model store; "
                    + "run `athena pull \(id)` to prefetch it, "
                    + "otherwise the next inference returns 503 "
                    + "module_loading while the daemon downloads.\n")
                    .utf8))
    }

    public static func remove(
        _ d: DaemonOptions, module: String, id: String
    ) async throws {
        let url =
            d.base + "/api/models/allow?module=" + enc(module)
            + "&id=" + enc(id)
        let (code, data): (Int, Data)
        do {
            (code, data) = try await HTTPClient.send(
                "DELETE", url, key: d.authKey)
        } catch { HTTPClient.noDaemon(d, error) }
        guard code < 400,
            let r = try? JSONDecoder().decode(
                MutateResp.self, from: data)
        else { return try fail(code, data) }
        print("\(r.status) \(r.module):\(r.id)")
    }

    public static func setDefault(
        _ d: DaemonOptions, module: String, id: String
    ) async throws {
        let body = try JSONSerialization.data(withJSONObject: [
            "module": module, "id": id,
        ])
        let (code, data): (Int, Data)
        do {
            (code, data) = try await HTTPClient.send(
                "PUT", d.base + "/api/models/allow/default",
                body: body, key: d.authKey)
        } catch { HTTPClient.noDaemon(d, error) }
        guard code < 400,
            let r = try? JSONDecoder().decode(
                MutateResp.self, from: data)
        else { return try fail(code, data) }
        print("\(r.status) \(r.module):\(r.id)")
    }
}

/// `athena allowlist …` — manage the persistent per-module
/// model allowlist (M42.2). Subcommand verbs: `list`, `add`, `rm`,
/// `default`. The server enforces RBAC (`model.read` for `list`,
/// `model.write` for the mutations); the client just routes.
public struct AllowlistCmd: AsyncParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "allowlist",
        abstract:
            "Manage the persistent per-module model allowlist.",
        subcommands: [
            AllowList.self, AllowAdd.self, AllowRm.self,
            AllowDefault.self,
        ])
    public init() {}
}

public struct AllowList: AsyncParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "list",
        abstract: "List allowlist entries (optionally one module).")
    @OptionGroup public var daemon: DaemonOptions
    @Option(help: "Narrow to one module (llm, textEmbedding, …).")
    public var module: String?
    public init() {}
    public func run() async throws {
        try await RemoteAllowlist.list(daemon, module: module)
    }
}

public struct AllowAdd: AsyncParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "add",
        abstract: "Add a model id to a module's allowlist.")
    @OptionGroup public var daemon: DaemonOptions
    @Option(help: "Module: llm, textEmbedding, transcription, diarization, speakerEmbedding.")
    public var module: String
    @Option(help: "Model id (HF id for aux modules, store name for llm).")
    public var id: String
    @Flag(
        name: .customLong("default"),
        help: "Mark this row as the module's default.")
    public var asDefault: Bool = false
    public init() {}
    public func run() async throws {
        try await RemoteAllowlist.add(
            daemon, module: module, id: id, asDefault: asDefault)
    }
}

public struct AllowRm: AsyncParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "rm",
        abstract: "Remove a model id from a module's allowlist.")
    @OptionGroup public var daemon: DaemonOptions
    @Option(help: "Module class.")
    public var module: String
    @Option(help: "Model id to remove.")
    public var id: String
    public init() {}
    public func run() async throws {
        try await RemoteAllowlist.remove(
            daemon, module: module, id: id)
    }
}

public struct AllowDefault: AsyncParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "default",
        abstract:
            "Mark an existing allowlist row as the module's default.")
    @OptionGroup public var daemon: DaemonOptions
    @Option(help: "Module class.")
    public var module: String
    @Option(help: "Model id to mark as default.")
    public var id: String
    public init() {}
    public func run() async throws {
        try await RemoteAllowlist.setDefault(
            daemon, module: module, id: id)
    }
}
