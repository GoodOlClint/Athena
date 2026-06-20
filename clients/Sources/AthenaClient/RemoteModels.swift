import ArgumentParser
import Foundation

#if canImport(FoundationNetworking)
    import FoundationNetworking
#endif

// M17.1 — remote model management. The portable `athena` (Linux/
// Windows: NO local daemon) drives a daemon's M16 `/api/models`
// surface over HTTP; the bearer is `DaemonOptions.authKey`. The
// SERVER enforces RBAC (model.read / model.write) — the client adds
// NO trust, only surfaces the server's status faithfully. On macOS
// these same verbs run LOCALLY by default and route here when
// `--host` is off-box (`DaemonOptions.isRemote`); the standalone
// portable structs below ALWAYS go remote.

/// HTTP routines shared by the portable command structs (below) and
/// the macOS local verbs' off-box branch. Each prints the daemon's
/// response and throws `ExitCode.failure` on a `>= 400` so the
/// server's 400/403/404 RBAC/validation outcome is the client's exit
/// status too.
public enum RemoteModels {
    // Wire contracts — kept byte-aligned with Sources/athena/Server/
    // NativeAPIDTO.swift. Decoded locally (the thin-client pattern:
    // no shared DTO module).
    private struct Entry: Decodable {
        let name: String
        let bytes: Int
        let modified: String
    }
    private struct ListResp: Decodable { let models: [Entry] }
    private struct DetailResp: Decodable {
        let name: String
        let path: String
        let bytes: Int
    }
    private struct DefaultResp: Decodable {
        let model: String
        let source: String
    }
    private struct JobResp: Decodable {
        let job_id: String
        let status: String
    }

    private static func fail(_ code: Int, _ data: Data) throws {
        HTTPClient.printJSON(data)
        if code >= 400 { throw ExitCode.failure }
    }

    public static func list(_ d: DaemonOptions) async throws {
        let (code, data): (Int, Data)
        do {
            (code, data) = try await HTTPClient.send(
                "GET", d.base + "/api/models", key: d.authKey)
        } catch { HTTPClient.noDaemon(d, error) }
        guard code < 400,
            let r = try? JSONDecoder().decode(ListResp.self, from: data)
        else { return try fail(code, data) }
        guard !r.models.isEmpty else {
            print("no models")
            return
        }
        print(
            "NAME".padding(toLength: 40, withPad: " ", startingAt: 0)
                + "SIZE".padding(
                    toLength: 12, withPad: " ", startingAt: 0)
                + "MODIFIED")
        for m in r.models {
            print(
                m.name.padding(
                    toLength: 40, withPad: " ", startingAt: 0)
                    + humanBytes(m.bytes).padding(
                        toLength: 12, withPad: " ", startingAt: 0)
                    + m.modified)
        }
    }

    public static func show(_ d: DaemonOptions, name: String)
        async throws
    {
        let (code, data): (Int, Data)
        do {
            (code, data) = try await HTTPClient.send(
                "GET", d.base + "/api/models/\(enc(name))",
                key: d.authKey)
        } catch { HTTPClient.noDaemon(d, error) }
        // `config` is echoed verbatim; pretty-print the whole body so
        // the embedded model config.json is visible (parity with the
        // local `show`).
        try fail(code, data)
    }

    public static func remove(_ d: DaemonOptions, name: String)
        async throws
    {
        let (code, data): (Int, Data)
        do {
            (code, data) = try await HTTPClient.send(
                "DELETE", d.base + "/api/models/\(enc(name))",
                key: d.authKey)
        } catch { HTTPClient.noDaemon(d, error) }
        try fail(code, data)
    }

    public static func copy(
        _ d: DaemonOptions, src: String, dst: String,
        deepCopy: Bool, force: Bool
    ) async throws {
        let body = try JSONSerialization.data(withJSONObject: [
            "src": src, "dst": dst, "copy": deepCopy, "force": force,
        ])
        let (code, data): (Int, Data)
        do {
            (code, data) = try await HTTPClient.send(
                "POST", d.base + "/api/models/copy", body: body,
                key: d.authKey)
        } catch { HTTPClient.noDaemon(d, error) }
        try fail(code, data)
    }

    public static func getDefault(_ d: DaemonOptions) async throws {
        let (code, data): (Int, Data)
        do {
            (code, data) = try await HTTPClient.send(
                "GET", d.base + "/api/models/default", key: d.authKey)
        } catch { HTTPClient.noDaemon(d, error) }
        guard code < 400,
            let r = try? JSONDecoder().decode(
                DefaultResp.self, from: data)
        else { return try fail(code, data) }
        print("\(r.model)  (\(r.source))")
    }

    public static func setDefault(_ d: DaemonOptions, name: String)
        async throws
    {
        let body = try JSONSerialization.data(
            withJSONObject: ["name": name])
        let (code, data): (Int, Data)
        do {
            (code, data) = try await HTTPClient.send(
                "PUT", d.base + "/api/models/default", body: body,
                key: d.authKey)
        } catch { HTTPClient.noDaemon(d, error) }
        guard code < 400,
            let r = try? JSONDecoder().decode(
                DefaultResp.self, from: data)
        else { return try fail(code, data) }
        print("default model = \(r.model)  (\(r.source))")
    }

    // MARK: - Per-module model lifecycle (M41.1)

    private struct LoadReq: Encodable {
        let module: String
        let id: String?
    }
    private struct LoadResp: Decodable {
        let module: String
        let id: String
        let status: String
    }
    private struct UnloadReq: Encodable { let module: String? }
    private struct UnloadResp: Decodable {
        let modules: [String]
        let status: String
    }
    private struct SlotResp: Decodable {
        let module: String
        let allowed: [String]
        let `default`: String
        let resident: String?
    }
    private struct ResidentResp: Decodable { let slots: [SlotResp] }

    /// Rebind a module's slot to `id` (nil ⇒ the module's default).
    /// Server-side `model.write` gate, allowlist check, governor load
    /// + in-place rebind. Prints the daemon's reply and exits non-zero
    /// on `>= 400` (faithful surfacing of the server's outcome).
    public static func load(
        _ d: DaemonOptions, module: String, id: String?
    ) async throws {
        let body = try JSONEncoder().encode(
            LoadReq(module: module, id: id))
        let (code, data): (Int, Data)
        do {
            (code, data) = try await HTTPClient.send(
                "POST", d.base + "/api/models/load", body: body,
                key: d.authKey)
        } catch { HTTPClient.noDaemon(d, error) }
        guard code < 400,
            let r = try? JSONDecoder().decode(
                LoadResp.self, from: data)
        else { return try fail(code, data) }
        print("loaded \(r.module) ⇐ \(r.id)")
    }

    /// Release a module's slot (or every slot when `module` is nil /
    /// "all"). Same idempotent shape as the prior `athena unload` — the
    /// next inference reloads lazily.
    public static func unload(
        _ d: DaemonOptions, module: String?
    ) async throws {
        let body = try JSONEncoder().encode(
            UnloadReq(module: module))
        let (code, data): (Int, Data)
        do {
            (code, data) = try await HTTPClient.send(
                "POST", d.base + "/api/models/unload", body: body,
                key: d.authKey)
        } catch { HTTPClient.noDaemon(d, error) }
        guard code < 400,
            let r = try? JSONDecoder().decode(
                UnloadResp.self, from: data)
        else { return try fail(code, data) }
        print("unloaded \(r.modules.joined(separator: ", "))")
    }

    /// `GET /api/models/resident` — every module's allowlist + default
    /// + resident-id. Prints a compact table.
    public static func resident(_ d: DaemonOptions) async throws {
        let (code, data): (Int, Data)
        do {
            (code, data) = try await HTTPClient.send(
                "GET", d.base + "/api/models/resident",
                key: d.authKey)
        } catch { HTTPClient.noDaemon(d, error) }
        guard code < 400,
            let r = try? JSONDecoder().decode(
                ResidentResp.self, from: data)
        else { return try fail(code, data) }
        print(
            "MODULE".padding(
                toLength: 20, withPad: " ", startingAt: 0)
                + "RESIDENT".padding(
                    toLength: 40, withPad: " ", startingAt: 0)
                + "DEFAULT".padding(
                    toLength: 40, withPad: " ", startingAt: 0)
                + "ALLOWED")
        for s in r.slots {
            print(
                s.module.padding(
                    toLength: 20, withPad: " ", startingAt: 0)
                    + (s.resident ?? "-").padding(
                        toLength: 40, withPad: " ", startingAt: 0)
                    + s.default.padding(
                        toLength: 40, withPad: " ", startingAt: 0)
                    + s.allowed.joined(separator: ", "))
        }
    }

    /// Submit a long-running op (pull/convert/prune). The route is
    /// `model.write`-gated server-side; the reply is `202 {job_id}`.
    /// Then `--follow` streams `/v1/queue/:id` transitions, `--wait N`
    /// long-polls once, otherwise just the id is printed.
    public static func job(
        _ d: DaemonOptions, op: String, body: [String: Any],
        follow: Bool, wait: Int?
    ) async throws {
        let payload = try JSONSerialization.data(withJSONObject: body)
        let (code, data): (Int, Data)
        do {
            (code, data) = try await HTTPClient.send(
                "POST", d.base + "/api/models/\(op)", body: payload,
                key: d.authKey)
        } catch { HTTPClient.noDaemon(d, error) }
        guard code < 400,
            let j = try? JSONDecoder().decode(JobResp.self, from: data)
        else { return try fail(code, data) }
        print("\(op) job \(j.job_id) (\(j.status))")
        if follow {
            try await JobPoll.follow(d, id: j.job_id)
        } else if let wait {
            let (c2, d2): (Int, Data)
            do {
                (c2, d2) = try await HTTPClient.send(
                    "GET",
                    d.base + "/v1/queue/\(j.job_id)?wait=\(wait)",
                    key: d.authKey)
            } catch { HTTPClient.noDaemon(d, error) }
            try fail(c2, d2)
        } else {
            print("poll: athena queue get \(j.job_id) --follow")
        }
    }

    /// Percent-escape a path segment (model/job names are validated
    /// server-side; this just keeps an odd char from breaking the URL).
    private static func enc(_ s: String) -> String {
        s.addingPercentEncoding(
            withAllowedCharacters: .urlPathAllowed) ?? s
    }
}

/// Poll a queue job until terminal. Apple: native SSE
/// (`/v1/queue/:id/events`); Linux (FoundationNetworking lacks
/// `URLSession.bytes`): status-poll fallback — same observable
/// transitions. Mirrors `QueueGet`'s pattern, factored out so the
/// model-op verbs reuse it.
public enum JobPoll {
    private static let terminal: Set<String> = [
        "done", "error", "canceled",
    ]

    public static func follow(_ d: DaemonOptions, id: String)
        async throws
    {
        #if canImport(Darwin)
            var req = URLRequest(
                url: URL(
                    string: d.base + "/v1/queue/\(id)/events")!)
            if let k = d.authKey, !k.isEmpty {
                req.setValue(
                    "Bearer \(k)",
                    forHTTPHeaderField: "Authorization")
            }
            do {
                let (bytes, resp) = try await URLSession.shared.bytes(
                    for: req)
                // K2 — a non-200 (bad job id ⇒ 404, auth ⇒ 401/403) is
                // NOT an event stream; parsing it as one silently
                // swallows the real error. Check status before reading.
                if let http = resp as? HTTPURLResponse,
                    http.statusCode >= 400
                {
                    FailableExit.die(
                        "error: /v1/queue/\(id)/events → "
                            + "\(http.statusCode) (check the job id and "
                            + "auth)")
                }
                for try await line in bytes.lines
                where line.hasPrefix("data: ") {
                    let p = String(line.dropFirst(6))
                    if p == "[DONE]" { return }
                    print(p)
                }
            } catch { HTTPClient.noDaemon(d, error) }
        #else
            struct S: Decodable { let status: String }
            var last = ""
            for _ in 0..<600 {
                let (code, data): (Int, Data)
                do {
                    (code, data) = try await HTTPClient.send(
                        "GET", d.base + "/v1/queue/\(id)",
                        key: d.authKey)
                } catch { HTTPClient.noDaemon(d, error) }
                guard code < 400,
                    let s = try? JSONDecoder().decode(
                        S.self, from: data)
                else {
                    HTTPClient.printJSON(data)
                    if code >= 400 { throw ExitCode.failure }
                    return
                }
                if s.status != last {
                    HTTPClient.printJSON(data)
                    last = s.status
                }
                if terminal.contains(s.status) { return }
                try await Task.sleep(nanoseconds: 1_000_000_000)
            }
        #endif
    }
}

// MARK: - Portable command structs (remote-only)

// These ARE `pull`/`convert`/…/`list` on the portable client (no
// local daemon there). On macOS the SAME command names are the local
// verbs in Sources/athena/Commands, which delegate here when
// `--host` is off-box (the "overload" model). Distinct Swift type
// names so the monorepo `athena` target — which imports AthenaClient
// AND defines its own `Pull`/`Show`/… — has no ambiguous reference.

public struct PullCmd: AsyncParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "pull",
        abstract: "Download a model (HF id) into the daemon's store.")
    @OptionGroup public var daemon: DaemonOptions
    @Argument(help: "HF repo id.") public var model: String
    @Option(help: "Git revision / branch / tag.")
    public var revision: String?
    @Flag(help: "Stream job progress until it finishes.")
    public var follow = false
    @Option(help: "Long-poll up to N seconds for completion.")
    public var wait: Int?
    public init() {}
    public func run() async throws {
        var b: [String: Any] = ["id": model]
        if let revision { b["revision"] = revision }
        try await RemoteModels.job(
            daemon, op: "pull", body: b, follow: follow, wait: wait)
    }
}

public struct ConvertCmd: AsyncParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "convert",
        abstract: "Quantize an HF model into the daemon's store.")
    @OptionGroup public var daemon: DaemonOptions
    @Argument(help: "HF repo id.") public var model: String
    @Option(help: "Git revision / branch / tag.")
    public var revision: String?
    @Option(help: "Quantization bits.") public var qBits: Int?
    @Option(help: "Quantization group size.")
    public var qGroupSize: Int?
    @Option(help: "Output name in the store.")
    public var name: String?
    @Flag(help: "Stream job progress until it finishes.")
    public var follow = false
    @Option(help: "Long-poll up to N seconds for completion.")
    public var wait: Int?
    public init() {}
    public func run() async throws {
        var b: [String: Any] = ["id": model]
        if let revision { b["revision"] = revision }
        if let qBits { b["bits"] = qBits }
        if let qGroupSize { b["group_size"] = qGroupSize }
        if let name { b["name"] = name }
        try await RemoteModels.job(
            daemon, op: "convert", body: b, follow: follow,
            wait: wait)
    }
}

public struct PruneCmd: AsyncParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "prune",
        abstract: "Remove broken/dangling models from the store.")
    @OptionGroup public var daemon: DaemonOptions
    @Flag(help: "Show what would be removed; change nothing.")
    public var dryRun = false
    @Flag(help: "Stream job progress until it finishes.")
    public var follow = false
    @Option(help: "Long-poll up to N seconds for completion.")
    public var wait: Int?
    public init() {}
    public func run() async throws {
        try await RemoteModels.job(
            daemon, op: "prune", body: ["dry_run": dryRun],
            follow: follow, wait: wait)
    }
}

public struct CpCmd: AsyncParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "cp",
        abstract: "Alias or copy a stored model under a new name.")
    @OptionGroup public var daemon: DaemonOptions
    @Argument(help: "Source model name.") public var src: String
    @Argument(help: "Destination name.") public var dst: String
    @Flag(help: "Deep-copy instead of symlink-aliasing.")
    public var copy = false
    @Flag(help: "Overwrite an existing destination.")
    public var force = false
    public init() {}
    public func run() async throws {
        try await RemoteModels.copy(
            daemon, src: src, dst: dst, deepCopy: copy,
            force: force)
    }
}

public struct DefaultCmd: AsyncParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "default",
        abstract: "Show or set the daemon's default served model.")
    @OptionGroup public var daemon: DaemonOptions
    @Argument(help: "Model name to make default (omit to show).")
    public var name: String?
    public init() {}
    public func run() async throws {
        if let name {
            try await RemoteModels.setDefault(daemon, name: name)
        } else {
            try await RemoteModels.getDefault(daemon)
        }
    }
}

public struct RmCmd: AsyncParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "rm",
        abstract: "Remove a model from the daemon's store.")
    @OptionGroup public var daemon: DaemonOptions
    @Argument(help: "Model name.") public var model: String
    public init() {}
    public func run() async throws {
        try await RemoteModels.remove(daemon, name: model)
    }
}

public struct ShowCmd: AsyncParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "show",
        abstract: "Show a model's config and size.")
    @OptionGroup public var daemon: DaemonOptions
    @Argument(help: "Model name.") public var model: String
    public init() {}
    public func run() async throws {
        try await RemoteModels.show(daemon, name: model)
    }
}

/// `athena load --module M [--id I]` — rebind a module's slot on the
/// daemon. Portable + remote-only by definition (no local daemon in
/// the portable client; the macOS `Load` reuses this same wire format
/// via its --module/--id overload). M41.1.
public struct LoadCmd: AsyncParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "load",
        abstract:
            "Rebind a module's resident slot to a model id.")
    @OptionGroup public var daemon: DaemonOptions
    @Option(
        help:
            "Module class: llm, textEmbedding, transcription, diarization, speakerEmbedding."
    )
    public var module: String
    @Option(help: "Model id (omit ⇒ the module's default).")
    public var id: String?
    public init() {}
    public func run() async throws {
        try await RemoteModels.load(daemon, module: module, id: id)
    }
}

/// `athena resident` — every module's allowlist + default + resident
/// id (M41.1). Read-only model-store projection (`model.read`).
public struct ResidentCmd: AsyncParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "resident",
        abstract: "Show every module's resident model slot.")
    @OptionGroup public var daemon: DaemonOptions
    public init() {}
    public func run() async throws {
        try await RemoteModels.resident(daemon)
    }
}

public struct ListCmd: AsyncParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "list",
        abstract: "List models in the daemon's store.",
        aliases: ["ls"])
    @OptionGroup public var daemon: DaemonOptions
    public init() {}
    public func run() async throws {
        try await RemoteModels.list(daemon)
    }
}
