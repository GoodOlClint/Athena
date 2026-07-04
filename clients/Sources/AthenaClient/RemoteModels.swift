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
        // Typed listing (audit §4) — omitted by pre-typing daemons ⇒ nil.
        let modality: String?
        let engine: String?
        let loadability: String?
        let draft: Bool?
        let fused_mtp: Bool?
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
    private static func fail(_ code: Int, _ data: Data) throws {
        HTTPClient.printJSON(data)
        if code >= 400 { throw ExitCode.failure }
    }

    public static func list(_ d: DaemonOptions, type: String? = nil)
        async throws
    {
        let (code, data): (Int, Data)
        do {
            (code, data) = try await HTTPClient.send(
                "GET", d.base + "/api/models", key: d.authKey)
        } catch { HTTPClient.noDaemon(d, error) }
        guard code < 400,
            let r = try? JSONDecoder().decode(ListResp.self, from: data)
        else { return try fail(code, data) }
        var models = r.models
        if let type {
            models = models.filter {
                ModelTypeFormat.matches(
                    filter: type, modality: $0.modality, engine: $0.engine,
                    draft: $0.draft ?? false, fusedMTP: $0.fused_mtp ?? false)
            }
        }
        guard !models.isEmpty else {
            print(type == nil ? "no models" : "no models of type '\(type!)'")
            return
        }
        print(
            "NAME".padding(toLength: 40, withPad: " ", startingAt: 0)
                + "TYPE".padding(toLength: 14, withPad: " ", startingAt: 0)
                + "SIZE".padding(
                    toLength: 12, withPad: " ", startingAt: 0)
                + "MODIFIED")
        for m in models {
            let typeCol = ModelTypeFormat.column(
                modality: m.modality, engine: m.engine,
                draft: m.draft ?? false, fusedMTP: m.fused_mtp ?? false)
            print(
                m.name.padding(
                    toLength: 40, withPad: " ", startingAt: 0)
                    + typeCol.padding(
                        toLength: 14, withPad: " ", startingAt: 0)
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
    /// Server-side `model.write` gate, store-presence check, governor load
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

    /// `GET /api/models/resident` — every module's available models + default
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

    /// Submit a long-running op (pull/convert/prune). ADR 025 S2: the
    /// `model.write`-gated route runs the op SYNCHRONOUSLY and streams SSE
    /// progress (no async queue, no job id). We consume the stream to
    /// completion, blocking until the op finishes. `progress` (the old
    /// `--follow`) prints progress events as they arrive; either way the
    /// terminal `done` result (or `error` envelope) is printed, and an
    /// error exits non-zero.
    public static func job(
        _ d: DaemonOptions, op: String, body: [String: Any],
        progress: Bool
    ) async throws {
        let payload = try JSONSerialization.data(withJSONObject: body)
        try await ModelOpStream.run(
            d, op: op, body: payload, progress: progress)
    }

    /// Percent-escape a path segment (model/job names are validated
    /// server-side; this just keeps an odd char from breaking the URL).
    private static func enc(_ s: String) -> String {
        s.addingPercentEncoding(
            withAllowedCharacters: .urlPathAllowed) ?? s
    }
}

/// Consume a model-op SSE stream from `POST /api/models/{op}` (ADR 025
/// S2 — synchronous + SSE; no async queue, no job id). Apple: incremental
/// via `URLSession.bytes` (live progress). Linux (FoundationNetworking
/// lacks `URLSession.bytes`): the streaming body is buffered by
/// `URLSession.data` and parsed once the op finishes — same terminal
/// outcome, no mid-flight progress. Either way it BLOCKS until the op
/// completes; an `error` event exits non-zero.
public enum ModelOpStream {
    public static func run(
        _ d: DaemonOptions, op: String, body: Data, progress: Bool
    ) async throws {
        let url = d.base + "/api/models/\(op)"
        var sawError = false
        // Ollama-style renderer for `--follow` (audit §2/§3): one row per shard
        // + phase/quantize. Nil when not following (progress:false).
        let renderer = progress ? ModelOpRenderer(label: op) : nil
        #if canImport(Darwin)
            var req = URLRequest(url: URL(string: url)!)
            req.httpMethod = "POST"
            req.httpBody = body
            req.setValue(
                "application/json", forHTTPHeaderField: "Content-Type")
            if let k = d.authKey, !k.isEmpty {
                req.setValue(
                    "Bearer \(k)", forHTTPHeaderField: "Authorization")
            }
            do {
                let (bytes, resp) = try await URLSession.shared.bytes(
                    for: req)
                // A non-200 (bad body ⇒ 400, auth ⇒ 401/403) is NOT an
                // event stream — surface the real error, not a silent swallow.
                if let http = resp as? HTTPURLResponse,
                    http.statusCode >= 400
                {
                    var buf = Data()
                    for try await b in bytes { buf.append(b) }
                    HTTPClient.printJSON(buf)
                    throw ExitCode.failure
                }
                for try await line in bytes.lines
                where line.hasPrefix("data: ") {
                    if handleFrame(
                        String(line.dropFirst(6)), op: op,
                        renderer: renderer, sawError: &sawError)
                    { break }
                }
            } catch let e as ExitCode {
                throw e
            } catch { HTTPClient.noDaemon(d, error) }
        #else
            // KNOWN LIMITATION (audit §2, recorded not fixed): FoundationNetworking
            // has no `URLSession.bytes`, so the portable Linux/Windows client
            // buffers the ENTIRE SSE body before parsing — the multi-row progress
            // replays all at once at the end instead of live. Fixing it needs a
            // streaming HTTP client on non-Darwin; separate follow-up.
            let (code, data): (Int, Data)
            do {
                (code, data) = try await HTTPClient.send(
                    "POST", url, body: body, key: d.authKey)
            } catch { HTTPClient.noDaemon(d, error) }
            guard code < 400 else {
                HTTPClient.printJSON(data)
                throw ExitCode.failure
            }
            let text = String(data: data, encoding: .utf8) ?? ""
            for line in text.split(separator: "\n", omittingEmptySubsequences: true)
            where line.hasPrefix("data: ") {
                if handleFrame(
                    String(line.dropFirst(6)), op: op,
                    renderer: renderer, sawError: &sawError)
                { break }
            }
        #endif
        renderer?.finish()
        if sawError { throw ExitCode.failure }
    }

    /// Render one `data:` frame. Returns true on the terminal `[DONE]`
    /// sentinel. Sets `sawError` if the frame is an `error` event. New
    /// `file`/`phase`/`quantize` events feed the multi-row renderer; a daemon
    /// that only sends `progress` still drives the aggregate row (compat).
    private static func handleFrame(
        _ json: String, op: String, renderer: ModelOpRenderer?,
        sawError: inout Bool
    ) -> Bool {
        if json == "[DONE]" { return true }
        guard let data = json.data(using: .utf8),
            let obj = try? JSONSerialization.jsonObject(with: data)
                as? [String: Any],
            let event = obj["event"] as? String
        else { return false }
        func i64(_ k: String) -> Int64 {
            (obj[k] as? NSNumber)?.int64Value ?? 0
        }
        func int(_ k: String) -> Int { (obj[k] as? NSNumber)?.intValue ?? 0 }
        switch event {
        case "progress":
            renderer?.download(
                fraction: obj["fraction"] as? Double ?? 0,
                bytes: i64("bytes"), total: i64("total"))
        case "file":
            renderer?.file(
                name: obj["name"] as? String ?? "?", index: int("index"),
                count: int("count"), bytes: i64("bytes"), total: i64("total"),
                done: obj["done"] as? Bool ?? false)
        case "phase":
            renderer?.phase(obj["phase"] as? String ?? "?")
        case "quantize":
            renderer?.quantize(index: int("index"), count: int("count"))
        case "done":
            renderer?.finish()
            if renderer != nil {
                FileHandle.standardError.write(Data("\n".utf8))
            }
            if let result = obj["result"],
                let d = try? JSONSerialization.data(withJSONObject: result)
            {
                HTTPClient.printJSON(d)
            } else {
                print("\(op): done")
            }
        case "error":
            sawError = true
            let env: [String: Any] = [
                "error": obj["error"]
                    ?? ["message": "model op failed"]
            ]
            if let d = try? JSONSerialization.data(withJSONObject: env) {
                HTTPClient.printJSON(d)
            }
        default:
            break
        }
        return false
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
    @Flag(help: "Print download progress as it streams.")
    public var follow = false
    public init() {}
    public func run() async throws {
        var b: [String: Any] = ["id": model]
        if let revision { b["revision"] = revision }
        try await RemoteModels.job(
            daemon, op: "pull", body: b, progress: follow)
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
    @Flag(help: "Print download progress as it streams.")
    public var follow = false
    public init() {}
    public func run() async throws {
        var b: [String: Any] = ["id": model]
        if let revision { b["revision"] = revision }
        if let qBits { b["bits"] = qBits }
        if let qGroupSize { b["group_size"] = qGroupSize }
        if let name { b["name"] = name }
        try await RemoteModels.job(
            daemon, op: "convert", body: b, progress: follow)
    }
}

public struct PruneCmd: AsyncParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "prune",
        abstract: "Remove broken/dangling models from the store.")
    @OptionGroup public var daemon: DaemonOptions
    @Flag(help: "Show what would be removed; change nothing.")
    public var dryRun = false
    @Flag(help: "Print progress as it streams.")
    public var follow = false
    public init() {}
    public func run() async throws {
        try await RemoteModels.job(
            daemon, op: "prune", body: ["dry_run": dryRun],
            progress: follow)
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

/// `athena resident` — every module's available models + default + resident
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
    @Option(
        help:
            "Filter by TYPE, e.g. llm, draft, vision, embed, asr, diar, speaker, unsupported."
    )
    public var type: String?
    public init() {}
    public func run() async throws {
        try await RemoteModels.list(daemon, type: type)
    }
}
