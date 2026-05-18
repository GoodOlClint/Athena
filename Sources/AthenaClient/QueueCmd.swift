import ArgumentParser
import Foundation

/// `athena queue …` — manage the async request queue (M9.1). Thin
/// HTTP client to a running daemon's `/v1/queue` surface.
public struct Queue: AsyncParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "queue",
        abstract: "Inspect and manage the async request queue.",
        subcommands: [
            QueueLs.self, QueueGet.self, QueueSubmit.self,
            QueueRm.self,
        ])
    public init() {}
}

public struct QueueLs: AsyncParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "ls", abstract: "List queued/processed jobs.")
    @OptionGroup public var daemon: DaemonOptions
    @Option(help: "Filter by status (queued|running|done|error).")
    public var status: String?
    public init() {}

    public func run() async throws {
        let url =
            daemon.base + "/v1/queue"
            + (status.map { "?status=\($0)" } ?? "")
        let (_, data): (Int, Data)
        do {
            (_, data) = try await HTTPClient.send(
                "GET", url, key: daemon.authKey)
        } catch { HTTPClient.noDaemon(daemon, error) }
        struct R: Decodable {
            struct J: Decodable {
                let id: String
                let kind: String
                let status: String
            }
            let jobs: [J]
        }
        guard let r = try? JSONDecoder().decode(R.self, from: data)
        else { return HTTPClient.printJSON(data) }
        if r.jobs.isEmpty {
            print("no jobs")
            return
        }
        print(
            "ID".padding(toLength: 38, withPad: " ", startingAt: 0)
                + "KIND".padding(
                    toLength: 16, withPad: " ", startingAt: 0)
                + "STATUS")
        for j in r.jobs {
            print(
                j.id.padding(
                    toLength: 38, withPad: " ", startingAt: 0)
                    + j.kind.padding(
                        toLength: 16, withPad: " ", startingAt: 0)
                    + j.status)
        }
    }
}

public struct QueueSubmit: AsyncParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "submit",
        abstract: "Enqueue a job (body from --file or stdin).")
    @OptionGroup public var daemon: DaemonOptions
    @Argument(help: "Kind: conversation | embeddings.")
    public var kind: String
    @Option(help: "JSON body file. Omit to read stdin.")
    public var file: String?
    public init() {}

    public func run() async throws {
        let body: Data
        if let file {
            body = try Data(contentsOf: URL(fileURLWithPath: file))
        } else {
            body = FileHandle.standardInput.readDataToEndOfFile()
        }
        guard !body.isEmpty else {
            FailableExit.die("error: empty body")
        }
        let (code, data): (Int, Data)
        do {
            (code, data) = try await HTTPClient.send(
                "POST", daemon.base + "/v1/queue/\(kind)",
                body: body, key: daemon.authKey)
        } catch { HTTPClient.noDaemon(daemon, error) }
        HTTPClient.printJSON(data)
        if code >= 400 { throw ExitCode.failure }
    }
}

public struct QueueGet: AsyncParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "get", abstract: "Get a job's status/result.")
    @OptionGroup public var daemon: DaemonOptions
    @Argument(help: "Job id.") public var id: String
    @Option(help: "Long-poll up to N seconds for completion.")
    public var wait: Int?
    @Flag(help: "Stream status transitions (SSE) until terminal.")
    public var follow = false
    public init() {}

    public func run() async throws {
        if follow {
            var req = URLRequest(
                url: URL(
                    string: daemon.base + "/v1/queue/\(id)/events")!)
            if let k = daemon.authKey, !k.isEmpty {
                req.setValue(
                    "Bearer \(k)",
                    forHTTPHeaderField: "Authorization")
            }
            do {
                let (bytes, _) = try await URLSession.shared.bytes(
                    for: req)
                for try await line in bytes.lines
                where line.hasPrefix("data: ") {
                    let p = String(line.dropFirst(6))
                    if p == "[DONE]" { return }
                    print(p)
                }
            } catch { HTTPClient.noDaemon(daemon, error) }
            return
        }
        let url =
            daemon.base + "/v1/queue/\(id)"
            + (wait.map { "?wait=\($0)" } ?? "")
        let (code, data): (Int, Data)
        do {
            (code, data) = try await HTTPClient.send(
                "GET", url, key: daemon.authKey)
        } catch { HTTPClient.noDaemon(daemon, error) }
        HTTPClient.printJSON(data)
        if code >= 400 { throw ExitCode.failure }
    }
}

public struct QueueRm: AsyncParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "rm", abstract: "Remove (cancel/delete) a job.")
    @OptionGroup public var daemon: DaemonOptions
    @Argument(help: "Job id.") public var id: String
    public init() {}

    public func run() async throws {
        let (code, data): (Int, Data)
        do {
            (code, data) = try await HTTPClient.send(
                "DELETE", daemon.base + "/v1/queue/\(id)",
                key: daemon.authKey)
        } catch { HTTPClient.noDaemon(daemon, error) }
        HTTPClient.printJSON(data)
        if code >= 400 { throw ExitCode.failure }
    }
}
