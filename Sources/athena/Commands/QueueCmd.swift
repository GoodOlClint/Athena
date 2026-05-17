import ArgumentParser
import Foundation

/// `athena queue …` — manage the async request queue (M9.1). Thin
/// HTTP client to a running daemon's `/v1/queue` surface.
struct Queue: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "queue",
        abstract: "Inspect and manage the async request queue.",
        subcommands: [
            QueueLs.self, QueueGet.self, QueueSubmit.self, QueueRm.self,
        ])
}

struct QueueLs: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "ls", abstract: "List queued/processed jobs.")
    @OptionGroup var daemon: DaemonOptions
    @Option(help: "Filter by status (queued|running|done|error).")
    var status: String?

    func run() async throws {
        let url =
            daemon.base + "/v1/queue"
            + (status.map { "?status=\($0)" } ?? "")
        let (_, data): (Int, Data)
        do {
            (_, data) = try await HTTPClient.send("GET", url)
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
                j.id.padding(toLength: 38, withPad: " ", startingAt: 0)
                    + j.kind.padding(
                        toLength: 16, withPad: " ", startingAt: 0)
                    + j.status)
        }
    }
}

struct QueueSubmit: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "submit",
        abstract: "Enqueue a job (body from --file or stdin).")
    @OptionGroup var daemon: DaemonOptions
    @Argument(help: "Kind: conversation | embeddings.")
    var kind: String
    @Option(help: "JSON body file. Omit to read stdin.")
    var file: String?

    func run() async throws {
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
                "POST", daemon.base + "/v1/queue/\(kind)", body: body)
        } catch { HTTPClient.noDaemon(daemon, error) }
        HTTPClient.printJSON(data)
        if code >= 400 { throw ExitCode.failure }
    }
}

struct QueueGet: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "get", abstract: "Get a job's status/result.")
    @OptionGroup var daemon: DaemonOptions
    @Argument(help: "Job id.") var id: String
    @Option(help: "Long-poll up to N seconds for completion.")
    var wait: Int?
    @Flag(help: "Stream status transitions (SSE) until terminal.")
    var follow = false

    func run() async throws {
        if follow {
            let req = URLRequest(
                url: URL(
                    string: daemon.base + "/v1/queue/\(id)/events")!)
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
            (code, data) = try await HTTPClient.send("GET", url)
        } catch { HTTPClient.noDaemon(daemon, error) }
        HTTPClient.printJSON(data)
        if code >= 400 { throw ExitCode.failure }
    }
}

struct QueueRm: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "rm", abstract: "Remove (cancel/delete) a job.")
    @OptionGroup var daemon: DaemonOptions
    @Argument(help: "Job id.") var id: String

    func run() async throws {
        let (code, data): (Int, Data)
        do {
            (code, data) = try await HTTPClient.send(
                "DELETE", daemon.base + "/v1/queue/\(id)")
        } catch { HTTPClient.noDaemon(daemon, error) }
        HTTPClient.printJSON(data)
        if code >= 400 { throw ExitCode.failure }
    }
}
