import ArgumentParser
import Foundation

/// `athena vectors …` — drive the built-in vector DB (M9.2). Thin HTTP
/// client to a running daemon's `/v1/vectors` surface.
public struct Vectors: AsyncParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "vectors",
        abstract:
            "Upsert, query, and inspect the built-in vector DB.",
        subcommands: [
            VectorsAdd.self, VectorsQuery.self, VectorsRm.self,
            VectorsStats.self,
        ])
    public init() {}
}

/// Parse a comma-separated float list ("0.1,0.2,-3"). Empty / malformed
/// ⇒ a hard exit so callers don't silently send a degenerate vector.
private func parseVector(_ s: String) -> [Float] {
    let parts = s.split(separator: ",").map {
        $0.trimmingCharacters(in: .whitespaces)
    }
    let floats = parts.compactMap { Float($0) }
    guard !floats.isEmpty, floats.count == parts.count else {
        FailableExit.die(
            "error: --vector must be a comma-separated float list")
    }
    return floats
}

/// Parse a JSON object string for `--metadata`. Non-object / malformed
/// ⇒ hard exit (the store expects an object or nothing).
private func parseMetadata(_ s: String) -> [String: Any] {
    guard let d = s.data(using: .utf8),
        let obj = try? JSONSerialization.jsonObject(with: d),
        let dict = obj as? [String: Any]
    else {
        FailableExit.die("error: --metadata must be a JSON object")
    }
    return dict
}

private func postJSON(
    _ daemon: DaemonOptions, _ path: String, _ body: [String: Any]
) async -> (Int, Data) {
    guard
        let data = try? JSONSerialization.data(withJSONObject: body)
    else { FailableExit.die("error: could not encode request") }
    do {
        return try await HTTPClient.send(
            "POST", daemon.base + path, body: data,
            key: daemon.authKey)
    } catch { HTTPClient.noDaemon(daemon, error) }
}

public struct VectorsAdd: AsyncParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "add",
        abstract:
            "Upsert a vector by id (embed --text or pass --vector).")
    @OptionGroup public var daemon: DaemonOptions
    @Argument(help: "Vector id.") public var id: String
    @Option(help: "Text to embed server-side.") public var text:
        String?
    @Option(help: "Literal vector: comma-separated floats.")
    public var vector: String?
    @Option(help: "Raw JSON body file (full passthrough).")
    public var file: String?
    @Option(help: "Metadata as a JSON object string.")
    public var metadata: String?
    public init() {}

    public func run() async throws {
        let code: Int
        let data: Data
        if let file {
            let raw = try Data(
                contentsOf: URL(fileURLWithPath: file))
            do {
                (code, data) = try await HTTPClient.send(
                    "POST", daemon.base + "/v1/vectors", body: raw,
                    key: daemon.authKey)
            } catch { HTTPClient.noDaemon(daemon, error) }
        } else {
            guard (text == nil) != (vector == nil) else {
                FailableExit.die(
                    "error: pass exactly one of --text / --vector "
                        + "(or --file)")
            }
            var body: [String: Any] = ["id": id]
            if let text { body["text"] = text }
            if let vector { body["vector"] = parseVector(vector) }
            if let metadata {
                body["metadata"] = parseMetadata(metadata)
            }
            (code, data) = await postJSON(
                daemon, "/v1/vectors", body)
        }
        HTTPClient.printJSON(data)
        if code >= 400 { throw ExitCode.failure }
    }
}

public struct VectorsQuery: AsyncParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "query",
        abstract: "k-NN search by --text or --vector.")
    @OptionGroup public var daemon: DaemonOptions
    @Option(help: "Query text to embed server-side.")
    public var text: String?
    @Option(help: "Query vector: comma-separated floats.")
    public var vector: String?
    @Option(name: .shortAndLong, help: "Neighbors to return.")
    public var k: Int = 5
    public init() {}

    public func run() async throws {
        guard (text == nil) != (vector == nil) else {
            FailableExit.die(
                "error: pass exactly one of --text / --vector")
        }
        var body: [String: Any] = ["k": k]
        if let text { body["text"] = text }
        if let vector { body["vector"] = parseVector(vector) }
        let (code, data) = await postJSON(
            daemon, "/v1/vectors/query", body)
        struct R: Decodable {
            struct M: Decodable {
                let id: String
                let score: Float
            }
            let matches: [M]
        }
        guard code < 400,
            let r = try? JSONDecoder().decode(R.self, from: data)
        else {
            HTTPClient.printJSON(data)
            if code >= 400 { throw ExitCode.failure }
            return
        }
        if r.matches.isEmpty {
            print("no matches")
            return
        }
        print(
            "ID".padding(toLength: 38, withPad: " ", startingAt: 0)
                + "SCORE")
        for m in r.matches {
            print(
                m.id.padding(
                    toLength: 38, withPad: " ", startingAt: 0)
                    + String(format: "%.4f", m.score))
        }
    }
}

public struct VectorsRm: AsyncParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "rm", abstract: "Delete a vector by id.")
    @OptionGroup public var daemon: DaemonOptions
    @Argument(help: "Vector id.") public var id: String
    public init() {}

    public func run() async throws {
        let (code, data): (Int, Data)
        do {
            (code, data) = try await HTTPClient.send(
                "DELETE", daemon.base + "/v1/vectors/\(id)",
                key: daemon.authKey)
        } catch { HTTPClient.noDaemon(daemon, error) }
        HTTPClient.printJSON(data)
        if code >= 400 { throw ExitCode.failure }
    }
}

public struct VectorsStats: AsyncParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "stats",
        abstract: "Vector count, dimension, and byte footprint.")
    @OptionGroup public var daemon: DaemonOptions
    public init() {}

    public func run() async throws {
        let (code, data): (Int, Data)
        do {
            (code, data) = try await HTTPClient.send(
                "GET", daemon.base + "/v1/vectors/stats",
                key: daemon.authKey)
        } catch { HTTPClient.noDaemon(daemon, error) }
        HTTPClient.printJSON(data)
        if code >= 400 { throw ExitCode.failure }
    }
}
