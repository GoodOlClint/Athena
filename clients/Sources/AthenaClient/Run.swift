import ArgumentParser
import Foundation

#if canImport(FoundationNetworking)
    import FoundationNetworking
#endif
#if canImport(Darwin)
    import Darwin
#elseif canImport(Glibc)
    import Glibc
#endif

/// `athena run MODEL [PROMPT]` — one-shot generation against a running
/// daemon. PROMPT from args or piped stdin. An interactive REPL (tty,
/// no prompt) is a documented later addition.
public struct Run: AsyncParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "run",
        abstract: "Run a one-shot prompt against a running daemon."
    )

    @Argument(help: "Model name (single-model: passed through).")
    public var model: String

    @Argument(
        parsing: .captureForPassthrough,
        help: "Prompt. Omit to read from piped stdin.")
    public var prompt: [String] = []

    @Option(help: "Daemon host.")
    public var host: String = "127.0.0.1"

    @Option(help: "Daemon port.")
    public var port: Int = athenaDefaultPort

    public init() {}

    public func run() async throws {
        var text = prompt.joined(separator: " ")
        if text.isEmpty {
            if isatty(FileHandle.standardInput.fileDescriptor) != 0 {
                print(
                    "error: provide a prompt argument or pipe input "
                        + "(interactive REPL is a later addition)")
                throw ExitCode.failure
            }
            let data = FileHandle.standardInput
                .readDataToEndOfFile()
            text = String(data: data, encoding: .utf8) ?? ""
        }
        text = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else {
            print("error: empty prompt")
            throw ExitCode.failure
        }

        var req = URLRequest(
            url: URL(string: "http://\(host):\(port)/api/chat")!)
        req.httpMethod = "POST"
        req.setValue(
            "application/json", forHTTPHeaderField: "Content-Type")
        if let k = Credentials.resolve(
            explicit: nil, host: host, port: port), !k.isEmpty
        {
            req.setValue(
                "Bearer \(k)", forHTTPHeaderField: "Authorization")
        }
        req.httpBody = try JSONSerialization.data(withJSONObject: [
            "model": model,
            "messages": [["role": "user", "content": text]],
            "stream": false,
        ])

        let data: Data
        do {
            (data, _) = try await URLSession.shared.data(for: req)
        } catch {
            print(
                "no running athena daemon at \(host):\(port) "
                    + "(\(error.localizedDescription))")
            throw ExitCode.failure
        }

        guard
            let obj = try? JSONSerialization.jsonObject(with: data)
                as? [String: Any]
        else {
            print(
                "unexpected response: "
                    + (String(data: data, encoding: .utf8)
                        ?? "<binary>"))
            throw ExitCode.failure
        }
        if let err = obj["error"] as? [String: Any],
            let msg = err["message"] as? String
        {
            print("error: \(msg)")
            throw ExitCode.failure
        }
        if let content = obj["content"] as? String {
            print(content)
        } else {
            print(
                "unexpected response: "
                    + (String(data: data, encoding: .utf8) ?? ""))
            throw ExitCode.failure
        }
    }
}
