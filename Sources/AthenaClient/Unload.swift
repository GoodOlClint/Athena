import ArgumentParser
import AthenaCore
import Foundation

/// `athena unload [MODEL]` — unload the running model so the governor
/// reclaims its budget; the daemon keeps running. (Was `stop`; `stop`
/// now controls the daemon process — M9.4.)
public struct Unload: AsyncParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "unload",
        abstract: "Unload the running model and free its budget."
    )

    @Argument(help: "Model name (single-model shim: passed through).")
    public var model: String?

    @Option(help: "Daemon host.")
    public var host: String = "127.0.0.1"

    @Option(help: "Daemon port.")
    public var port: Int = GovernorConfig.defaultPort

    public init() {}

    public func run() async throws {
        var req = URLRequest(
            url: URL(string: "http://\(host):\(port)/api/stop")!)
        req.httpMethod = "POST"
        if let k = Credentials.resolve(
            explicit: nil, host: host, port: port), !k.isEmpty
        {
            req.setValue(
                "Bearer \(k)", forHTTPHeaderField: "Authorization")
        }
        let data: Data
        do {
            (data, _) = try await URLSession.shared.data(for: req)
        } catch {
            print(
                "no running athena daemon at \(host):\(port) "
                    + "(\(error.localizedDescription))")
            throw ExitCode.failure
        }
        let obj =
            (try? JSONSerialization.jsonObject(with: data))
            as? [String: Any]
        print(
            "unloaded "
                + "\(obj?["model"] as? String ?? model ?? "model")")
    }
}
