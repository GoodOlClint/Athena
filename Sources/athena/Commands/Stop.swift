import ArgumentParser
import AthenaCore
import Foundation

/// `athena stop [MODEL]` — unload the running model so the governor
/// reclaims its budget (ollama-style; single-model shim, MODEL echoed).
struct Stop: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "stop",
        abstract: "Unload the running model and free its budget."
    )

    @Argument(help: "Model name (single-model shim: passed through).")
    var model: String?

    @Option(help: "Daemon host.")
    var host: String = "127.0.0.1"

    @Option(help: "Daemon port.")
    var port: Int = GovernorConfig.defaultPort

    func run() async throws {
        var req = URLRequest(
            url: URL(string: "http://\(host):\(port)/api/stop")!)
        req.httpMethod = "POST"
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
        print("stopped \(obj?["model"] as? String ?? model ?? "model")")
    }
}
