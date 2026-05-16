import ArgumentParser
import AthenaCore
import AthenaEmbedding
import AthenaLLM
import AthenaTranscription
import Foundation

enum Engine: String, CaseIterable, ExpressibleByArgument {
    case mlx
    case stub
}

/// Start the governed HTTP surface. This is the launchd-able daemon.
struct Serve: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "serve",
        abstract: "Start the governed HTTP inference surface."
    )

    @Option(help: "Listen host.")
    var host: String = "127.0.0.1"

    @Option(help: "Listen port. Default 7447 — Athena's own port.")
    var port: Int = GovernorConfig.defaultPort

    @Option(help: "Global memory budget in bytes. Defaults to 75% of RAM.")
    var budgetBytes: Int?

    @Option(help: "LLM engine: mlx (real Qwen3.5) or stub (no model).")
    var engine: Engine = .mlx

    @Option(help: "Sampling temperature (0 = deterministic greedy).")
    var temperature: Double?

    @Option(help: "Max generated tokens.")
    var maxTokens: Int = 1024

    @Flag(help: "Enable MTP speculative decoding (greedy/temp 0 only).")
    var speculative = false

    @Option(help: "Model directory path, or a name under the model store.")
    var model: String?

    @Option(help: "Model store root. Default: external SSD mlx-models.")
    var modelStore: String?

    mutating func run() async throws {
        let config = GovernorConfig(
            totalBudgetBytes: budgetBytes,
            listenHost: host,
            listenPort: port
        )
        let governor = MemoryGovernor(config: config)

        let store = ModelStore(
            rootDirectory: modelStore.map {
                URL(fileURLWithPath: $0, isDirectory: true)
            } ?? ModelStore.defaultRoot)
        let modelURL = store.resolve(model)

        // The LLM is non-evictable (the primary workload); transcription and
        // embedding remain governed stubs (real impls land in M4) and are
        // evictable so the governor can reclaim their budget under pressure.
        let llm: any LLMModule
        switch engine {
        case .stub:
            llm = StubLLMModule()
        case .mlx:
            llm = MLXLLMModule(
                modelDirectory: modelURL,
                parameters: .init(
                    maxTokens: maxTokens,
                    temperature: Float(temperature ?? 0.7),
                    speculative: speculative))
        }
        await governor.register(llm, evictable: false)
        await governor.register(StubTranscriptionModule(), evictable: true)
        await governor.register(StubEmbeddingModule(), evictable: true)

        print(
            "athena: engine=\(engine.rawValue) "
                + "model=\(modelURL.path) "
                + "budget=\(config.totalBudgetBytes)B "
                + "listen=\(config.listenHost):\(config.listenPort)")

        let server = AthenaServer(
            config: config, governor: governor, llm: llm)
        try await server.run()
    }
}
