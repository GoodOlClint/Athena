import ArgumentParser
import AthenaCore
import AthenaEmbedding
import AthenaLLM
import AthenaTranscription
import Foundation

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

    mutating func run() async throws {
        let config = GovernorConfig(
            totalBudgetBytes: budgetBytes,
            listenHost: host,
            listenPort: port
        )
        let governor = MemoryGovernor(config: config)

        // M0 registers governed stubs. The LLM is non-evictable (it is the
        // primary workload); transcription and embedding are evictable so
        // the governor can reclaim their budget under LLM pressure.
        let llm = StubLLMModule()
        await governor.register(llm, evictable: false)
        await governor.register(StubTranscriptionModule(), evictable: true)
        await governor.register(StubEmbeddingModule(), evictable: true)

        let server = AthenaServer(
            config: config, governor: governor, llm: llm)
        try await server.run()
    }
}
