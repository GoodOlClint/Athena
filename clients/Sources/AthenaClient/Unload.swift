import ArgumentParser
import Foundation

#if canImport(FoundationNetworking)
    import FoundationNetworking
#endif

/// `athena unload [MODEL] [--module M]` — release a module's slot so
/// the governor reclaims its budget; the daemon keeps running. M41.1
/// generalizes this from "the LLM only" to any module (or all). The
/// default `--module llm` preserves the prior single-arg behavior.
/// (Was `stop`; `stop` now controls the daemon process — M9.4.)
public struct Unload: AsyncParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "unload",
        abstract: "Release a module's slot and free its budget."
    )

    @Argument(
        help: "Model name (single-model: passed through; informational).")
    public var model: String?

    @Option(
        help: """
            Module class to unload (llm, textEmbedding, transcription, \
            diarization, speakerEmbedding). Pass `all` to release every \
            module. Default: llm.
            """)
    public var module: String = "llm"

    @OptionGroup public var daemon: DaemonOptions

    public init() {}

    public func run() async throws {
        try await RemoteModels.unload(
            daemon, module: module.isEmpty ? nil : module)
    }
}
