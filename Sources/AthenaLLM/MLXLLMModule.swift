import AthenaCore
import Foundation
import HuggingFace
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import Tokenizers

/// Sampling/length knobs for the LLM module. Kept Sendable so the governed
/// serve path can hold them across isolation boundaries.
public struct LLMGenerationParameters: Sendable {
    public var maxTokens: Int
    public var temperature: Float
    public var topP: Float

    public init(maxTokens: Int = 1024, temperature: Float = 0.7, topP: Float = 0.95) {
        self.maxTokens = maxTokens
        self.temperature = temperature
        self.topP = topP
    }
}

/// The real MLX-backed LLM module (M1). Loads a Qwen3.5 model from a local
/// directory — no download, no HF hub round-trip — and streams native
/// `TokenIterator` generation via the substrate's `ModelContainer`.
///
/// Memory accounting in M1 is the on-disk safetensors footprint: an honest
/// pre-load admission estimate for the governor. Live Metal-footprint
/// reconciliation is deferred to M5 (OOM/cache hardening).
public actor MLXLLMModule: LLMModule {
    public nonisolated let id: ModuleID = .llm

    private let modelDirectory: URL
    private let params: LLMGenerationParameters
    private let estimatedBytes: Int
    private var container: ModelContainer?

    public init(
        modelDirectory: URL,
        parameters: LLMGenerationParameters = .init()
    ) {
        self.modelDirectory = modelDirectory
        self.params = parameters
        self.estimatedBytes = Self.estimateBytes(forModelAt: modelDirectory)
    }

    public var residentBytes: Int { container == nil ? 0 : estimatedBytes }

    public func memoryEstimate() -> Int { estimatedBytes }

    public func load(reservation: MemoryReservation) async throws {
        if container != nil { return }
        // Route Qwen3.5 directories to Athena's vendored model so the
        // substrate stays pristine. Idempotent; must precede the load.
        // Debug seam: ATHENA_DISABLE_VENDORED_MODEL=1 keeps the substrate's
        // own Qwen35 (used for the M2.1 deterministic parity A/B).
        if ProcessInfo.processInfo.environment[
            "ATHENA_DISABLE_VENDORED_MODEL"] != "1"
        {
            await AthenaModelRegistration.install()
        }
        guard
            FileManager.default.fileExists(
                atPath: modelDirectory.appendingPathComponent("config.json").path)
        else {
            throw AthenaError.moduleLoadFailed(
                .llm,
                reason:
                    "no model at \(modelDirectory.path) (missing config.json)")
        }
        self.container = try await loadModelContainer(
            from: modelDirectory, using: #huggingFaceTokenizerLoader())
    }

    public func unload() async {
        container = nil
    }

    public nonisolated func generate(prompt: String) -> AsyncStream<String> {
        AsyncStream { continuation in
            let task = Task {
                do {
                    let stream = try await self.beginGeneration(prompt: prompt)
                    for await event in stream {
                        if case .chunk(let text) = event {
                            continuation.yield(text)
                        }
                    }
                } catch {
                    continuation.yield(
                        "[athena: generation failed: \(error)]")
                }
                continuation.finish()
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    private func beginGeneration(
        prompt: String
    ) async throws -> AsyncStream<Generation> {
        guard let container else {
            throw AthenaError.moduleLoadFailed(
                .llm, reason: "generate called before load")
        }
        let userInput = UserInput(chat: [.user(prompt)])
        let lmInput = try await container.prepare(input: userInput)
        let gp = GenerateParameters(
            maxTokens: params.maxTokens,
            temperature: params.temperature,
            topP: params.topP)
        return try await container.generate(input: lmInput, parameters: gp)
    }

    /// Resident footprint estimate = sum of the model's `*.safetensors`
    /// bytes. For 4-/8-bit quantized weights this closely tracks the bytes
    /// MLX maps resident, so it is an honest governor admission estimate.
    static func estimateBytes(forModelAt directory: URL) -> Int {
        let fm = FileManager.default
        guard
            let entries = try? fm.contentsOfDirectory(
                at: directory,
                includingPropertiesForKeys: [.fileSizeKey])
        else { return 0 }
        var total = 0
        for url in entries where url.pathExtension == "safetensors" {
            let size =
                (try? url.resourceValues(forKeys: [.fileSizeKey]))?.fileSize
                ?? 0
            total += size
        }
        return total
    }
}
