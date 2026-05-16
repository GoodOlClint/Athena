import AthenaCore
import AthenaModels
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
    /// Opt-in MTP speculative decoding. Greedy-only: takes effect only
    /// when temperature == 0 AND the model has an MTP head; otherwise the
    /// standard path is used (temp>0 speculative is the M2.3 named risk).
    public var speculative: Bool

    public init(
        maxTokens: Int = 1024, temperature: Float = 0.7,
        topP: Float = 0.95, speculative: Bool = false
    ) {
        self.maxTokens = maxTokens
        self.temperature = temperature
        self.topP = topP
        self.speculative = speculative
    }

    /// MTP speculative decoding is **greedy-only**: it engages only at
    /// temperature 0. temp>0 residual-sampling acceptance is a deliberate
    /// deferral (bench-unvalidated even in the Python reference) — see
    /// GoodOlClint/athena#1. temp>0 requests fall back to the standard
    /// (correct, non-accelerated) path.
    public var speculativeGreedyEligible: Bool {
        speculative && temperature == 0
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
            // Tell the registry which checkpoint is about to load so it
            // can suppress the MTP head when the weights lack mtp.*.
            AthenaModelRegistration.currentModelDirectory = modelDirectory
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
                    if let speculative = try await self.runSpeculative(
                        prompt: prompt)
                    {
                        continuation.yield(speculative)
                    } else {
                        let stream = try await self.beginGeneration(
                            prompt: prompt)
                        for await event in stream {
                            if case .chunk(let text) = event {
                                continuation.yield(text)
                            }
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

    /// MTP greedy speculative path. Returns the full decoded text, or nil
    /// when not eligible (speculative off, temp>0, or no MTP head) so the
    /// caller falls back to the standard substrate stream. Non-streaming:
    /// one decode of the full id sequence keeps the bit-identical-greedy
    /// comparison unambiguous.
    private func runSpeculative(prompt: String) async throws -> String? {
        guard params.speculativeGreedyEligible, let container
        else { return nil }

        let lmInput = try await container.prepare(
            input: UserInput(chat: [.user(prompt)]))
        let promptTokens = lmInput.text.tokens.asArray(Int.self)
        let maxTokens = params.maxTokens

        return try await container.perform {
            (ctx: ModelContext) -> String? in
            guard let model = ctx.model as? AthenaQwen35Model,
                model.hasMTPHead
            else { return nil }
            let ids = SpeculativeGeneration.generate(
                model: model,
                promptTokens: promptTokens,
                maxTokens: maxTokens,
                eosTokenId: ctx.tokenizer.eosTokenId)
            return ctx.tokenizer.decode(tokenIds: ids)
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
