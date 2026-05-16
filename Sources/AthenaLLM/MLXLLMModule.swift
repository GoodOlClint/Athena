import AthenaCore
import AthenaModels
import AthenaStructured
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
    /// Cached structured-output vocabulary tokens (the ~150k
    /// `tokenizer.decode` calls are model-fixed and schema-independent —
    /// build once, reuse every structured request). Sendable, so it
    /// crosses into the `container.perform` closure safely.
    private var cachedVocabTokens:
        (tokens: [VocabToken], eos: UInt32, opener: [UInt32: UInt32])?

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
        generate(prompt: prompt, schemaJSON: nil, tools: nil)
    }

    public nonisolated func generate(
        prompt: String, schemaJSON: String?,
        tools: [[String: any Sendable]]?
    ) -> AsyncStream<String> {
        AsyncStream { continuation in
            let task = Task {
                do {
                    if let speculative = try await self.runSpeculative(
                        prompt: prompt, schemaJSON: schemaJSON,
                        tools: tools)
                    {
                        continuation.yield(speculative)
                    } else {
                        // No structured constraint on the non-speculative
                        // substrate path yet (M3.3c); a schema request
                        // that can't take the guided speculative path
                        // falls back to unconstrained generation.
                        let stream = try await self.beginGeneration(
                            prompt: prompt, tools: tools)
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
    /// Returns the full decoded text, or nil to fall back to the
    /// unconstrained substrate stream. A structured request (`schemaJSON`)
    /// ALWAYS takes a guided path (greedy): the MTP speculative loop when
    /// eligible (faster), else a plain guided-greedy loop. An
    /// unstructured request takes the opt-in speculative path only.
    private func runSpeculative(
        prompt: String, schemaJSON: String?,
        tools: [[String: any Sendable]]?
    ) async throws -> String? {
        guard let container else { return nil }
        let speculativeEligible = params.speculativeGreedyEligible
        if schemaJSON == nil && !speculativeEligible { return nil }

        let lmInput = try await container.prepare(
            input: UserInput(chat: [.user(prompt)], tools: tools))
        let promptTokens = lmInput.text.tokens.asArray(Int.self)
        let maxTokens = params.maxTokens

        // Build (or reuse) the structured vocabulary tokens once per
        // model. The ~150k tokenizer.decode calls are the dominant
        // structured-request cost and are schema-independent.
        if schemaJSON != nil, cachedVocabTokens == nil {
            let built = try await container.perform {
                (ctx: ModelContext) -> ([VocabToken], UInt32)? in
                guard let model = ctx.model as? AthenaQwen35Model
                else { return nil }
                let (t, e) = StructuredVocab.tokens(
                    tokenizer: ctx.tokenizer,
                    vocabSize: model.vocabularySize)
                return (t, e)
            }
            if let built {
                cachedVocabTokens = (
                    built.0, built.1,
                    StructuredVocabulary.openerAliases(tokens: built.0))
            }
        }
        let vocabTokens = cachedVocabTokens

        return try await container.perform {
            (ctx: ModelContext) -> String? in
            guard let model = ctx.model as? AthenaQwen35Model
            else { return nil }

            // Structured ⇒ NO-THINK by construction: the Guide masks
            // from token 0, so the schema is enforced immediately and
            // the model's <think>…</think> is suppressed (matches
            // the consuming application's enable_thinking=False production config).
            // Schema-enforced output that ALSO permits a thinking prefix
            // (deferred enforcement, Patch 6) is tracked in
            // GoodOlClint/athena#2.
            var guide: StructuredGuide?
            if let schemaJSON, let vt = vocabTokens {
                let g = try StructuredGuide(
                    index: StructuredIndex(
                        jsonSchema: schemaJSON,
                        vocabulary: StructuredVocabulary(
                            tokens: vt.tokens, eosTokenId: vt.eos)))
                g.openerAlias = vt.opener
                guide = g
            }

            let ids: [Int]
            if speculativeEligible && model.hasMTPHead {
                ids = SpeculativeGeneration.generate(
                    model: model, promptTokens: promptTokens,
                    maxTokens: maxTokens,
                    eosTokenId: ctx.tokenizer.eosTokenId, guide: guide)
            } else if guide != nil {
                // Structured but no speculative/MTP path available.
                ids = GuidedGreedy.generate(
                    model: model, promptTokens: promptTokens,
                    maxTokens: maxTokens,
                    eosTokenId: ctx.tokenizer.eosTokenId, guide: guide)
            } else {
                return nil  // unstructured + no MTP ⇒ substrate fallback
            }
            return ctx.tokenizer.decode(tokenIds: ids)
        }
    }

    private func beginGeneration(
        prompt: String, tools: [[String: any Sendable]]?
    ) async throws -> AsyncStream<Generation> {
        guard let container else {
            throw AthenaError.moduleLoadFailed(
                .llm, reason: "generate called before load")
        }
        let userInput = UserInput(chat: [.user(prompt)], tools: tools)
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
