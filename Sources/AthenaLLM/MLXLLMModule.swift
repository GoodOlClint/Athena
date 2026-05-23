import AthenaCore
import AthenaModels
import AthenaStructured
import Foundation
import HuggingFace
import MLX
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
    /// KV-cache compression codec (the `kv_compression` knob). M20.
    public var kvCompression: KVCompression

    public init(
        maxTokens: Int = 1024, temperature: Float = 0.7,
        topP: Float = 0.95, speculative: Bool = false,
        kvCompression: KVCompression = .none
    ) {
        self.maxTokens = maxTokens
        self.temperature = temperature
        self.topP = topP
        self.speculative = speculative
        self.kvCompression = kvCompression
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

/// The real MLX-backed LLM module (M1). Loads a model from a local
/// directory — no download, no HF hub round-trip — and streams native
/// `TokenIterator` generation via the substrate's `ModelContainer`.
///
/// A Qwen3.5 directory routes to Athena's vendored model (MTP speculative,
/// GDN/Mamba, the bit-identical greedy path); any other architecture the
/// substrate factory supports (Llama, Gemma, Mistral, Phi, …) loads
/// through the substrate's general path and streams plain generation.
/// Structured output / tool calls are guided on BOTH paths (M23 fork A);
/// MTP speculative + TriAttention eviction apply to Qwen3.5 only and
/// degrade cleanly elsewhere.
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

    /// Governor-owned prompt-cache cap in bytes (0 ⇒ disabled). The
    /// per-token KV figure is derived from the model's own config.json so
    /// the cap is sized for THIS architecture; the cap errs toward
    /// refusing early — the safe direction for an OOM guard. Brief 4b /
    /// M23 fork C.
    private let promptCacheCapBytes: Int
    /// Per-token KV upper bound fed to the governor's prompt-cache cap,
    /// derived from config.json (2·layers·kv_heads·head_dim·fp16) so it
    /// tracks the real KV geometry per arch; falls back to a conservative
    /// constant when the dims are absent. TurboQuant stores ~4-bit K/V so
    /// it gets a quarter of the fp16 bound (still over-estimates — the
    /// safe direction for an OOM guard). M20.2 / M23 fork C.
    private let perTokenKVBytes: Int
    /// `vocab_size` from config.json, captured at init so structured
    /// output works on any architecture (the vendored Qwen3.5 model
    /// exposes it directly; substrate arches do not). M23 fork A.
    private let configVocabSize: Int?

    public init(
        modelDirectory: URL,
        parameters: LLMGenerationParameters = .init(),
        promptCacheCapBytes: Int = 0
    ) {
        self.modelDirectory = modelDirectory
        self.params = parameters
        self.estimatedBytes = Self.estimateBytes(forModelAt: modelDirectory)
        self.promptCacheCapBytes = promptCacheCapBytes

        // Size the prompt-cache cap from the model's own KV geometry. The
        // old flat 256 KiB constant over-capped small models ~8× (refusing
        // valid prompts) and would under-cap an arch with larger KV — the
        // dangerous direction. Fall back to the conservative constant when
        // config.json omits the dims. M23 fork C.
        let info = ModelConfigInfo.read(modelDirectory: modelDirectory)
        self.configVocabSize = info?.vocabSize
        let fp16PerToken =
            info?.perTokenKVBytes(bytesPerElement: 2) ?? (256 * 1024)
        // TurboQuant ~4-bit K/V → a quarter of the fp16 bound (still
        // conservative). TriAttention does NOT shrink per-token bytes (it
        // caps token COUNT and pins the prefill at full precision), so it
        // keeps the full fp16 bound like `none`.
        switch parameters.kvCompression {
        case .turboquant: self.perTokenKVBytes = max(1, fp16PerToken / 4)
        case .none, .triattention: self.perTokenKVBytes = fp16PerToken
        }
    }

    /// Brief 4b: refuse a prompt whose KV/prompt-cache would exceed the
    /// governor-owned cap, before any generation, as a governed 503.
    public func preflightPromptCache(prompt: String) async throws {
        try await preflightPromptCache(
            messages: [ChatTurn(role: "user", content: prompt)])
    }

    public func preflightPromptCache(messages: [ChatTurn]) async throws {
        guard promptCacheCapBytes > 0, let container else { return }
        let lmInput = try await container.prepare(
            input: UserInput(chat: Self.chatMessages(messages)))
        let tokens = lmInput.text.tokens.size
        let needed = tokens * perTokenKVBytes
        if needed > promptCacheCapBytes {
            throw AthenaError.promptCacheCapExceeded(
                requestedBytes: needed, capBytes: promptCacheCapBytes)
        }
    }

    /// Map transport-neutral `ChatTurn`s to substrate `Chat.Message`s so
    /// the model's chat template sees real roles. Unknown roles fall back
    /// to `.user`; an empty list becomes a single empty user turn (the
    /// substrate requires at least one message).
    static func chatMessages(_ turns: [ChatTurn]) -> [Chat.Message] {
        let mapped = turns.map { turn in
            Chat.Message(
                role: Chat.Message.Role(rawValue: turn.role) ?? .user,
                content: turn.content)
        }
        return mapped.isEmpty ? [.user("")] : mapped
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
        // Resolve the store-entry symlink: `athena pull` lands a model as
        // a symlink (~/.athena/models/<name> → HF snapshot dir). The
        // substrate's weight loader enumerates the directory but does NOT
        // follow a symlinked ROOT, so it would load ZERO shards → the model
        // fails with keyNotFound on its first parameter. Convert-produced
        // models are real dirs and were unaffected; every pulled model was.
        self.container = try await loadModelContainer(
            from: modelDirectory.resolvingSymlinksInPath(),
            using: #huggingFaceTokenizerLoader())
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
        generate(
            messages: [ChatTurn(role: "user", content: prompt)],
            schemaJSON: schemaJSON, tools: tools,
            maxTokens: nil, temperature: nil)
    }

    public nonisolated func generateMetered(
        messages: [ChatTurn], schemaJSON: String?,
        tools: [[String: any Sendable]]?,
        maxTokens: Int?, temperature: Double?,
        topP: Double?, seed: Int?
    ) -> AsyncStream<GenChunk> {
        // `messages` ([ChatTurn]) is Sendable and crosses into the actor;
        // the non-Sendable `Chat.Message` mapping happens INSIDE the actor
        // methods (Swift 6 strict-concurrency). A single terminal
        // `.usage` carries the true token counts (M27.1): prompt = the
        // tokenized input length, completion = tokens emitted. A terminal
        // `.finish` follows with the stop reason (M31.2): the request
        // truncated when its completion count reached the same effective
        // cap both generation paths enforce.
        AsyncStream { continuation in
            let task = Task {
                var usage = TokenUsage.zero
                do {
                    if let speculative = try await self.runSpeculative(
                        messages: messages, schemaJSON: schemaJSON,
                        tools: tools, maxTokens: maxTokens)
                    {
                        usage = speculative.usage
                        continuation.yield(.text(speculative.text))
                        continuation.yield(.usage(usage))
                    } else {
                        // runSpeculative returns nil only for UNstructured
                        // requests (no schema) — those stream from the
                        // standard substrate path. Structured requests are
                        // always guided: the vendored Qwen3.5 path, or the
                        // substrate-guided path for other arches (M23).
                        let stream = try await self.beginGeneration(
                            messages: messages, tools: tools,
                            maxTokens: maxTokens, temperature: temperature,
                            topP: topP, seed: seed)
                        for await event in stream {
                            switch event {
                            case .chunk(let text):
                                continuation.yield(.text(text))
                            case .info(let info):
                                // Substrate's terminal completion record
                                // carries the real token geometry.
                                usage = TokenUsage(
                                    promptTokens: info.promptTokenCount,
                                    completionTokens:
                                        info.generationTokenCount)
                            default:
                                break
                            }
                        }
                        continuation.yield(.usage(usage))
                    }
                    // M31.2: the effective cap is the same positive-wins
                    // resolution both paths apply; hitting it ⇒ truncated.
                    let cap = await Self.effectiveMaxTokens(
                        maxTokens, self.params.maxTokens)
                    continuation.yield(
                        .finish(
                            usage.completionTokens >= cap
                                ? .length : .stop))
                } catch {
                    continuation.yield(
                        .text("[athena: generation failed: \(error)]"))
                    continuation.yield(.usage(.zero))
                    continuation.yield(.finish(.stop))
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
        messages: [ChatTurn], schemaJSON: String?,
        tools: [[String: any Sendable]]?, maxTokens: Int?
    ) async throws -> (text: String, usage: TokenUsage)? {
        guard let container else { return nil }
        let speculativeEligible = params.speculativeGreedyEligible
        if schemaJSON == nil && !speculativeEligible { return nil }

        let lmInput = try await container.prepare(
            input: UserInput(
                chat: Self.chatMessages(messages), tools: tools))
        let promptTokens = lmInput.text.tokens.asArray(Int.self)
        // M24.3: a positive per-request override wins over the loaded
        // default; the greedy/MTP paths are length-only (temperature is
        // inert under the Guide / speculative greedy).
        let maxTokens = Self.effectiveMaxTokens(maxTokens, params.maxTokens)

        // Build (or reuse) the structured vocabulary tokens once per
        // model. The ~150k tokenizer.decode calls are the dominant
        // structured-request cost and are schema-independent.
        if schemaJSON != nil, cachedVocabTokens == nil {
            let cfgVocab = configVocabSize
            let built = try await container.perform {
                (ctx: ModelContext) -> ([VocabToken], UInt32)? in
                // Qwen3.5 exposes vocabularySize directly; any other
                // architecture uses config.json's vocab_size, so guided
                // structured output is available everywhere (M23 fork A).
                guard
                    let vocabSize =
                        (ctx.model as? AthenaQwen35Model)?.vocabularySize
                        ?? cfgVocab
                else { return nil }
                let (t, e) = StructuredVocab.tokens(
                    tokenizer: ctx.tokenizer, vocabSize: vocabSize)
                return (t, e)
            }
            if let built {
                cachedVocabTokens = (
                    built.0, built.1,
                    StructuredVocabulary.openerAliases(tokens: built.0))
            }
        }
        let vocabTokens = cachedVocabTokens

        let cfgVocab = configVocabSize
        // The closure returns the decoded text plus the completion token
        // count (`ids.count`); the prompt count is `promptTokens.count`
        // from the outer scope. nil ⇒ fall back to the substrate stream.
        let decoded = try await container.perform {
            (ctx: ModelContext) -> (text: String, completion: Int)? in

            // Structured ⇒ NO-THINK by construction: the Guide masks
            // from token 0, so the schema is enforced immediately and
            // the model's <think>…</think> is suppressed (matches
            // the consuming application's enable_thinking=False production config).
            // Schema-enforced output that ALSO permits a thinking prefix
            // (deferred enforcement, Patch 6) is tracked in
            // GoodOlClint/athena#2.
            func makeGuide() throws -> StructuredGuide? {
                guard let schemaJSON, let vt = vocabTokens else {
                    return nil
                }
                let g = try StructuredGuide(
                    index: StructuredIndex(
                        jsonSchema: schemaJSON,
                        vocabulary: StructuredVocabulary(
                            tokens: vt.tokens, eosTokenId: vt.eos)))
                g.openerAlias = vt.opener
                return g
            }
            let guide = try makeGuide()

            // Vendored Qwen3.5 path — UNCHANGED (MTP speculative when
            // eligible, else guided/plain greedy on the vendored model).
            if let model = ctx.model as? AthenaQwen35Model {
                // TriAttention is inert on the MTP/speculative + guided
                // paths: eviction can't un-mix the GDN/Mamba recurrent
                // state, and these paths must stay bit-identical greedy.
                // Clear any policy a prior standard request left on the
                // shared model instance before building caches here.
                model.triAttentionEviction = nil

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
                    return nil  // unstructured + no MTP ⇒ substrate stream
                }
                return (ctx.tokenizer.decode(tokenIds: ids), ids.count)
            }

            // Any other architecture: schema-guided decoding on the
            // substrate generation path (M23 fork A). Unstructured
            // requests (no guide) return nil → the standard substrate
            // stream in beginGeneration. MTP speculative does not apply
            // (no mtp.* weights) and degrades to this path cleanly.
            guard let guide, let vt = vocabTokens,
                let vocabSize = cfgVocab
            else { return nil }
            let ids = try GuidedSubstrate.generate(
                model: ctx.model, promptTokens: promptTokens,
                vocab: vocabSize, maxTokens: maxTokens,
                eosTokenId: Int(vt.eos), guide: guide)
            return (ctx.tokenizer.decode(tokenIds: ids), ids.count)
        }
        guard let decoded else { return nil }
        return (
            decoded.text,
            TokenUsage(
                promptTokens: promptTokens.count,
                completionTokens: decoded.completion))
    }

    private func beginGeneration(
        messages: [ChatTurn], tools: [[String: any Sendable]]?,
        maxTokens: Int?, temperature: Double?,
        topP: Double?, seed: Int?
    ) async throws -> AsyncStream<Generation> {
        guard let container else {
            throw AthenaError.moduleLoadFailed(
                .llm, reason: "generate called before load")
        }
        let userInput = UserInput(
            chat: Self.chatMessages(messages), tools: tools)
        let lmInput = try await container.prepare(input: userInput)
        // The standard attention path is the ONLY place TriAttention
        // eviction applies. Set it on the model so the substrate's
        // newCache(parameters:) builds evicting attention caches; nil
        // (any non-triattention knob) leaves KVCacheSimple intact.
        let evictionPolicy = params.kvCompression.eviction
        try await container.perform { (ctx: ModelContext) in
            (ctx.model as? AthenaQwen35Model)?.triAttentionEviction =
                evictionPolicy
        }
        let gen = params.kvCompression.generation
        // M24.3: per-request max_tokens/temperature override the loaded
        // defaults (a negative/zero temperature override is ignored).
        let temp =
            (temperature.map { Float($0) }).flatMap { $0 >= 0 ? $0 : nil }
            ?? params.temperature
        // M31.3: per-request top_p overrides the loaded default; only the
        // (0,1) range engages nucleus sampling, and only when temp>0 (at
        // temp==0 the substrate uses argmax — top_p/seed are inert).
        let tp =
            topP.map { Float($0) }.flatMap { $0 > 0 && $0 < 1 ? $0 : nil }
            ?? params.topP
        // M31.3: a per-request seed makes temp>0 sampling reproducible by
        // pinning the substrate's global RNG before generation.
        if let seed, seed >= 0 { MLXRandom.seed(UInt64(seed)) }
        let gp = GenerateParameters(
            maxTokens: Self.effectiveMaxTokens(maxTokens, params.maxTokens),
            kvBits: gen.kvBits,
            kvQuantizationScheme: gen.scheme,
            temperature: temp,
            topP: tp)
        return try await container.generate(input: lmInput, parameters: gp)
    }

    /// A positive per-request `max_tokens` override wins; otherwise the
    /// loaded default. Guards against 0/negative overrides truncating to
    /// nothing.
    static func effectiveMaxTokens(_ override: Int?, _ fallback: Int) -> Int {
        if let o = override, o > 0 { return o }
        return fallback
    }

    /// Resident footprint estimate = sum of the model's `*.safetensors`
    /// bytes. For 4-/8-bit quantized weights this closely tracks the bytes
    /// MLX maps resident, so it is an honest governor admission estimate.
    static func estimateBytes(forModelAt directory: URL) -> Int {
        let fm = FileManager.default
        // Resolve the store-entry symlink first: `pull` lands a model as a
        // symlink (~/.athena/models/<name> → HF snapshot). `contentsOfDirectory`
        // does NOT traverse a symlinked root, so a pulled model would
        // enumerate to nothing → 0 B estimate → defeated OOM gate. (Then the
        // per-shard resolve below handles the blob symlinks inside.)
        let dir = directory.resolvingSymlinksInPath()
        guard
            let entries = try? fm.contentsOfDirectory(
                at: dir,
                includingPropertiesForKeys: [.fileSizeKey])
        else { return 0 }
        var total = 0
        for url in entries where url.pathExtension == "safetensors" {
            // Resolve the shard symlink before sizing: `pull` stores
            // shards in the HF-cache layout (each a symlink to
            // ../../blobs/<sha>). Sizing the link itself (~76 B) would
            // make the governor's pre-load estimate ~0 and defeat its OOM
            // admission gate — same root cause as ModelHealth's size check.
            let size =
                (try? url.resolvingSymlinksInPath()
                    .resourceValues(forKeys: [.fileSizeKey]))?.fileSize
                ?? 0
            total += size
        }
        return total
    }
}
