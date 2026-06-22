import AthenaCore
import AthenaModels
import AthenaStructured
import CoreImage
import Foundation
import HuggingFace
import Logging
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import MLXVLM
import Tokenizers

/// Sampling/length knobs for the LLM module. Kept Sendable so the governed
/// serve path can hold them across isolation boundaries.
public struct LLMGenerationParameters: Sendable {
    public var maxTokens: Int
    public var temperature: Float
    public var topP: Float
    /// Opt-in MTP speculative decoding. Takes effect when the loaded
    /// model has an MTP head; the runtime picks the greedy speculative
    /// loop at temperature 0 (bit-identical to non-speculative greedy)
    /// or the sampling speculative loop at temperature > 0
    /// (distributionally identical to non-speculative sampling).
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

    /// Whether THIS request is opted into MTP speculative decoding.
    /// M40 lifted the "greedy only" restriction — both temp == 0
    /// (greedy speculative, bit-identical to non-speculative greedy)
    /// and temp > 0 (sampling speculative, distributionally identical
    /// to non-speculative sampling) are honored. Engagement still
    /// requires an MTP head on the loaded model.
    public var speculativeEligible: Bool { speculative }
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

public actor MLXLLMModule: LLMModule, ModelSelectable {
    public nonisolated let id: ModuleID = .llm
    public nonisolated var moduleID: ModuleID { .llm }

    /// M48.2 — daemon-side logger for the dispatch decision (which
    /// internal generate path each request takes). `.debug` so
    /// production stays quiet by default; flip per-subsystem via
    /// `sudo log config --mode "level:debug" --subsystem athena` when
    /// diagnosing a specific request shape.
    nonisolated static let log = Logger(label: "athena.llm")

    /// ADR 026 — the selectable set is the model store classified by
    /// `ModelSupport` (an LLM slot accepts `.llm` and `.vision`), scanned live
    /// from `modelStoreRoot`; there is no pushed-in allowlist. `configuredDefault`
    /// is the per-module TOML default used when a request omits `model` (nil ⇒
    /// resolve by the ambiguity rule). A request naming a model absent from the
    /// store is `modelNotAvailable` (400) — never a silent fallback or an
    /// on-request download.
    private let configuredDefault: String?
    private let modelStoreRoot: URL
    private let params: LLMGenerationParameters
    /// Governor admission estimate. M41.2 sizes this as the MAX across
    /// the allowlist, so the fixed slot bounds the largest member —
    /// admitting a small default but later rebinding to a giant would
    /// otherwise under-account. Same approach M39 took (implicit) for
    /// embeddings.
    private let estimatedBytes: Int
    private var container: ModelContainer?
    /// Currently-resident model name (nil ⇒ slot unloaded).
    private var residentName: String?
    /// M71.2 — true when the resident container was loaded via the substrate's
    /// `VLMModelFactory` (a vision checkpoint with an image tower). Drives the
    /// `servesVision` capability.
    private var residentIsVision = false
    /// M62 — the model the NEXT cold `load(reservation:)` will bind, set by
    /// `selectColdLoadModel` before the governor's non-blocking cold-load is
    /// kicked off. nil ⇒ bind the default. Only consulted while the slot is
    /// unloaded; a warm swap goes through `rebind`.
    private var desiredName: String?
    /// Structured-output vocabulary tokens: the model's full token set with
    /// decoded bytes (the ~150k `tokenizer.decode` calls are model-fixed and
    /// schema-independent), plus the eos id and the opener-alias table.
    typealias VocabBundle = (
        tokens: [VocabToken], eos: UInt32, opener: [UInt32: UInt32]
    )
    /// C3 (M68.2) — the per-model structured-vocab build, memoized as a Task.
    /// The build is tens of seconds (the 150k decodes), so concurrent
    /// first-of-model structured requests must COALESCE onto ONE build. The
    /// pre-fix code did an actor-reentrant check-then-act
    /// (`cachedVocabTokens == nil` → `await container.perform { … }` → write):
    /// two requests both saw nil and both built, and worse, a rebind landing
    /// in the `await` window nil'd the cache and changed the tokenizer while
    /// the in-flight build then wrote a STALE-tokenizer result back — a wrong
    /// structured guide (silent corruption). Memoizing as a Task makes the
    /// check-then-create actor-atomic (no `await` between the nil-check and
    /// the assignment), so exactly one build runs and concurrent callers
    /// await the same Task. The Task captures THIS model's `container`, so a
    /// rebind mid-build still yields the correct vocab for the request that
    /// started it; the field is invalidated (cancel+nil) wherever the model
    /// changes, so the next request after a rebind builds fresh. nil ⇒ no
    /// build started yet for the resident model.
    private var vocabBuild: Task<VocabBundle?, Error>?

    /// M53 — cached per-model structured-output vocabulary + parser
    /// factory (the llguidance token trie + vocab slicer). Building it
    /// (~0.24 s) is the only non-trivial structured-output cost and is
    /// schema-independent, so it is built once from the `vocabBuild` bundle
    /// and reused for every schema and every request. Invalidated alongside
    /// `vocabBuild` (in `resetStructuredCaches`) — a rebind changes the
    /// vocabulary. Its build is synchronous (no `await`), so the
    /// check-then-set below is actor-atomic and needs no Task memoization.
    /// `@unchecked Sendable`, so it crosses into the `container.perform`
    /// closure. The per-schema `StructuredIndex` is now ~1 ms (vs the old
    /// ~60 s outlines DFA compile), so it is NOT cached — built fresh per
    /// request from this factory.
    private var cachedStructuredVocabulary: StructuredVocabulary?

    /// Governor-owned prompt-cache cap in bytes (0 ⇒ disabled). The
    /// per-token KV figure is derived from the model's own config.json so
    /// the cap is sized for THIS architecture; the cap errs toward
    /// refusing early — the safe direction for an OOM guard. Brief 4b /
    /// M23 fork C.
    private let promptCacheCapBytes: Int
    /// Per-token KV upper bound fed to the governor's prompt-cache cap,
    /// derived from config.json (2·layers·kv_heads·head_dim·fp16) so it
    /// tracks the real KV geometry per arch; falls back to a conservative
    /// constant when the dims are absent. M23 fork C.
    /// M41.2: recomputed on each rebind from the new model's config.
    private var perTokenKVBytes: Int
    /// `vocab_size` from config.json (captured per loaded model). M23
    /// fork A; M41.2 makes it per-rebind.
    private var configVocabSize: Int?

    /// M59.1 — cross-request prompt-prefix KV cache. `nil` unless the
    /// operator enabled `[prompt_cache]`. The SAME instance is shared with
    /// the governor (M59.2: pool-byte snapshot + pressure relief), so it is
    /// constructed once in the serve path and injected, not owned here.
    /// Entries are keyed by resident model id (M59.1 scope).
    private let prefixCache: PrefixKVCache?

    /// Single-model convenience init kept for source-compat with M27/M40
    /// call sites (and tests): the directory IS a store entry, so the store
    /// root is its parent and the directory's basename is the configured
    /// default (ADR 026).
    public init(
        modelDirectory: URL,
        parameters: LLMGenerationParameters = .init(),
        promptCacheCapBytes: Int = 0,
        prefixCache: PrefixKVCache? = nil
    ) {
        self.init(
            modelStoreRoot: modelDirectory.deletingLastPathComponent(),
            configuredDefault: modelDirectory.lastPathComponent,
            parameters: parameters,
            promptCacheCapBytes: promptCacheCapBytes,
            prefixCache: prefixCache)
    }

    /// ADR 026 — the LLM slot serves whatever LLM/vision models are in the
    /// store under `modelStoreRoot`; `configuredDefault` (the TOML `model` key)
    /// is loaded when a request omits `model` (nil ⇒ ambiguity rule).
    public init(
        modelStoreRoot: URL,
        configuredDefault: String? = nil,
        parameters: LLMGenerationParameters = .init(),
        promptCacheCapBytes: Int = 0,
        prefixCache: PrefixKVCache? = nil
    ) {
        self.prefixCache = prefixCache
        self.modelStoreRoot = modelStoreRoot
        let cleanDefault = (configuredDefault?.isEmpty == true)
            ? nil : configuredDefault
        self.configuredDefault = cleanDefault
        self.params = parameters
        // Governor admission estimate: MAX across the store's LLM/vision
        // models so the fixed slot still bounds the largest member after a
        // rebind. Errs high (the safe direction for an OOM gate); 0 when the
        // store has no model of the class yet (e.g. pulled post-boot).
        let classIds = StoreModelClass.ids(
            storeRoot: modelStoreRoot, accept: { $0.isLLMSlot })
        self.estimatedBytes =
            classIds
            .compactMap {
                ModelStoreLayout.localDirectory(
                    for: $0, storeRoot: modelStoreRoot)
            }
            .map { Self.estimateBytes(forModelAt: $0) }
            .max() ?? 0
        self.promptCacheCapBytes = promptCacheCapBytes

        // Initial cap geometry seeded from the configured default (or the
        // first store model of the class); rebind recomputes from the loaded
        // model's config.
        let seedName = cleanDefault ?? classIds.first
        let seedDir = seedName.flatMap {
            ModelStoreLayout.localDirectory(
                for: $0, storeRoot: modelStoreRoot)
        }
        let info = seedDir.flatMap { ModelConfigInfo.read(modelDirectory: $0) }
        self.configVocabSize = info?.vocabSize
        let fp16PerToken =
            info?.perTokenKVBytes(bytesPerElement: 2) ?? (256 * 1024)
        // `none`/`triattention` both leave KV numerics fp16 (TriAttention
        // evicts tokens, it does not quantize), so the per-token bound is
        // the fp16 geometry for every codec.
        self.perTokenKVBytes = fp16PerToken
    }

    /// The store directory for `name` (ADR 026): resolve via the shared
    /// store-layout helper so a bare name or full HF id both land, and the
    /// `pull` symlink is followed.
    private func directoryURL(for name: String) -> URL? {
        ModelStoreLayout.localDirectory(for: name, storeRoot: modelStoreRoot)
    }

    /// The store dirs of this slot's modality (ADR 026 live scan).
    private func storeModelIds() -> [String] {
        StoreModelClass.ids(
            storeRoot: modelStoreRoot, accept: { $0.isLLMSlot })
    }

    /// Resolve the default-or-throw name when a request omits `model`
    /// (cold-load / preload warm). ADR 026 ambiguity rule.
    private func resolvedDefaultName() throws -> String {
        let available = storeModelIds()
        switch ModelSelection.resolve(
            available: available, configuredDefault: configuredDefault,
            requested: nil)
        {
        case .resolved(let t): return t
        case .notAvailable:
            throw AthenaError.modelNotAvailable(
                requested: configuredDefault ?? "", available: available)
        case .ambiguous:
            throw AthenaError.ambiguousModel(
                module: .llm, available: available)
        }
    }

    /// Brief 4b: refuse a prompt whose KV/prompt-cache would exceed the
    /// governor-owned cap, before any generation, as a governed 503.
    public func preflightPromptCache(prompt: String) async throws {
        try await preflightPromptCache(
            messages: [ChatTurn(role: "user", content: prompt)])
    }

    public func preflightPromptCache(
        messages: [ChatTurn],
        tools: [[String: any Sendable]]? = nil,
        chatTemplateKwargs: [String: any Sendable]? = nil
    ) async throws {
        guard promptCacheCapBytes > 0, let container else { return }
        // NC3: render the SAME prompt generation will — tools + chat-template
        // kwargs included. Counting only the bare messages undercounts a
        // tool/kwargs-bearing request, so it could pass the cap here and then
        // exceed it during the real prefill, defeating the OOM guard for
        // exactly the large tool-augmented requests it exists to protect.
        let lmInput = try await container.prepare(
            input: UserInput(
                chat: Self.chatMessages(messages), tools: tools,
                additionalContext: chatTemplateKwargs))
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
            // M71.2 — carry any decoded images as `UserInput.Image` so the
            // VLM processor (e.g. Gemma4Processor) splices them. The bytes
            // were already validated decodable at the HTTP boundary
            // (`ChatImage.fromImageURL`), so `CIImage(data:)` is non-nil;
            // compactMap is belt-and-suspenders. A text-only model ignores
            // them (the serve path 400s an image to a non-vision model first).
            let images: [UserInput.Image] = turn.images.compactMap {
                CIImage(data: $0.data).map(UserInput.Image.ciImage)
            }
            return Chat.Message(
                role: Chat.Message.Role(rawValue: turn.role) ?? .user,
                content: turn.content, images: images)
        }
        return mapped.isEmpty ? [.user("")] : mapped
    }

    public var residentBytes: Int {
        container == nil ? 0 : estimatedBytes
    }

    /// M71.2 — true when the resident model accepts image inputs (loaded via
    /// the substrate VLM path). The serve path gates `image_url` content-parts
    /// on this: a vision request to a text-only model is a 400.
    public var servesVision: Bool { residentIsVision }

    public func memoryEstimate() -> Int { estimatedBytes }

    /// C10 (M68.2) — invalidate every per-model DERIVED structured-output
    /// cache in ONE place. The vocab build (C3) and the parser-factory share
    /// the resident model's tokenizer, so they must die together whenever the
    /// model changes; centralizing it stops a new cache field from being
    /// added to some invalidation sites and missed at others (the M53 drift
    /// that left `cachedStructuredVocabulary` nil'd at only some of the five
    /// former copy-pasted sites). Idempotent.
    private func resetStructuredCaches() {
        vocabBuild?.cancel()
        vocabBuild = nil
        cachedStructuredVocabulary = nil  // M53 — mirror the vocab lifecycle
    }

    /// C10 — drop the resident container and all derived per-model state in
    /// one place (unload / load-failure / rebind / allowlist-drop). Idempotent.
    private func dropResidentModel() {
        container = nil
        residentName = nil
        residentIsVision = false  // M71.2
        resetStructuredCaches()
    }

    /// C3 (M68.2) — the resident model's structured-output vocab bundle, built
    /// at most once (coalesced) against the resident container. Returns nil
    /// when no model is resident or the model's vocab size can't be resolved
    /// (the caller then fails the structured request closed — G4/NC2). The
    /// memoized `vocabBuild` Task is BOTH the cache (held until the model
    /// changes) and the coalescer (concurrent callers await the same Task).
    /// The nil-check and the assignment below are actor-atomic — there is no
    /// `await` between them — so exactly one build is ever started per model;
    /// the Task captures `container`, so a rebind landing mid-build still
    /// yields the correct vocab for the request that started it.
    private func structuredVocab() async throws -> VocabBundle? {
        if let existing = vocabBuild {
            return try await existing.value
        }
        guard let container else { return nil }
        let cfgVocab = configVocabSize
        let builtForName = residentName
        let task = Task<VocabBundle?, Error> {
            DecodeProgress.counter?.setSetupStage("build-vocab")
            defer { DecodeProgress.counter?.setSetupStage(nil) }
            let vocabT0 = Date()
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
            guard let built else { return nil }
            Self.log.notice(
                """
                structured-vocab built elapsed=\
                \(String(format: "%.1f", Date().timeIntervalSince(vocabT0)))s \
                tokens=\(built.0.count)
                """,
                metadata: ["function": "structuredVocab"])
            return (
                built.0, built.1,
                StructuredVocabulary.openerAliases(tokens: built.0))
        }
        vocabBuild = task  // actor-atomic with the nil-check (no await between)
        do {
            return try await task.value
        } catch {
            // A throwing build (e.g. the container faulted) must not stick as
            // the cached result and wedge every later request on the same
            // throw — drop it so a retry rebuilds. Only if no rebind replaced
            // it while we awaited (a rebind already invalidated `vocabBuild`).
            if residentName == builtForName { vocabBuild = nil }
            throw error
        }
    }

    public func load(reservation: MemoryReservation) async throws {
        if container != nil { return }
        // M62 — bind the requested cold-load target (set via
        // selectColdLoadModel) so a cold slot serves the requested model,
        // not the default. residentName is normally nil here (slot empty);
        // the resolved configured default is the final fallback (ADR 026 —
        // throws if the store is empty/ambiguous with no configured default).
        let name = try desiredName ?? residentName ?? resolvedDefaultName()
        try await loadModel(name: name)
    }

    public func unload() async {
        dropResidentModel()  // C10
    }

    /// M41.2 — load `name`'s directory into `container` and (re)seed the
    /// per-model state (KV geometry, vocab size, structured-output cache
    /// invalidation, registry's Qwen3.5 routing). Caller guarantees
    /// `name` is in the allowlist; container is left nil on failure so
    /// the next request reattempts.
    private func loadModel(name: String) async throws {
        guard let url = directoryURL(for: name) else {
            throw AthenaError.modelNotAvailable(
                requested: name, available: storeModelIds())
        }
        // Route Qwen3.5 directories to Athena's vendored model so the
        // substrate stays pristine. Idempotent; must precede the load.
        // Debug seam: ATHENA_DISABLE_VENDORED_MODEL=1 keeps the substrate's
        // own Qwen35 (used for the M2.1 deterministic parity A/B).
        if ProcessInfo.processInfo.environment[
            "ATHENA_DISABLE_VENDORED_MODEL"] != "1"
        {
            // Register the vendored creators (idempotent). The checkpoint
            // directory is conveyed to them per-load via the `withValue`
            // binding around `loadModelContainer` below (NC1), not a shared
            // global — so a concurrent queued convert can't clobber it.
            await AthenaModelRegistration.install()
        }
        guard
            FileManager.default.fileExists(
                atPath: url.appendingPathComponent("config.json").path)
        else {
            throw AthenaError.moduleLoadFailed(
                .llm,
                reason: "no model at \(url.path) (missing config.json)")
        }
        // Resolve the store-entry symlink: `athena pull` lands a model as
        // a symlink (~/.athena/models/<name> → HF snapshot dir). The
        // substrate's weight loader enumerates the directory but does NOT
        // follow a symlinked ROOT, so it would load ZERO shards → the model
        // fails with keyNotFound on its first parameter. Convert-produced
        // models are real dirs and were unaffected; every pulled model was.
        // Read config BEFORE the load: a `vision_config` routes the load
        // through the substrate's VLMModelFactory (M71.2) so the image tower
        // is built, not stripped. (The generic loadModelContainer tries
        // factories in registration order and the text LLMModelFactory would
        // win for gemma4 — so vision must be selected EXPLICITLY.)
        let info = ModelConfigInfo.read(modelDirectory: url)
        let isVision = info?.hasVisionConfig ?? false
        let loaded: ModelContainer
        do {
            // NC1: bind the checkpoint directory for the registry creators,
            // request-scoped, so the MTP-suppression decision reads THIS
            // load's directory even if a queued convert loads concurrently.
            loaded = try await AthenaModelRegistration.$currentModelDirectory
                .withValue(url) {
                    let resolved = url.resolvingSymlinksInPath()
                    let loader = #huggingFaceTokenizerLoader()
                    if isVision {
                        // Single VLM container serves BOTH text and image for a
                        // vision checkpoint (ADR 010/011: one resident copy).
                        return try await VLMModelFactory.shared.loadContainer(
                            from: resolved, using: loader)
                    }
                    return try await loadModelContainer(
                        from: resolved, using: loader)
                }
        } catch {
            dropResidentModel()  // C10
            throw error
        }
        configVocabSize = info?.vocabSize
        let fp16PerToken =
            info?.perTokenKVBytes(bytesPerElement: 2) ?? (256 * 1024)
        perTokenKVBytes = fp16PerToken
        // Different model ⇒ a fresh structured-vocab cache; the old
        // tokens belong to the previous tokenizer (C3/C10).
        resetStructuredCaches()
        container = loaded
        residentName = name
        residentIsVision = isVision  // M71.2
    }

    // M41 / ADR 026 — ModelSelectable over the store-classified selectable set.
    public func allowedModelIds() -> [String] { storeModelIds() }
    public func defaultModelId() -> String {
        ModelSelection.displayDefault(
            available: storeModelIds(), configuredDefault: configuredDefault)
    }
    public func residentModelId() -> String? {
        container == nil ? nil : residentName
    }
    public func rebind(to id: String?) async throws {
        let target = try resolve(id)
        if residentName == target, container != nil { return }
        // Drop the current container (and its caches) before swapping
        // so the substrate's working set is released before the new
        // load — same fixed governor reservation either way (C10).
        dropResidentModel()
        try await loadModel(name: target)
    }

    public func selectColdLoadModel(_ id: String?) async throws {
        desiredName = try resolve(id)
    }

    /// ADR 026 resolution against the live store scan: a named id matches by
    /// store-dir identity (bare name OR full HF id, case-insensitive); an
    /// omitted id resolves the configured default / sole model / ambiguity.
    private func resolve(_ id: String?) throws -> String {
        let available = storeModelIds()
        switch ModelSelection.resolve(
            available: available, configuredDefault: configuredDefault,
            requested: id)
        {
        case .resolved(let t): return t
        case .notAvailable:
            throw AthenaError.modelNotAvailable(
                requested: id ?? (configuredDefault ?? ""),
                available: available)
        case .ambiguous:
            throw AthenaError.ambiguousModel(
                module: .llm, available: available)
        }
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
            maxTokens: nil, temperature: nil, speculative: nil)
    }

    public nonisolated func generateMetered(
        messages: [ChatTurn], schemaJSON: String?,
        tools: [[String: any Sendable]]?,
        maxTokens: Int?, temperature: Double?,
        topP: Double?, seed: Int?,
        speculative: Bool?,
        chatTemplateKwargs: [String: any Sendable]?,
        promptCacheKey: String? = nil,
        principal: String? = nil,
        logprobs: LogprobsRequest? = nil
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
                        tools: tools, maxTokens: maxTokens,
                        requestSpeculative: speculative,
                        requestTemperature: temperature,
                        requestTopP: topP, requestSeed: seed,
                        chatTemplateKwargs: chatTemplateKwargs,
                        promptCacheKey: promptCacheKey, principal: principal,
                        logprobs: logprobs)
                    {
                        usage = speculative.usage
                        continuation.yield(.text(speculative.text))
                        continuation.yield(.usage(usage))
                        // C2 (ADR 013 §4): per-token logprobs when requested.
                        if let lps = speculative.logprobs {
                            continuation.yield(.logprobs(lps))
                        }
                    } else {
                        // runSpeculative returns nil only for UNstructured
                        // requests (no schema) — those stream from the
                        // standard substrate path. Structured requests are
                        // always guided: the vendored Qwen3.5 path, or the
                        // substrate-guided path for other arches (M23).
                        let stream = try await self.beginGeneration(
                            messages: messages, tools: tools,
                            maxTokens: maxTokens, temperature: temperature,
                            topP: topP, seed: seed,
                            chatTemplateKwargs: chatTemplateKwargs)
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
                    // M49.5.2 — classify and yield as a real .error event
                    // so the consumer can re-throw and the HTTP layer can
                    // map to the right status (e.g. 503 metal_oom, 504
                    // inference_timeout, 400 model_not_available). Pre-
                    // M49.5.2 this swallowed the throw into a fake .text
                    // response and the request returned 200 with the error
                    // stringified into the chat content — confirmed when a
                    // classified 400 (the v0.10.84 schema-complexity refusal,
                    // since removed in M53) landed in the response body
                    // instead of being a real 400.
                    let classified =
                        (error as? AthenaError)
                        ?? AthenaError.classify(error, module: .llm)
                    continuation.yield(.error(classified))
                }
                continuation.finish()
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    /// MTP speculative path — greedy at temp == 0 (bit-identical to the
    /// non-speculative greedy path) AND sampling-mode at temp > 0
    /// (Leviathan/Chen; distributionally identical to non-speculative
    /// sampling at the same temp/top_p/seed). Returns the full decoded
    /// text, or nil to fall back to the unconstrained substrate stream.
    /// A structured request (`schemaJSON`) ALWAYS takes a guided path
    /// (greedy): the MTP speculative loop when eligible (faster), else
    /// a plain guided-greedy loop. An unstructured request takes the
    /// opt-in speculative path only; sampling-mode is unstructured-only
    /// (the Guide masks to one valid token, so sampling has no meaning
    /// there).
    private func runSpeculative(
        messages: [ChatTurn], schemaJSON: String?,
        tools: [[String: any Sendable]]?, maxTokens: Int?,
        requestSpeculative: Bool?,
        requestTemperature: Double?,
        requestTopP: Double?,
        requestSeed: Int?,
        chatTemplateKwargs: [String: any Sendable]?,
        promptCacheKey: String? = nil,
        principal: String? = nil,
        logprobs: LogprobsRequest? = nil
    ) async throws -> (text: String, usage: TokenUsage, logprobs: [TokenLogprob]?)? {
        guard let container else { return nil }
        // C2 (ADR 013 §4) — per-token logprob capture sink. Non-nil only when
        // the caller asked for logprobs; the server has already enforced that
        // this is a deterministic (greedy/structured, temp==0) request, so the
        // capture routes through the ArgMax pick/processor seams below.
        let logprobSink = logprobs.map { LogprobSink(topLogprobs: $0.topLogprobs) }
        // Per-request override (the consuming application intent): if the caller
        // explicitly passes `speculative=true/false`, that wins for THIS
        // request; if nil, fall back to the daemon's loaded default. An
        // opt-in `speculative=true` without an explicit temperature
        // stays implicitly greedy (temperature 0) so the older
        // "speculative implies greedy" contract is preserved through
        // M40.2; sampling-mode requires both `speculative=true` AND an
        // explicit `temperature > 0`.
        let effectiveSpec = requestSpeculative ?? params.speculative
        let effectiveTemp: Double =
            requestTemperature
            ?? (requestSpeculative == true ? 0.0 : Double(params.temperature))
        // M48.3 — temperature is INERT under a Guide: the schema mask
        // collapses every position's distribution to its allowed set,
        // and the greedy speculative loop (SpeculativeGeneration) picks
        // the masked argmax regardless of the temp the caller asked for.
        // So a structured request with `speculative: true` should engage
        // the speculative loop whatever the temperature, not be forced
        // into the non-speculative GuidedGreedy path just because the
        // caller passed `temperature: 0.1`. Pre-M48.3 the gate was
        // `effectiveSpec && effectiveTemp == 0`, which silently dropped
        // every the consuming application-style request (`spec=true temp=0.1
        // schema=true`) to GuidedGreedy and gave up M47.2's speculative
        // win. The bit-identical-greedy contract is preserved — both
        // paths produce the same masked-argmax sequence; only the speed
        // changes.
        let greedyEligible =
            effectiveSpec
            && (effectiveTemp == 0 || schemaJSON != nil)
        // Sampling-mode speculative covers temp > 0 unstructured requests.
        // Structured/guided is out of scope (the Guide masks to one valid
        // token; sampling has no meaning), so `schemaJSON == nil` is the
        // hard precondition; the guided path keeps using the greedy
        // speculative loop above when eligible. M40.3.
        let samplingEligible =
            effectiveSpec && effectiveTemp > 0 && schemaJSON == nil
        // C2: a logprobs request must NOT fall through to the substrate stream
        // (beginGeneration has no logit-capture seam). Keep it on this path so
        // the closure below routes it to GuidedGreedy/GuidedSubstrate capture.
        if schemaJSON == nil && !greedyEligible && !samplingEligible
            && logprobSink == nil
        {
            Self.log.debug(
                """
                dispatch path=substrate-stream spec=\(effectiveSpec) \
                temp=\(effectiveTemp) schema=false
                """,
                metadata: ["function": "runSpeculative"])
            return nil
        }
        // M48.2 — declare which internal generate path this request
        // will take, BEFORE any model work begins. Lets operators
        // (and the consuming application) see at a glance whether a structured
        // request engaged the speculative loop or fell to the
        // non-speculative GuidedGreedy/GuidedSubstrate path. The
        // final architecture-dispatched branch (Qwen3.5 vendored vs
        // any-other) is resolved inside container.perform; the
        // selected path identity is correct either way because the
        // any-other branch always lands at GuidedSubstrate when a
        // schema is present.
        let dispatchPath: String = {
            if greedyEligible { return "speculative-greedy" }
            if samplingEligible { return "speculative-sampling" }
            return "guided-greedy-or-substrate"
        }()
        Self.log.debug(
            """
            dispatch path=\(dispatchPath) spec=\(effectiveSpec) \
            temp=\(effectiveTemp) schema=\(schemaJSON != nil)
            """,
            metadata: ["function": "runSpeculative"])

        let lmInput = try await container.prepare(
            input: UserInput(
                chat: Self.chatMessages(messages), tools: tools,
                additionalContext: chatTemplateKwargs))
        let promptTokens = lmInput.text.tokens.asArray(Int.self)
        // M24.3: a positive per-request override wins over the loaded
        // default; the greedy/MTP paths are length-only (temperature is
        // inert under the Guide / speculative greedy).
        let maxTokens = Self.effectiveMaxTokens(maxTokens, params.maxTokens)

        // Build (or reuse) the structured vocabulary tokens once per model.
        // The ~150k tokenizer.decode calls are the dominant structured-request
        // cost and are schema-independent. C3 (M68.2): the build is memoized
        // as a Task (`structuredVocab()`) so concurrent first-of-model
        // requests coalesce onto ONE build and a rebind can't clobber an
        // in-flight build's result with a stale tokenizer.
        let vocabTokens = schemaJSON != nil ? try await structuredVocab() : nil
        // NC2/G4 fail-closed: a structured request whose guide cannot be
        // built must 400 — never stream unconstrained text with a 200. If
        // the resident model's vocabulary can't be resolved (e.g. a
        // non-Qwen substrate arch whose config.json lacks vocab_size),
        // refuse here, before either the MTP or substrate dispatch branch
        // can run guide-less.
        if schemaJSON != nil, vocabTokens == nil {
            throw AthenaError.structuredOutputUnavailable(
                detail: "the resident model's vocabulary could not be "
                    + "resolved to enforce the requested JSON schema")
        }

        // M53 — build (once per model) the structured vocabulary + parser
        // factory, then a FRESH per-request `StructuredIndex` from it. The
        // factory build (~0.24 s, the only non-trivial cost) is pure CPU
        // and schema-independent, so it lives outside the main
        // `container.perform { ... }` block and is cached across requests
        // (`StructuredVocabulary` is `@unchecked Sendable` — its factory is
        // immutable and internally `Arc`-shared, so guides can be spawned
        // from it concurrently). The per-schema `StructuredIndex` is now
        // ~1 ms (vs the old ~60 s outlines DFA compile), so it is built per
        // request rather than cached — and llguidance parses incrementally,
        // so a `maxItems`-bounded schema can no longer blow up memory (the
        // old complexity gate is gone).
        let structuredIndex: StructuredIndex? = try {
            guard let schemaJSON, let vt = vocabTokens else {
                return nil
            }
            let vocabulary: StructuredVocabulary
            if let cached = cachedStructuredVocabulary {
                vocabulary = cached
            } else {
                // M53: annotate the heartbeat as `setup:build-factory` for
                // the one-time per-model vocab-slicer build.
                DecodeProgress.counter?.setSetupStage("build-factory")
                defer { DecodeProgress.counter?.setSetupStage(nil) }
                let factoryT0 = Date()
                vocabulary = try StructuredVocabulary(
                    tokens: vt.tokens, eosTokenId: vt.eos)
                cachedStructuredVocabulary = vocabulary
                Self.log.notice(
                    """
                    structured factory built elapsed=\
                    \(String(format: "%.2f", Date().timeIntervalSince(factoryT0)))s \
                    vocab_tokens=\(vt.tokens.count)
                    """,
                    metadata: ["function": "runSpeculative"])
            }
            // A client schema that won't compile is a 400, not a 500 or a
            // silent unconstrained fall-through (G4/NC2).
            do {
                return try StructuredIndex(
                    jsonSchema: schemaJSON, vocabulary: vocabulary)
            } catch {
                throw AthenaError.structuredOutputUnavailable(
                    detail:
                        "the requested JSON schema could not be compiled: "
                        + "\(error)")
            }
        }()

        let cfgVocab = configVocabSize
        // Resolve sampling knobs for the M40.2 sampling-mode branch
        // outside the (non-isolated) closure: same positive-wins
        // resolution the standard substrate path uses, expressed in
        // the types the pure-math helper wants. Inert when the branch
        // doesn't engage (greedy path ignores them).
        let samplingTemp = Float(effectiveTemp)
        let samplingTopP: Float? = {
            if let t = requestTopP, t > 0, t < 1 { return Float(t) }
            if params.topP > 0, params.topP < 1 { return params.topP }
            return nil
        }()
        let samplingSeed: Int? =
            requestSeed.flatMap { $0 >= 0 ? $0 : nil }
        // M59.1/.3 — prefix-cache handle + scope captured for the closure.
        // `prefixCache` is nil unless the operator enabled `[prompt_cache]`.
        // The scope ALWAYS includes the resident model id (a rebind must not
        // serve one model's KV to another) plus, per the configured scope
        // mode, the authenticated principal (default — cross-principal reuse
        // is never allowed) and/or the OpenAI `prompt_cache_key` hint.
        let prefixCache = self.prefixCache
        let cacheScope: String? = prefixCache.map {
            $0.scopeKey(
                model: residentName ?? (configuredDefault ?? ""),
                principal: principal, cacheKey: promptCacheKey)
        }
        // The closure returns the decoded text, the completion token count
        // (`ids.count`), and the cached-prefix token count (M59.3); the
        // prompt count is `promptTokens.count` from the outer scope. nil ⇒
        // fall back to the substrate stream.
        let decoded = try await container.perform {
            (ctx: ModelContext) -> (
                text: String, completion: Int, cached: Int,
                logprobs: [TokenLogprob]?
            )? in
            // C2: decode the captured raw logprobs (ids → token strings/bytes)
            // here, where the tokenizer is in scope. nil sink ⇒ nil.
            func builtLogprobs() -> [TokenLogprob]? {
                guard let s = logprobSink else { return nil }
                return Self.buildLogprobs(s) { ctx.tokenizer.decode(tokenIds: $0) }
            }

            // Structured ⇒ NO-THINK by construction: the Guide masks
            // from token 0, so the schema is enforced immediately and
            // the model's <think>…</think> is suppressed (matches
            // the consuming application's enable_thinking=False production config).
            // Schema-enforced output that ALSO permits a thinking prefix
            // (deferred enforcement, Patch 6) is tracked in
            // GoodOlClint/athena#2.
            // M49.1 — `structuredIndex` is the cached/compiled DFA
            // captured from the outer scope. `StructuredGuide` wraps
            // it with per-request walker state (advance / rollback /
            // mask buffer). The DFA itself is reused; only the
            // stateful walker is fresh per request.
            func makeGuide() throws -> StructuredGuide? {
                guard let index = structuredIndex,
                    let vt = vocabTokens
                else { return nil }
                let g = try StructuredGuide(index: index)
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
                // Nothing to clear — this path never binds
                // `TriAttentionRequestPolicy.current`, so the caches these
                // generators build (`newCache(parameters: nil)`) read the
                // task-local's default `nil` ⇒ `KVCacheSimple` (NF2).
                let ids: [Int]
                var cachedTokens = 0
                if logprobSink == nil && greedyEligible && model.hasMTPHead {
                    let r = SpeculativeGeneration.generate(
                        model: model, promptTokens: promptTokens,
                        maxTokens: maxTokens,
                        eosTokenId: ctx.tokenizer.eosTokenId, guide: guide,
                        prefixCache: prefixCache, cacheScope: cacheScope)
                    ids = r.ids
                    cachedTokens = r.cachedTokens
                } else if logprobSink == nil && samplingEligible
                    && model.hasMTPHead
                {
                    // M40.2 sampling-mode (internal). The Guide is nil
                    // here by construction — `samplingEligible` requires
                    // `schemaJSON == nil`.
                    ids = SpeculativeSampling.generate(
                        model: model, promptTokens: promptTokens,
                        maxTokens: maxTokens,
                        eosTokenId: ctx.tokenizer.eosTokenId,
                        temperature: samplingTemp,
                        topP: samplingTopP, seed: samplingSeed)
                } else if guide != nil || logprobSink != nil {
                    // Structured, OR a logprobs request (C2): force the
                    // non-speculative greedy path so each token has clean
                    // last-position logits to capture (bit-identical output;
                    // `guide` may be nil ⇒ plain greedy capture).
                    ids = GuidedGreedy.generate(
                        model: model, promptTokens: promptTokens,
                        maxTokens: maxTokens,
                        eosTokenId: ctx.tokenizer.eosTokenId, guide: guide,
                        sink: logprobSink)
                } else {
                    return nil  // unstructured + no MTP ⇒ substrate stream
                }
                return (
                    ctx.tokenizer.decode(tokenIds: ids), ids.count,
                    cachedTokens, builtLogprobs())
            }

            // Any other architecture: schema-guided decoding on the
            // substrate generation path (M23 fork A). Unstructured
            // requests with no logprobs (no guide, no sink) return nil → the
            // standard substrate stream in beginGeneration. A logprobs request
            // (C2) routes here with `guide` possibly nil — plain greedy capture
            // via the LogitProcessor seam (no beginGeneration, which has no
            // capture hook). MTP speculative does not apply (no mtp.* weights).
            if guide == nil && logprobSink == nil { return nil }
            // Guided needs the structured vocab (eos + vocab size); unguided
            // capture uses the tokenizer EOS and doesn't mask, so vocab is
            // unused there (pass 0).
            let substrateEos = vocabTokens.map { Int($0.eos) }
                ?? ctx.tokenizer.eosTokenId
            let substrateVocab = cfgVocab ?? 0
            if guide != nil && cfgVocab == nil { return nil }  // can't mask
            let ids = try GuidedSubstrate.generate(
                model: ctx.model, promptTokens: promptTokens,
                vocab: substrateVocab, maxTokens: maxTokens,
                eosTokenId: substrateEos, guide: guide, sink: logprobSink)
            return (
                ctx.tokenizer.decode(tokenIds: ids), ids.count, 0,
                builtLogprobs())
        }
        guard let decoded else { return nil }
        return (
            decoded.text,
            TokenUsage(
                promptTokens: promptTokens.count,
                completionTokens: decoded.completion,
                cachedTokens: decoded.cached),
            decoded.logprobs)
    }

    /// C2 — turn a `LogprobSink`'s numeric captures into `[TokenLogprob]` by
    /// decoding each token id to its string/bytes via `decode` (the model's
    /// tokenizer, passed as a closure so this stays MLX/tokenizer-type-free).
    private static func buildLogprobs(
        _ sink: LogprobSink, decode: ([Int]) -> String
    ) -> [TokenLogprob] {
        sink.committed.map { raw in
            let tok = decode([raw.chosen])
            let top = raw.top.map { alt -> TopLogprob in
                let s = decode([alt.token])
                return TopLogprob(
                    token: s, logprob: alt.logprob,
                    bytes: Array(s.utf8).map(Int.init))
            }
            return TokenLogprob(
                token: tok, logprob: raw.logprob,
                bytes: Array(tok.utf8).map(Int.init), top: top)
        }
    }

    private func beginGeneration(
        messages: [ChatTurn], tools: [[String: any Sendable]]?,
        maxTokens: Int?, temperature: Double?,
        topP: Double?, seed: Int?,
        chatTemplateKwargs: [String: any Sendable]?
    ) async throws -> AsyncStream<Generation> {
        guard let container else {
            throw AthenaError.moduleLoadFailed(
                .llm, reason: "generate called before load")
        }
        // The standard attention path is the ONLY place TriAttention
        // eviction applies. NF2: bind the policy as a request-scoped
        // task-local around prepare+generate (below) rather than stashing
        // it on the shared model instance — the substrate's
        // newCache(parameters:) reads it when it builds the per-layer
        // caches. nil (any non-triattention knob) ⇒ KVCacheSimple. The
        // `sending` UserInput/LMInput are constructed and consumed entirely
        // inside the closure so no non-Sendable value crosses its boundary.
        let evictionPolicy = params.kvCompression.eviction
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
        // C11: a per-request seed makes temp>0 sampling reproducible by
        // seeding THIS request's sampler RNG (GenerateParameters.seed →
        // TopPSampler/CategoricalSampler RandomState(seed:)), not the
        // process-global `MLXRandom.seed`. The substrate samplers each hold
        // a private RandomState and ignore the global state, so the old
        // global seed was both ineffective for them AND a cross-request race
        // (a concurrent temp>0 request reseeding the shared global between
        // this request's seed-set and its sampler construction). Per-request
        // GenerateParameters.seed is race-free by construction. Inert at
        // temp==0 (argmax).
        let requestSeed: UInt64? = seed.flatMap {
            $0 >= 0 ? UInt64($0) : nil
        }
        let gp = GenerateParameters(
            maxTokens: Self.effectiveMaxTokens(maxTokens, params.maxTokens),
            temperature: temp,
            topP: tp,
            seed: requestSeed)
        // NF2: the eviction policy is visible to the substrate's eager,
        // same-Task `newCache` call inside `generate`; the deferred decode
        // loop never needs it, so binding around this call is sufficient.
        return try await TriAttentionRequestPolicy.$current.withValue(
            evictionPolicy
        ) {
            let userInput = UserInput(
                chat: Self.chatMessages(messages), tools: tools,
                additionalContext: chatTemplateKwargs)
            let lmInput = try await container.prepare(input: userInput)
            return try await container.generate(
                input: lmInput, parameters: gp)
        }
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
