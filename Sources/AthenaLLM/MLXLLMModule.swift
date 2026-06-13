import AthenaCore
import AthenaModels
import AthenaStructured
import Foundation
import HuggingFace
import Logging
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
/// `@unchecked Sendable` holder so the (non-Sendable, MLXArray-bearing)
/// DFlash drafter can cross into the `container.perform` `@Sendable` closure
/// — the same pattern the module uses for `StructuredVocabulary`. The actor
/// owns the single instance; the closure only reads it under the actor's
/// serialized model domain. M63.3b.
final class DFlashDraftBox: @unchecked Sendable {
    let model: DFlashDraftModel
    init(_ model: DFlashDraftModel) { self.model = model }
}

public actor MLXLLMModule: LLMModule, ModelSelectable {
    public nonisolated let id: ModuleID = .llm
    public nonisolated var moduleID: ModuleID { .llm }

    /// M48.2 — daemon-side logger for the dispatch decision (which
    /// internal generate path each request takes). `.debug` so
    /// production stays quiet by default; flip per-subsystem via
    /// `sudo log config --mode "level:debug" --subsystem athena` when
    /// diagnosing a specific request shape.
    nonisolated static let log = Logger(label: "athena.llm")

    /// M41.2: operator-declared allowlist as store-name → directory URL.
    /// First-declared = the default (loaded when the slot first comes up
    /// or when a request omits `model`). A request whose `model` is
    /// outside this map is `modelNotAvailable` (400) — never a silent
    /// fallback, never an on-request download.
    /// M42.2: mutable so the persistent allowlist can be pushed in at
    /// runtime; `modelStoreRoot` lets us resolve a newly-allowed name
    /// to its directory without operator restart.
    private var modelDirectories: [(name: String, url: URL)]
    private var defaultName: String
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
    /// M62 — the model the NEXT cold `load(reservation:)` will bind, set by
    /// `selectColdLoadModel` before the governor's non-blocking cold-load is
    /// kicked off. nil ⇒ bind the default. Only consulted while the slot is
    /// unloaded; a warm swap goes through `rebind`.
    private var desiredName: String?
    /// Cached structured-output vocabulary tokens (the ~150k
    /// `tokenizer.decode` calls are model-fixed and schema-independent —
    /// build once, reuse every structured request). Sendable, so it
    /// crosses into the `container.perform` closure safely. M41.2:
    /// invalidated on rebind — different models have different vocabs.
    private var cachedVocabTokens:
        (tokens: [VocabToken], eos: UInt32, opener: [UInt32: UInt32])?

    /// M53 — cached per-model structured-output vocabulary + parser
    /// factory (the llguidance token trie + vocab slicer). Building it
    /// (~0.24 s) is the only non-trivial structured-output cost and is
    /// schema-independent, so it is built once from `cachedVocabTokens`
    /// and reused for every schema and every request. Invalidated
    /// wherever `cachedVocabTokens` is — a rebind changes the vocabulary.
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
    /// constant when the dims are absent. TurboQuant stores ~4-bit K/V so
    /// it gets a quarter of the fp16 bound (still over-estimates — the
    /// safe direction for an OOM guard). M20.2 / M23 fork C.
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

    /// M63.3b — DFlash lossless speculative decoding (default off). When
    /// enabled, an unguided greedy request to a target with a registered
    /// drafter (`DFlashRegistry`) decodes through the block draft/verify
    /// engine. The drafter is operator-pulled from Hugging Face on first
    /// use and held here, boxed `@unchecked Sendable` so it crosses into the
    /// `container.perform` closure; cleared on rebind. `dflashDraftBytes`
    /// feeds `residentBytes` so the governor accounts for the loaded drafter.
    private let dflashEnabled: Bool
    private var dflashDraft: DFlashDraftBox?
    private var dflashDraftFor: String?
    private var dflashDraftBytes: Int = 0

    /// Single-model convenience init kept for source-compat with M27/M40
    /// call sites; forwards to the M41.2 list form with a 1-entry
    /// allowlist. The store root is inferred from the directory's
    /// parent (so new ids added via /api/models/allow at runtime
    /// resolve to siblings of the seed model).
    public init(
        modelDirectory: URL,
        parameters: LLMGenerationParameters = .init(),
        promptCacheCapBytes: Int = 0,
        prefixCache: PrefixKVCache? = nil,
        dflashEnabled: Bool = false
    ) {
        self.init(
            modelDirectories: [modelDirectory],
            modelStoreRoot: modelDirectory.deletingLastPathComponent(),
            parameters: parameters,
            promptCacheCapBytes: promptCacheCapBytes,
            prefixCache: prefixCache,
            dflashEnabled: dflashEnabled)
    }

    /// M41.2: operator-declared list (first = default). Empty ⇒
    /// precondition trap — every daemon must declare at least one LLM.
    /// `modelStoreRoot` (M42.2) is the directory that hosts every
    /// declared model and any later additions via /api/models/allow;
    /// defaults to the parent of `urls[0]` for source-compat.
    public init(
        modelDirectories urls: [URL],
        modelStoreRoot: URL? = nil,
        parameters: LLMGenerationParameters = .init(),
        promptCacheCapBytes: Int = 0,
        prefixCache: PrefixKVCache? = nil,
        dflashEnabled: Bool = false
    ) {
        precondition(
            !urls.isEmpty,
            "MLXLLMModule needs at least one model directory")
        self.prefixCache = prefixCache
        self.dflashEnabled = dflashEnabled
        self.modelDirectories = urls.map {
            (name: $0.lastPathComponent, url: $0)
        }
        self.defaultName = self.modelDirectories[0].name
        self.modelStoreRoot =
            modelStoreRoot ?? urls[0].deletingLastPathComponent()
        self.params = parameters
        // MAX across the allowlist so the slot still bounds the largest
        // declared member after a rebind.
        self.estimatedBytes = urls.lazy
            .map { Self.estimateBytes(forModelAt: $0) }
            .max() ?? 0
        self.promptCacheCapBytes = promptCacheCapBytes

        // Initial cap geometry seeded from the DEFAULT model; rebind
        // recomputes from the new model's config.
        let info = ModelConfigInfo.read(modelDirectory: urls[0])
        self.configVocabSize = info?.vocabSize
        let fp16PerToken =
            info?.perTokenKVBytes(bytesPerElement: 2) ?? (256 * 1024)
        switch parameters.kvCompression {
        case .turboquant: self.perTokenKVBytes = max(1, fp16PerToken / 4)
        case .none, .triattention: self.perTokenKVBytes = fp16PerToken
        }
    }

    private func directoryURL(for name: String) -> URL? {
        modelDirectories.first { $0.name == name }?.url
    }

    /// NC11: returns `URL?` and falls back via `.first?.url`, never a
    /// literal `[0]` subscript — `setAllowedModelIds([])` (empty DB
    /// allowlist) leaves `modelDirectories` empty while `defaultName` is
    /// stale, so a `[0]` access would trap. (Currently uncalled; this keeps
    /// it from becoming an allowlist-API-reachable crash if it is wired up.)
    private var residentDirectory: URL? {
        directoryURL(for: residentName ?? defaultName)
            ?? modelDirectories.first?.url
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
            Chat.Message(
                role: Chat.Message.Role(rawValue: turn.role) ?? .user,
                content: turn.content)
        }
        return mapped.isEmpty ? [.user("")] : mapped
    }

    public var residentBytes: Int {
        container == nil ? 0 : estimatedBytes + dflashDraftBytes
    }

    public func memoryEstimate() -> Int { estimatedBytes }

    public func load(reservation: MemoryReservation) async throws {
        if container != nil { return }
        // M62 — bind the requested cold-load target (set via
        // selectColdLoadModel) so a cold slot serves the requested model,
        // not the default. residentName is normally nil here (slot empty);
        // defaultName is the final fallback.
        try await loadModel(name: desiredName ?? residentName ?? defaultName)
    }

    public func unload() async {
        container = nil
        residentName = nil
        cachedVocabTokens = nil
        cachedStructuredVocabulary = nil  // M53 — mirror vocabTokens lifecycle
    }

    /// M41.2 — load `name`'s directory into `container` and (re)seed the
    /// per-model state (KV geometry, vocab size, structured-output cache
    /// invalidation, registry's Qwen3.5 routing). Caller guarantees
    /// `name` is in the allowlist; container is left nil on failure so
    /// the next request reattempts.
    private func loadModel(name: String) async throws {
        guard let url = directoryURL(for: name) else {
            throw AthenaError.modelNotAvailable(
                requested: name,
                available: modelDirectories.map { $0.name })
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
        let loaded: ModelContainer
        do {
            // NC1: bind the checkpoint directory for the registry creators,
            // request-scoped, so the MTP-suppression decision reads THIS
            // load's directory even if a queued convert loads concurrently.
            loaded = try await AthenaModelRegistration.$currentModelDirectory
                .withValue(url) {
                    try await loadModelContainer(
                        from: url.resolvingSymlinksInPath(),
                        using: #huggingFaceTokenizerLoader())
                }
        } catch {
            container = nil
            residentName = nil
            cachedVocabTokens = nil
        cachedStructuredVocabulary = nil  // M53 — mirror vocabTokens lifecycle
            throw error
        }
        // Refresh per-model geometry: vocab size + KV per-token bound
        // for THIS model's config.json (each rebind may change arch).
        let info = ModelConfigInfo.read(modelDirectory: url)
        configVocabSize = info?.vocabSize
        let fp16PerToken =
            info?.perTokenKVBytes(bytesPerElement: 2) ?? (256 * 1024)
        switch params.kvCompression {
        case .turboquant: perTokenKVBytes = max(1, fp16PerToken / 4)
        case .none, .triattention: perTokenKVBytes = fp16PerToken
        }
        // Different model ⇒ a fresh structured-vocab cache; the old
        // tokens belong to the previous tokenizer.
        cachedVocabTokens = nil
        cachedStructuredVocabulary = nil  // M53 — mirror vocabTokens lifecycle
        // M63.3b — the drafter is target-specific; drop it on rebind so the
        // next DFlash request reloads the one registered for the new model.
        dflashDraft = nil
        dflashDraftFor = nil
        dflashDraftBytes = 0
        container = loaded
        residentName = name
    }

    /// M63.3b — ensure the DFlash drafter registered for the resident model
    /// is loaded, returning it boxed for the `perform` closure. Returns nil
    /// when DFlash is off, no model is resident, or no drafter is registered
    /// for it — in which case the request decodes normally. The drafter is
    /// fetched from Hugging Face on first use (the passive-oracle weight-fetch
    /// carve-out) and cached until rebind.
    private func ensureDFlashDraft() async throws -> DFlashDraftBox? {
        guard dflashEnabled, let name = residentName,
            let draftId = DFlashRegistry.draftId(forModel: name)
        else { return nil }
        if let box = dflashDraft, dflashDraftFor == name { return box }
        let dir = try await #hubDownloader(
            HuggingFace.HubClient(session: AthenaProxy.proxiedURLSession())
        ).download(
            id: draftId, revision: nil,
            matching: ["*.json", "*.safetensors"],
            useLatest: false, progressHandler: { _ in })
        let model = try DFlashDraftLoader.load(directory: dir)
        let box = DFlashDraftBox(model)
        dflashDraft = box
        dflashDraftFor = name
        dflashDraftBytes = Self.directoryWeightBytes(dir)
        Self.log.notice(
            "DFlash drafter loaded id=\(draftId) for=\(name) bytes=\(dflashDraftBytes)",
            metadata: ["function": "ensureDFlashDraft"])
        return box
    }

    /// Sum of the `*.safetensors` file sizes in a checkpoint dir — the
    /// drafter's resident-byte estimate for the governor.
    private static func directoryWeightBytes(_ dir: URL) -> Int {
        guard
            let items = try? FileManager.default.contentsOfDirectory(
                at: dir, includingPropertiesForKeys: [.fileSizeKey])
        else { return 0 }
        return items
            .filter { $0.pathExtension == "safetensors" }
            .reduce(0) {
                $0 + ((try? $1.resourceValues(forKeys: [.fileSizeKey]))?
                    .fileSize ?? 0)
            }
    }

    // M41 — ModelSelectable. M41.2 generalizes to a repeatable
    // `--llm-model` allowlist (M41.1 shipped the protocol shape with a
    // single-id allowlist).
    public func allowedModelIds() -> [String] {
        modelDirectories.map { $0.name }
    }
    public func defaultModelId() -> String { defaultName }
    public func residentModelId() -> String? {
        container == nil ? nil : residentName
    }
    public func rebind(to id: String?) async throws {
        let requested = id ?? defaultName
        let allowed = modelDirectories.map { $0.name }
        // NE5 — resolve by store-dir identity so a request naming the model
        // by either its bare store-dir name OR its full HF org/name id
        // resolves uniformly, matching the embedding/transcription modules
        // (and case-insensitive, so `foo-4b` still finds `foo-4B`). The
        // CANONICAL stored id drives every downstream step.
        guard let target =
            allowed.canonicalByStoreIdentity(requested)
        else {
            throw AthenaError.modelNotAvailable(
                requested: requested, available: allowed)
        }
        if residentName == target, container != nil { return }
        // Drop the current container (and its caches) before swapping
        // so the substrate's working set is released before the new
        // load — same fixed governor reservation either way.
        container = nil
        residentName = nil
        cachedVocabTokens = nil
        cachedStructuredVocabulary = nil  // M53 — mirror vocabTokens lifecycle
        try await loadModel(name: target)
    }

    public func selectColdLoadModel(_ id: String?) async throws {
        guard let id, !id.isEmpty else { desiredName = nil; return }
        let allowed = modelDirectories.map { $0.name }
        // Same store-identity discipline as rebind (NE5).
        guard let target = allowed.canonicalByStoreIdentity(id) else {
            throw AthenaError.modelNotAvailable(
                requested: id, available: allowed)
        }
        desiredName = target
    }

    public func setAllowedModelIds(_ ids: [String]) {
        // Resolve every name to a URL under the model store root. An
        // absolute path (rare; an operator who pre-knows the directory)
        // is honored as-is. New names get a fresh URL; existing names
        // keep their URL.
        let existing = Dictionary(
            uniqueKeysWithValues: modelDirectories.map { ($0.name, $0.url) })
        modelDirectories = ids.map { name in
            let url =
                existing[name]
                ?? (name.hasPrefix("/")
                    ? URL(fileURLWithPath: name, isDirectory: true)
                    : modelStoreRoot.appendingPathComponent(
                        name, isDirectory: true))
            return (name: name, url: url)
        }
        defaultName = modelDirectories.first?.name ?? defaultName
        if let r = residentName, !ids.contains(r) {
            container = nil
            residentName = nil
            cachedVocabTokens = nil
        cachedStructuredVocabulary = nil  // M53 — mirror vocabTokens lifecycle
        }
        // M62 — drop a stale cold-load target no longer in the allowlist.
        if let d = desiredName, !ids.contains(d) { desiredName = nil }
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
        principal: String? = nil
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
                        promptCacheKey: promptCacheKey, principal: principal)
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
        principal: String? = nil
    ) async throws -> (text: String, usage: TokenUsage)? {
        guard let container else { return nil }
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
        if schemaJSON == nil && !greedyEligible && !samplingEligible {
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

        // Build (or reuse) the structured vocabulary tokens once per
        // model. The ~150k tokenizer.decode calls are the dominant
        // structured-request cost and are schema-independent.
        // M49.3: annotate the setup stage so the heartbeat reports
        // `phase=setup:build-vocab` while this runs — first-of-model
        // requests can sit here for tens of seconds.
        if schemaJSON != nil, cachedVocabTokens == nil {
            let cfgVocab = configVocabSize
            DecodeProgress.counter?.setSetupStage("build-vocab")
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
            if let built {
                cachedVocabTokens = (
                    built.0, built.1,
                    StructuredVocabulary.openerAliases(tokens: built.0))
            }
            DecodeProgress.counter?.setSetupStage(nil)
            Self.log.notice(
                """
                structured-vocab built elapsed=\
                \(String(format: "%.1f", Date().timeIntervalSince(vocabT0)))s \
                tokens=\(built?.0.count ?? 0)
                """,
                metadata: ["function": "runSpeculative"])
        }
        let vocabTokens = cachedVocabTokens
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
                model: residentName ?? defaultName,
                principal: principal, cacheKey: promptCacheKey)
        }
        // M63.3b — for an unguided greedy request, load the DFlash drafter
        // registered for the resident model (lazy HF pull on first use).
        // Non-nil only when DFlash is enabled AND a drafter exists for this
        // model AND the request is speculative-greedy + unstructured; the
        // dispatch branch below then routes a Gemma4 target through the
        // DFlash engine. nil ⇒ unchanged behavior.
        let dflashBox: DFlashDraftBox? =
            (dflashEnabled && schemaJSON == nil && greedyEligible)
            ? try await ensureDFlashDraft() : nil

        // The closure returns the decoded text, the completion token count
        // (`ids.count`), and the cached-prefix token count (M59.3); the
        // prompt count is `promptTokens.count` from the outer scope. nil ⇒
        // fall back to the substrate stream.
        let decoded = try await container.perform {
            (ctx: ModelContext) -> (text: String, completion: Int, cached: Int)? in

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
                if greedyEligible && model.hasMTPHead {
                    let r = SpeculativeGeneration.generate(
                        model: model, promptTokens: promptTokens,
                        maxTokens: maxTokens,
                        eosTokenId: ctx.tokenizer.eosTokenId, guide: guide,
                        prefixCache: prefixCache, cacheScope: cacheScope)
                    ids = r.ids
                    cachedTokens = r.cachedTokens
                } else if samplingEligible && model.hasMTPHead {
                    // M40.2 sampling-mode (internal). The Guide is nil
                    // here by construction — `samplingEligible` requires
                    // `schemaJSON == nil`.
                    ids = SpeculativeSampling.generate(
                        model: model, promptTokens: promptTokens,
                        maxTokens: maxTokens,
                        eosTokenId: ctx.tokenizer.eosTokenId,
                        temperature: samplingTemp,
                        topP: samplingTopP, seed: samplingSeed)
                } else if guide != nil {
                    // Structured but no speculative/MTP path available.
                    ids = GuidedGreedy.generate(
                        model: model, promptTokens: promptTokens,
                        maxTokens: maxTokens,
                        eosTokenId: ctx.tokenizer.eosTokenId, guide: guide)
                } else {
                    return nil  // unstructured + no MTP ⇒ substrate stream
                }
                return (
                    ctx.tokenizer.decode(tokenIds: ids), ids.count,
                    cachedTokens)
            }

            // M63.3b — DFlash lossless speculative decoding for an
            // attention-only target (Gemma4) with a loaded drafter. Engages
            // for the unguided greedy path only (`dflashBox` is nil
            // otherwise); structured output stays on the substrate path
            // until M63.4. Output is the target's block-forward greedy.
            if let dflashBox, let target = ctx.model as? DFlashGemma4Backbone {
                // Full stop-token set, matching the substrate generation path
                // (config eos_token_id + tokenizer EOS + extra EOS like
                // Gemma's <end_of_turn>) so DFlash stops where normal decode
                // would, not just on the tokenizer's single eosTokenId.
                var stopTokens = ctx.configuration.eosTokenIds
                if let e = ctx.tokenizer.eosTokenId { stopTokens.insert(e) }
                for tok in ctx.configuration.extraEOSTokens {
                    if let id = ctx.tokenizer.convertTokenToId(tok) {
                        stopTokens.insert(id)
                    }
                }
                let ids = DFlashGeneration.generate(
                    target: target, draft: dflashBox.model,
                    promptTokens: promptTokens, maxTokens: maxTokens,
                    stopTokens: stopTokens)
                return (ctx.tokenizer.decode(tokenIds: ids), ids.count, 0)
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
            return (ctx.tokenizer.decode(tokenIds: ids), ids.count, 0)
        }
        guard let decoded else { return nil }
        return (
            decoded.text,
            TokenUsage(
                promptTokens: promptTokens.count,
                completionTokens: decoded.completion,
                cachedTokens: decoded.cached))
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
