import AthenaCore
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
    /// Operator cap on the post-chat-template prompt length, in tokens
    /// (`max_prompt_tokens`). nil ⇒ unbounded (legacy behavior). Prefill
    /// attention is O(seq²); past a model/hardware-specific length a single
    /// score buffer exceeds Metal's `maxBufferLength` and the MLX eval aborts
    /// the daemon process-wide (observed: gemma4-MoE at ~61k tokens needing a
    /// 111 GiB buffer vs an 80.6 GiB device cap). This refuses an oversized
    /// prompt with a clean 400 (`input_too_long`) before prefill, instead of a
    /// crash. The right value is hardware+model specific — it's a calibration
    /// knob, not a derivable constant — so it is operator-set, default-off.
    public var maxPromptTokens: Int?
    /// ADR 032 — explicit MTP speculative-drafter store id (`mtp_drafter`)
    /// paired to the resident target; overrides the seeded default-drafter map.
    /// nil ⇒ the map (loaded from `dataDir`) resolves it, else no drafter.
    public var mtpDrafter: String?
    /// The daemon data dir, used only to find an operator override of the
    /// default-drafter map (`<dataDir>/mtp-drafters.toml`). nil ⇒ bundled seed.
    public var dataDir: URL?

    public init(
        maxTokens: Int = 1024, temperature: Float = 0.7,
        topP: Float = 0.95, speculative: Bool = false,
        kvCompression: KVCompression = .none,
        maxPromptTokens: Int? = nil,
        mtpDrafter: String? = nil,
        dataDir: URL? = nil
    ) {
        self.maxTokens = maxTokens
        self.temperature = temperature
        self.topP = topP
        self.speculative = speculative
        self.kvCompression = kvCompression
        self.maxPromptTokens = maxPromptTokens
        self.mtpDrafter = mtpDrafter
        self.dataDir = dataDir
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

    /// ADR 030 — the device's maximum single Metal buffer size, read once.
    /// Feeds the default prompt-length ceiling (`defaultPromptTokenCeiling`)
    /// when the operator hasn't set `max_prompt_tokens`. Device-constant, so a
    /// lazily-computed global is correct and cheap.
    /// ADR 042 — `public` so the `/v1/models` handler can publish the same
    /// derived ceiling the decode path enforces.
    public nonisolated static let deviceMaxBufferBytes: Int =
        MLX.GPU.deviceInfo().maxBufferSize

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
    /// ADR 032 — the resident target's paired MTP speculative drafter, loaded
    /// alongside a Gemma 4 target when `speculative` is on and a drafter pairs.
    /// Holds no target-derived state (substrate contract) so it is reused across
    /// requests; nil ⇒ MTP inert (single-token decode). Driven via the
    /// substrate `generate(… mtpDrafter:)` overload in `beginGeneration`.
    ///
    /// Boxed `@unchecked Sendable`: the drafter is a reference type with no
    /// mutable target-derived state and only ever executes under the serialized
    /// inference gate (ADR 029), so capturing it across the `container.perform`
    /// actor boundary is race-free (the substrate contract: "safe to share").
    private struct DrafterBox: @unchecked Sendable {
        let model: any MTPDrafterModel
    }
    private var mtpDrafterModel: DrafterBox?
    private var mtpDrafterName: String?
    /// M71.2 — true when the resident container was loaded via the substrate's
    /// `VLMModelFactory` (a vision checkpoint with an image tower). Drives the
    /// `servesVision` capability.
    private var residentIsVision = false
    /// M62 — the model the NEXT cold `load(reservation:)` will bind, set by
    /// `selectColdLoadModel` before the governor's non-blocking cold-load is
    /// kicked off. nil ⇒ bind the default. Only consulted while the slot is
    /// unloaded; a warm swap goes through `rebind`.
    private var desiredName: String?

    // ADR 039 S2 — continuous-batching queue state (fixed-batch minimal core).
    // Plain-text-chat requests enqueue here (behind `BatchScheduler.enabled`)
    // instead of each taking the ADR-029 gate; one detached worker drains the
    // queue, admits a batch (SequenceKVLedger), and drives `BatchGenerator`
    // under one gated span. See `BatchScheduler.swift`. All actor-isolated.
    var batchQueue: [BatchPending] = []
    var batchWorkerRunning = false
    var batchUIDCounter = 0
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

    /// Single-model convenience init kept for source-compat with M27/M40
    /// call sites (and tests): the directory IS a store entry, so the store
    /// root is its parent and the directory's basename is the configured
    /// default (ADR 026).
    public init(
        modelDirectory: URL,
        parameters: LLMGenerationParameters = .init(),
        promptCacheCapBytes: Int = 0
    ) {
        self.init(
            modelStoreRoot: modelDirectory.deletingLastPathComponent(),
            configuredDefault: modelDirectory.lastPathComponent,
            parameters: parameters,
            promptCacheCapBytes: promptCacheCapBytes)
    }

    /// ADR 026 — the LLM slot serves whatever LLM/vision models are in the
    /// store under `modelStoreRoot`; `configuredDefault` (the TOML `model` key)
    /// is loaded when a request omits `model` (nil ⇒ ambiguity rule).
    public init(
        modelStoreRoot: URL,
        configuredDefault: String? = nil,
        parameters: LLMGenerationParameters = .init(),
        promptCacheCapBytes: Int = 0
    ) {
        self.modelStoreRoot = modelStoreRoot
        let cleanDefault =
            (configuredDefault?.isEmpty == true)
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

    /// ADR 042 — the exact prompt-token count for this request shape, rendered
    /// through the SAME `container.prepare` (chat template + tokenizer + tool
    /// serialization) the decode path uses, so the number equals the
    /// `usage.prompt_tokens` the identical body will report.
    ///
    /// Reads `shape` only — never `asArray` — so no MLX evaluation happens: the
    /// substrate's text and Gemma4-VLM processors both build the token
    /// `MLXArray` host-side and image expansion is guarded on a non-empty image
    /// list (the route refuses image parts up front). Hence no ADR 029 gate:
    /// nothing executes. Measured ~34 ms on an idle engine.
    ///
    /// It is NOT concurrent with a decode: `container.prepare` reaches the
    /// processor through `SerialAccessContainer.read`, whose async mutex the
    /// generating request holds for its whole decode — so a count issued
    /// mid-generation waits it out (~8 s behind an 11 s decode, measured
    /// 2026-07-25). Recorded in the ADR 042 §4(b) amendment. Upgrade path if a
    /// consumer needs true concurrency: hold the processor/tokenizer outside
    /// the container, which is a substrate-side change, not a serve-path one.
    public func countPromptTokens(
        messages: [ChatTurn],
        tools: [[String: any Sendable]]? = nil,
        chatTemplateKwargs: [String: any Sendable]? = nil
    ) async throws -> Int {
        guard let container else {
            throw AthenaError.moduleNotRegistered(.llm)
        }
        let lmInput = try await container.prepare(
            input: UserInput(
                chat: Self.chatMessages(messages), tools: tools,
                additionalContext: chatTemplateKwargs))
        return lmInput.text.tokens.shape.last ?? 0
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
            // ADR 034 — carry tool history so the template pairs the
            // assistant's call with its tool result. Assistant `tool_calls` →
            // `.calls`; a tool-result turn's `tool_call_id` → `.result`.
            let tool: Chat.Message.Tool? = {
                if !turn.toolCalls.isEmpty {
                    return .calls(
                        turn.toolCalls.map { tc in
                            ToolCall(
                                function: .init(
                                    name: tc.name,
                                    arguments: Self.toolArgsObject(
                                        tc.argumentsJSON)),
                                id: tc.id)
                        })
                }
                if let id = turn.toolCallID { return .result(id: id) }
                return nil
            }()
            return Chat.Message(
                role: Chat.Message.Role(rawValue: turn.role) ?? .user,
                content: turn.content, images: images, tool: tool)
        }
        return mapped.isEmpty ? [.user("")] : mapped
    }

    /// ADR 034 — parse a tool call's stringified `arguments` back to the
    /// substrate's `[String: JSONValue]` for `ToolCall.Function`. Malformed /
    /// non-object ⇒ empty (the template still renders an argument-less call).
    /// `MLXLMCommon.JSONValue` is qualified to disambiguate from
    /// `AthenaStructured.JSONValue`.
    static func toolArgsObject(
        _ json: String
    ) -> [String: MLXLMCommon.JSONValue] {
        (try? JSONDecoder().decode(
            [String: MLXLMCommon.JSONValue].self,
            from: Data(json.utf8))) ?? [:]
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
        mtpDrafterModel = nil  // ADR 032 — drafter is paired to the target
        mtpDrafterName = nil
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
            let built = await container.perform {
                (ctx: ModelContext) -> ([VocabToken], UInt32)? in
                // Every architecture uses config.json's vocab_size, so guided
                // structured output is available everywhere (M23 fork A).
                guard let vocabSize = cfgVocab else { return nil }
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
                StructuredVocabulary.openerAliases(tokens: built.0)
            )
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
        // Publication S0 de-vendor: Qwen3.5 loads the substrate's own
        // `Qwen35Model`/`Qwen35TextModel`/`Qwen35MoEModel` (auto-registered in
        // `LLMTypeRegistry`), NOT the vendored `AthenaQwen35*`. The fused
        // in-model MTP head is retired — MTP now rides the substrate's
        // separate-drafter path (`loadMTPDrafterIfEligible` below), the same
        // mechanism Gemma 4 uses (ADR 032). TriAttention eviction moved to the
        // substrate `kvScheme` hook (see `beginGeneration`).
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
            let resolved = url.resolvingSymlinksInPath()
            let loader = #huggingFaceTokenizerLoader()
            if isVision {
                // Single VLM container serves BOTH text and image for a
                // vision checkpoint (ADR 010/011: one resident copy).
                loaded = try await VLMModelFactory.shared.loadContainer(
                    from: resolved, using: loader)
            } else {
                loaded = try await loadModelContainer(
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
        // ADR 032 — pair an MTP speculative drafter for a Gemma 4 target when
        // speculative is enabled. Best-effort: a missing/failed drafter logs and
        // leaves MTP inert, never failing the target load.
        await loadMTPDrafterIfEligible(
            targetName: name, modelType: info?.modelType)
    }

    /// ADR 032 — load the Gemma 4 MTP drafter paired to the just-loaded target,
    /// when `speculative` is on. Resolution: explicit `mtp_drafter` > seeded
    /// default-drafter map > none. The drafter must already be in the store
    /// (`athena pull <target> --with-drafter`); a miss leaves MTP inert with a
    /// pointer, it does not download mid-load (ADR 015 cold-load posture) or
    /// fail the target. Loaded via the substrate `MTPDrafterModelFactory` after
    /// a one-time idempotent type registration.
    private func loadMTPDrafterIfEligible(
        targetName: String, modelType: String?
    ) async {
        mtpDrafterModel = nil
        mtpDrafterName = nil
        guard params.speculative else { return }
        guard let mt = modelType?.lowercased() else { return }
        // Qwen3.5 MTP (publication S0): the fused `mtp.*` head ships inside the
        // TARGET checkpoint, so the substrate drafter loads from the same
        // directory — no separate drafter repo / pairing map. Inert when the
        // checkpoint carries no `mtp.*` weights.
        if mt.hasPrefix("qwen3_5") {
            await loadQwen35MTPDrafter(targetName: targetName)
            return
        }
        // The substrate drafter (`Gemma4AssistantDraftModel`) only drafts for a
        // Gemma 4 target; gate on the family so a stray map entry can't mis-pair.
        guard mt.hasPrefix("gemma4") else { return }
        guard
            let drafterId = MTPDrafterPairing.resolve(
                targetID: targetName, explicit: params.mtpDrafter,
                defaults: MTPDrafterPairing.defaultMap(dataDir: params.dataDir))
        else { return }
        guard let dir = directoryURL(for: drafterId),
            FileManager.default.fileExists(
                atPath: dir.appendingPathComponent("config.json").path)
        else {
            Self.log.warning(
                """
                MTP drafter '\(drafterId)' for target '\(targetName)' is not in \
                the store — speculative decoding stays inert. Pull it with \
                `athena pull \(targetName) --with-drafter`.
                """,
                metadata: ["function": "loadMTPDrafterIfEligible"])
            return
        }
        do {
            // Register the `gemma4_assistant` drafter type once (idempotent); the
            // creator lives in MLXVLM and self-registers on import, but the
            // registry actor still needs this awaited bootstrap.
            await Gemma4AssistantRegistration.register()
            let loader = #huggingFaceTokenizerLoader()
            let ctx = try await MTPDrafterModelFactory.shared.load(
                from: dir.resolvingSymlinksInPath(), using: loader)
            mtpDrafterModel = DrafterBox(model: ctx.model)
            mtpDrafterName = drafterId
            Self.log.notice(
                "MTP drafter loaded id=\(drafterId) target=\(targetName)",
                metadata: ["function": "loadMTPDrafterIfEligible"])
        } catch {
            mtpDrafterModel = nil
            mtpDrafterName = nil
            Self.log.warning(
                """
                MTP drafter load failed for '\(drafterId)' (\(error)) — \
                speculative decoding stays inert.
                """,
                metadata: ["function": "loadMTPDrafterIfEligible"])
        }
    }

    /// Publication S0 — load the Qwen3.5 MTP drafter from the target's OWN
    /// checkpoint (the fused `mtp.*` head lives in the same directory; the
    /// substrate `Qwen35MTPDraftModel.sanitize` picks it out and shares the
    /// target's embeddings + head). Replaces the retired vendored fused-head
    /// decode loop with the substrate's separate-drafter path. Best-effort:
    /// a checkpoint with no `mtp.*` weights, or a load failure, leaves MTP
    /// inert (single-token) without failing the target load.
    private func loadQwen35MTPDrafter(targetName: String) async {
        guard let dir = directoryURL(for: targetName) else { return }
        // No `mtp.*` in the checkpoint ⇒ nothing to draft with; stay inert.
        guard MTPCheckpoint.checkpointHasMTP(dir) else { return }
        do {
            await Qwen35TextMTPRegistration.register()
            let loader = #huggingFaceTokenizerLoader()
            let ctx = try await MTPDrafterModelFactory.shared.load(
                from: dir.resolvingSymlinksInPath(), using: loader)
            mtpDrafterModel = DrafterBox(model: ctx.model)
            mtpDrafterName = targetName
            Self.log.notice(
                "MTP drafter (Qwen3.5 fused) loaded target=\(targetName)",
                metadata: ["function": "loadQwen35MTPDrafter"])
        } catch {
            mtpDrafterModel = nil
            mtpDrafterName = nil
            Self.log.warning(
                """
                Qwen3.5 MTP drafter load failed for '\(targetName)' (\(error)) \
                — speculative decoding stays inert.
                """,
                metadata: ["function": "loadQwen35MTPDrafter"])
        }
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

    // The String `generate(prompt:)` overload comes from the LLMModule
    // extension default (filters `generateMetered` → `.text`); the real
    // engine below is `generateMetered`. (Test-only convenience overloads
    // removed — the audit tail.)
    public nonisolated func generateMetered(
        messages: [ChatTurn], schemaJSON: String?,
        tools: [[String: any Sendable]]?,
        maxTokens: Int?, temperature: Double?,
        topP: Double?, seed: Int?,
        speculative: Bool?,
        chatTemplateKwargs: [String: any Sendable]?,
        promptCacheKey: String? = nil,
        principal: String? = nil,
        logprobs: LogprobsRequest? = nil,
        requestedModel: String? = nil
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
                // ADR 039 S2 — batched fast path: a plain-text-chat request on
                // the resident model enqueues into the shared batch worker
                // (which owns the gate for the batch) instead of taking the gate
                // itself. Returns true when it took ownership of `continuation`
                // (the worker drives usage/finish/finish() from here); false ⇒
                // not batchable, fall through to the unchanged per-request path.
                if await self.tryEnqueueBatched(
                    messages: messages, schemaJSON: schemaJSON, tools: tools,
                    maxTokens: maxTokens, temperature: temperature, topP: topP,
                    seed: seed, speculative: speculative,
                    chatTemplateKwargs: chatTemplateKwargs, logprobs: logprobs,
                    requestedModel: requestedModel, continuation: continuation)
                {
                    return
                }
                do {
                    // ADR 029 — hold the process-global inference execution gate
                    // for the WHOLE decode so no other tenant (or a warm rebind)
                    // drives MLX kernels on the one Metal pool concurrently. The
                    // gate wraps token production, not the lazy stream
                    // consumption: `AsyncStream`'s default unbounded buffering
                    // means this Task is decode-bound, so a slow SSE consumer
                    // never holds the gate. `usage` is returned OUT of the
                    // @Sendable closure (not captured as a mutable var).
                    let usage = try await InferenceGate.shared
                        .withExclusiveExecution { () async throws -> TokenUsage in
                            // WP6 (ADR 029 residual) — bind the requested model
                            // INSIDE the gate so a concurrent different-model
                            // rebind can't swap the shared slot between the
                            // server's preflight rebind and this decode (the H3
                            // wrong-model window). No-op when already bound (the
                            // common single-model case); corrects a drift only
                            // under actual cross-model contention. Mirrors the
                            // embedding module's `embedInFlight` atomic rebind.
                            if let m = requestedModel, !m.isEmpty {
                                try await self.rebind(to: m)
                            }
                            var u = TokenUsage.zero
                            if let speculative = try await self.runSpeculative(
                                messages: messages, schemaJSON: schemaJSON,
                                tools: tools, maxTokens: maxTokens,
                                requestSpeculative: speculative,
                                requestTemperature: temperature,
                                chatTemplateKwargs: chatTemplateKwargs,
                                promptCacheKey: promptCacheKey,
                                principal: principal, logprobs: logprobs)
                            {
                                u = speculative.usage
                                continuation.yield(.text(speculative.text))
                                continuation.yield(.usage(u))
                                // C2 (ADR 013 §4): per-token logprobs.
                                if let lps = speculative.logprobs {
                                    continuation.yield(.logprobs(lps))
                                }
                            } else {
                                // runSpeculative returns nil only for
                                // UNstructured requests (no schema) — those
                                // stream from the standard substrate path.
                                let stream = try await self.beginGeneration(
                                    messages: messages, tools: tools,
                                    maxTokens: maxTokens,
                                    temperature: temperature,
                                    topP: topP, seed: seed,
                                    chatTemplateKwargs: chatTemplateKwargs,
                                    requestSpeculative: speculative)
                                for await event in stream {
                                    switch event {
                                    case .chunk(let text):
                                        continuation.yield(.text(text))
                                    case .toolCall(let tc):
                                        // ADR 034 — a freely-chosen tool call
                                        // (tool_choice:auto), detected by the
                                        // substrate's native .gemma4 handler.
                                        // Serialize args with the SAME
                                        // sorted-key form as the Guide-forced
                                        // parse so both surfaces agree.
                                        continuation.yield(
                                            .toolCall(
                                                name: tc.function.name,
                                                argumentsJSON: toolArgumentsJSON(
                                                    tc.function.arguments
                                                        .mapValues { $0.anyValue }
                                                )))
                                    case .info(let info):
                                        // Substrate's terminal completion
                                        // record carries the token geometry.
                                        u = TokenUsage(
                                            promptTokens: info.promptTokenCount,
                                            completionTokens:
                                                info.generationTokenCount)
                                        // ADR 032 S4 — surface MTP acceptance
                                        // for parity with the Qwen path's rate
                                        // log. Present only when the drafter ran
                                        // (the mtpDrafter overload populates it);
                                        // a non-nil passthroughReason means MTP
                                        // silently fell back to single-token.
                                        Self.logMTPStats(info)
                                    @unknown default:
                                        break
                                    }
                                }
                                continuation.yield(.usage(u))
                            }
                            return u
                        }
                    // M31.2: the effective cap is the same positive-wins
                    // resolution both paths apply; hitting it ⇒ truncated.
                    let cap = Self.effectiveMaxTokens(
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

    /// The masking / logit-capture decode path. Returns the full decoded text,
    /// or nil to defer to the unconstrained substrate stream in
    /// `beginGeneration`.
    ///
    /// Serves exactly the two cases the substrate stream cannot: a structured
    /// request (`schemaJSON`), decoded under the Guide's mask, and a logprobs
    /// request, which needs a logit-capture seam `beginGeneration` does not
    /// have. See `DecodeDispatch` for the routing.
    ///
    /// The name is historical. This was the MTP speculative path, hosting an
    /// in-closure greedy loop and an M40.2/M40.3 sampling-mode loop; publication
    /// S0 removed both along with the Qwen3.5 vendored fork, and speculative now
    /// means the ADR 032 separate-drafter overload inside `beginGeneration` —
    /// i.e. on the path this function DEFERS to, not the one it runs.
    private func runSpeculative(
        messages: [ChatTurn], schemaJSON: String?,
        tools: [[String: any Sendable]]?, maxTokens: Int?,
        requestSpeculative: Bool?,
        requestTemperature: Double?,
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
        // Per-request override (a downstream client's intent): if the caller
        // explicitly passes `speculative=true/false`, that wins for THIS
        // request; if nil, fall back to the daemon's loaded default. Both
        // values below exist ONLY to be logged — nothing routes on them.
        let effectiveSpec = requestSpeculative ?? params.speculative
        // The temperature the request will actually decode at. `speculative`
        // used to coerce this to 0 ("speculative implies greedy", M40.2), which
        // was true while an in-closure greedy loop served those requests.
        // Publication S0 removed that loop, so the coercion reached nothing but
        // this log line — where it lied, reporting `temp=0.0` for a request
        // `beginGeneration` decodes at `params.temperature`. Report what the
        // decode uses, matching `beginGeneration`'s own resolution.
        let reportedTemp: Double =
            requestTemperature.flatMap { $0 >= 0 ? $0 : nil }
            ?? Double(params.temperature)
        // M48.3 — temperature is INERT under a Guide: the schema mask collapses
        // every position's distribution to its allowed set, so a structured
        // request decodes the same masked-argmax sequence whatever temperature
        // the caller passed. That is why the routing below can ignore
        // temperature outright; both values survive only to be logged.
        // M48.2 — declare which internal generate path this request will take,
        // BEFORE any model work begins, so an operator can see at a glance
        // which decode served a request.
        //
        // Routed on what the request needs (schema ⇒ mask, logprobs ⇒ capture),
        // not on `speculative`/`temperature`. Those used to gate
        // `greedyEligible`/`samplingEligible`, which named in-closure decode
        // branches publication S0 deleted — so a speculative unstructured
        // request was admitted past `container.prepare` below and then turned
        // away by `guide == nil && logprobSink == nil`, having paid a full
        // chat-template render + tokenize (and the substrate's
        // `SerialAccessContainer` mutex) for nothing, while this line claimed a
        // `speculative-greedy`/`speculative-sampling` path that no longer
        // exists. Speculative is unaffected: its MTP drafter overload lives in
        // `beginGeneration`, which is exactly where the deferral sends it.
        let dispatch = DecodeDispatch.route(
            hasSchema: schemaJSON != nil, hasLogprobSink: logprobSink != nil)
        Self.log.debug(
            """
            dispatch path=\(dispatch.rawValue) spec=\(effectiveSpec) \
            temp=\(reportedTemp) schema=\(schemaJSON != nil)
            """,
            metadata: ["function": "runSpeculative"])
        // C2: a logprobs request must NOT defer — `beginGeneration` has no
        // logit-capture seam — so it stays here even with no schema.
        //
        // ADR 030: the prompt ceiling is NOT skipped by deferring.
        // `beginGeneration` enforces it at its own `container.prepare`
        // chokepoint (verified, not assumed), so both decode routes stay
        // covered.
        if dispatch.defersToSubstrateStream { return nil }

        let lmInput = try await container.prepare(
            input: UserInput(
                chat: Self.chatMessages(messages), tools: tools,
                additionalContext: chatTemplateKwargs))
        let promptTokens = lmInput.text.tokens.asArray(Int.self)
        // ADR 030 — refuse an over-ceiling prompt with a 400 before prefill so
        // the O(seq²) score buffer can't exceed the device's max Metal buffer
        // and abort the daemon. Shared with the substrate-stream path
        // (`beginGeneration`) so BOTH decode routes are covered.
        try enforcePromptCeiling(tokenCount: promptTokens.count)
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
        // Publication S0 — the M40.2/M40.3 in-closure sampling-mode branch went
        // away with the vendored Qwen3.5 decode fork, and with it the locally
        // resolved temp/top_p/seed it consumed. An unstructured sampling request
        // now returns nil below and lands on the substrate stream, which resolves
        // the same knobs onto `GenerateParameters` in `beginGeneration`.
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
            // a downstream client's enable_thinking=False production config).
            // Schema-enforced output that ALSO permits a thinking prefix
            // (deferred enforcement, Patch 6) is tracked in
            // an internal issue.
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

            // Publication S0 — Qwen3.5 no longer has a vendored decode fork.
            // It routes like every other architecture: structured/logprobs via
            // the substrate GuidedSubstrate path below; unstructured returns nil
            // → the substrate stream in `beginGeneration` (which drives the MTP
            // separate-drafter overload when a drafter is resident).

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
            let substrateEos =
                vocabTokens.map { Int($0.eos) }
                ?? ctx.tokenizer.eosTokenId
            let substrateVocab = cfgVocab ?? 0
            if guide != nil && cfgVocab == nil { return nil }  // can't mask
            let ids = try GuidedSubstrate.generate(
                model: ctx.model, promptTokens: promptTokens,
                vocab: substrateVocab, maxTokens: maxTokens,
                eosTokenId: substrateEos, guide: guide, sink: logprobSink)
            return (
                ctx.tokenizer.decode(tokenIds: ids), ids.count, 0,
                builtLogprobs()
            )
        }
        guard let decoded else { return nil }
        return (
            decoded.text,
            TokenUsage(
                promptTokens: promptTokens.count,
                completionTokens: decoded.completion,
                cachedTokens: decoded.cached),
            decoded.logprobs
        )
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
        chatTemplateKwargs: [String: any Sendable]?,
        requestSpeculative: Bool? = nil
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
            kvScheme: params.kvCompression.kvScheme,
            temperature: temp,
            topP: tp,
            seed: requestSeed)
        // Publication S0 — TriAttention eviction now rides the substrate's
        // `kvScheme` hook on `gp` (upstream `applyKVScheme` swaps each
        // `KVCacheSimple` for a `TriAttentionKVCache` when the cache is built
        // inside `generate`); no task-local, no vendored model. The substrate
        // deliberately skips eviction on the MTP/speculative path, matching the
        // old inert-on-MTP behavior.
        do {
            let userInput = UserInput(
                chat: Self.chatMessages(messages), tools: tools,
                additionalContext: chatTemplateKwargs)
            let lmInput = try await container.prepare(input: userInput)
            // ADR 030 — same prefill ceiling as runSpeculative; this is the
            // substrate-stream path the gemma4-MoE long-context abort took.
            try enforcePromptCeiling(tokenCount: lmInput.text.tokens.size)
            // ADR 032 — Gemma 4 MTP: when speculative resolves on AND a paired
            // drafter is resident, drive the substrate's MTP overload (target +
            // drafter, lossless via target-verify) instead of the plain stream.
            // Unstructured-only by construction (this is the no-schema path) and
            // already inside the ADR 029 inference gate. `gp` carries the
            // effective temp/top_p/seed → greedy at temp 0, lossless sampling at
            // temp>0. blockSize 4 matches the mlx-vlm default.
            // ponytail: blockSize is the substrate default; expose `mtp_block_size`
            // only if a measured speedup justifies a tuning knob.
            if (requestSpeculative ?? params.speculative),
                let drafterBox = mtpDrafterModel
            {
                return try await container.perform(nonSendable: lmInput) {
                    ctx, lmInput in
                    try MLXLMCommon.generate(
                        input: lmInput, parameters: gp, context: ctx,
                        mtpDrafter: drafterBox.model, blockSize: 4)
                }
            }
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

    /// ADR 032 S4 — MTP draft acceptance rate from the substrate completion
    /// info, or nil when the drafter did not run (no proposed tokens ⇒ the plain
    /// non-speculative path). Pure decision logic, unit-pinned (ADR 008/009).
    static func mtpAcceptanceRate(proposed: Int?, accepted: Int?) -> Double? {
        guard let proposed, proposed > 0 else { return nil }
        return Double(accepted ?? 0) / Double(proposed)
    }

    /// Log MTP acceptance for parity with the Qwen path's rate log. No-op unless
    /// the drafter ran. A non-nil `passthrough` means MTP degraded to
    /// single-token mid-stream (correctness kept, speedup dropped) — an operator
    /// signal. ponytail: the substrate exposes AGGREGATE counts via `.info`,
    /// while `SpeculativeStats`'s observer is per-iteration, so they don't bridge
    /// cleanly; the log line is the operability surface (add an observer feed
    /// only if a metrics consumer needs the structured stream).
    private static func logMTPStats(_ info: GenerateCompletionInfo) {
        guard
            let rate = mtpAcceptanceRate(
                proposed: info.proposedDraftTokens,
                accepted: info.acceptedDraftTokens)
        else { return }
        log.notice(
            """
            MTP speculative proposed=\(info.proposedDraftTokens ?? 0) \
            accepted=\(info.acceptedDraftTokens ?? 0) \
            accept_rate=\(String(format: "%.3f", rate)) \
            passthrough=\(info.passthroughReason ?? "none")
            """,
            metadata: ["function": "generate"])
    }

    /// True iff a `max_prompt_tokens` cap is set (positive) and the prompt
    /// exceeds it. nil/non-positive cap ⇒ unbounded (never exceeds). The
    /// MLX-free decision seam for the prefill prompt-length guard (ADR 009).
    static func promptExceedsCap(_ promptCount: Int, cap: Int?) -> Bool {
        guard let cap, cap > 0 else { return false }
        return promptCount > cap
    }

    /// ADR 030 — enforce the prefill prompt ceiling for THIS request before any
    /// MLX eval: the operator's `max_prompt_tokens` if set, else the
    /// device-derived default (`defaultPromptTokenCeiling`). An over-ceiling
    /// prompt throws `inputTooLong` (400) instead of letting the O(seq²) score
    /// buffer abort the daemon. An explicit operator `0` (≤0) stays the
    /// unbounded opt-out (calibration knob). Called at every `container.prepare`
    /// chokepoint (runSpeculative + beginGeneration).
    private func enforcePromptCeiling(tokenCount: Int) throws {
        // ADR 042 — the same resolution `GET /v1/models` publishes as
        // `max_prompt_tokens`, so the advertised ceiling is the enforced one.
        // nil ⇒ the operator's explicit unbounded opt-out.
        guard
            let cap = GovernorMemory.effectivePromptTokenCeiling(
                configured: params.maxPromptTokens,
                maxBufferBytes: Self.deviceMaxBufferBytes)
        else { return }
        if Self.promptExceedsCap(tokenCount, cap: cap) {
            throw AthenaError.inputTooLong(
                module: .llm, tokens: tokenCount, maxTokens: cap)
        }
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

// MARK: - ADR 039 S2 — batch worker (fixed-batch, default-off). Same-file so it
// reaches the actor's private resident-model state; types + knob live in
// `BatchScheduler.swift`.
extension MLXLLMModule {
    /// Try to enqueue a plain-text-chat request for batching. Returns `true` when
    /// it took ownership of `continuation` (enqueued, or failed it with an error
    /// chunk); `false` ⇒ not batchable, caller runs the unchanged serial path.
    func tryEnqueueBatched(
        messages: [ChatTurn], schemaJSON: String?,
        tools: [[String: any Sendable]]?, maxTokens: Int?,
        temperature: Double?, topP: Double?, seed: Int?, speculative: Bool?,
        chatTemplateKwargs: [String: any Sendable]?,
        logprobs: LogprobsRequest?, requestedModel: String?,
        continuation: AsyncStream<GenChunk>.Continuation
    ) async -> Bool {
        // Batchable iff: flag on, no structured output / tools / logprobs / MTP,
        // not a vision model, a model is resident, and the request targets the
        // resident model (a different-model request rebinds via the serial path).
        guard BatchScheduler.enabled,
            schemaJSON == nil,
            tools?.isEmpty ?? true,
            logprobs == nil,
            speculative != true,
            // Batch text-only chat on ANY model, including a VLM's text path
            // (verified end-to-end on gemma-4-26b-a4b). An IMAGE-bearing request
            // must not batch — the batch engine decodes text tokens only and has
            // no image-embedding path, so those take the serial VLM route.
            !messages.contains(where: { !$0.images.isEmpty }),
            let container = self.container,
            (requestedModel?.isEmpty ?? true) || requestedModel == residentName
        else { return false }

        do {
            // Same tokenization/chat-template as the serial path (parity); only
            // the token ids are extracted, the batch engine re-embeds per row.
            let userInput = UserInput(
                chat: Self.chatMessages(messages), tools: nil,
                additionalContext: chatTemplateKwargs)
            let lmInput = try await container.prepare(input: userInput)
            let promptTokens = lmInput.text.tokens.asArray(Int.self)
            try enforcePromptCeiling(tokenCount: promptTokens.count)

            let cap = Self.effectiveMaxTokens(maxTokens, params.maxTokens)
            let temp =
                (temperature.map { Float($0) }).flatMap { $0 >= 0 ? $0 : nil }
                ?? params.temperature
            let tp =
                topP.map { Float($0) }.flatMap { $0 > 0 && $0 < 1 ? $0 : nil }
                ?? params.topP
            let sd: UInt64? = seed.flatMap { $0 >= 0 ? UInt64($0) : nil }
            let sampler = makeRowSampler(temperature: temp, topP: tp, seed: sd)
            let kvBytes = GovernorMemory.sequenceKVReservation(
                maxTokens: cap, perTokenKVBytes: perTokenKVBytes)

            let uid = batchUIDCounter
            batchUIDCounter += 1
            batchQueue.append(
                BatchPending(
                    uid: uid, promptTokens: promptTokens,
                    promptCount: promptTokens.count, maxTokens: cap,
                    sampler: sampler, kvBytes: kvBytes,
                    continuation: continuation))
            startBatchWorkerIfIdle()
            return true
        } catch {
            let classified =
                (error as? AthenaError) ?? AthenaError.classify(error, module: .llm)
            continuation.yield(.error(classified))
            continuation.finish()
            return true
        }
    }

    private func startBatchWorkerIfIdle() {
        guard !batchWorkerRunning else { return }
        batchWorkerRunning = true
        Task.detached { await self.runBatchWorker() }
    }

    /// Detached worker: drain-admit-drive until the queue empties. Its heavy work
    /// is inside `container.perform` (a different actor), so this module actor is
    /// free to accept new enqueues between batches.
    private func runBatchWorker() async {
        while true {
            let batch = await takeBatch()
            if batch.isEmpty {
                batchWorkerRunning = false
                if !batchQueue.isEmpty { startBatchWorkerIfIdle() }
                return
            }
            await runOneBatch(batch)
        }
    }

    /// Admit as many queued rows as fit the ADR-023 budget (worst-case KV). A row
    /// that can't fit even alone (over budget) is failed, not requeued; a row that
    /// only fails because the batch is full is requeued for the next batch.
    private func takeBatch() async -> [BatchPending] {
        guard !batchQueue.isEmpty else { return [] }
        let (denominator, budget) = await BatchScheduler.admissionInputs()
        var admitted: [BatchPending] = []
        var remaining: [BatchPending] = []
        var ledger = SequenceKVLedger()
        for p in batchQueue {
            if ledger.admit(
                uid: p.uid, rowKVBytes: p.kvBytes,
                denominator: denominator, budget: budget)
            {
                admitted.append(p)
            } else if admitted.isEmpty {
                p.continuation.yield(
                    .error(
                        AthenaError.memoryBudgetExceeded(
                            requested: p.kvBytes, available: budget - denominator,
                            module: .llm)))
                p.continuation.finish()
            } else {
                remaining.append(p)
            }
        }
        batchQueue = remaining
        return admitted
    }

    /// Drive one fixed batch to completion under one gated span, fanning each
    /// row's tokens into its own stream via a per-row streaming detokenizer.
    private func runOneBatch(_ batch: [BatchPending]) async {
        guard let container = self.container else {
            for p in batch {
                p.continuation.yield(
                    .error(
                        AthenaError.moduleLoadFailed(
                            .llm, reason: "batch drive before load")))
                p.continuation.finish()
            }
            return
        }
        do {
            try await InferenceGate.shared.withExclusiveExecution {
                await container.perform { ctx in
                    let gen = BatchGenerator(model: ctx.model, defaultMaxTokens: 1)
                    let uids = gen.insert(
                        prompts: batch.map(\.promptTokens),
                        maxTokens: batch.map(\.maxTokens),
                        samplers: batch.map { Optional($0.sampler) })
                    var detoks: [Int: NaiveStreamingDetokenizer] = [:]
                    var conts: [Int: AsyncStream<GenChunk>.Continuation] = [:]
                    var prompt: [Int: Int] = [:]
                    var produced: [Int: Int] = [:]
                    for (i, uid) in uids.enumerated() {
                        detoks[uid] = NaiveStreamingDetokenizer(
                            tokenizer: ctx.tokenizer)
                        conts[uid] = batch[i].continuation
                        prompt[uid] = batch[i].promptCount
                        produced[uid] = 0
                    }
                    while gen.hasWork {
                        for r in gen.next() {
                            guard let cont = conts[r.uid] else { continue }
                            produced[r.uid, default: 0] += 1
                            if var d = detoks[r.uid] {
                                d.append(token: r.token)
                                let piece = d.next()
                                detoks[r.uid] = d
                                if let piece { cont.yield(.text(piece)) }
                            }
                            if let reason = r.finishReason {
                                let completion =
                                    r.allTokens?.count ?? produced[r.uid] ?? 0
                                cont.yield(
                                    .usage(
                                        TokenUsage(
                                            promptTokens: prompt[r.uid] ?? 0,
                                            completionTokens: completion)))
                                cont.yield(
                                    .finish(reason == "length" ? .length : .stop))
                                cont.finish()
                                conts[r.uid] = nil
                            }
                        }
                    }
                    for (_, cont) in conts { cont.finish() }
                }
            }
        } catch {
            let classified =
                (error as? AthenaError) ?? AthenaError.classify(error, module: .llm)
            for p in batch {
                p.continuation.yield(.error(classified))
                p.continuation.finish()
            }
        }
    }
}
