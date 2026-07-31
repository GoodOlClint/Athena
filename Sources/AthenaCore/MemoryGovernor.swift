import Foundation

/// Point-in-time view of the global budget, exposed by the health endpoint
/// and the `athena ps` CLI.
///
/// M46.5 renamed the bytes-resident counter on both `ModuleSnapshot`
/// and `GovernorSnapshot` from its pre-M46.7 admission-bookkeeping name
/// (the value once reflected reservations issued at admission, not
/// observed residency). After M46.7's serial preload fix the reconciled
/// value tracks real process RSS within the reconcile's RSS-delta probe
/// accuracy, so `residentBytes` is the honest name for what the field
/// actually reports. No compat shim — no in-tree consumer reads the
/// old field.
///
/// `unloadedReason` (M46.5) records WHY a module's slot is in
/// `.unloaded` state — a fresh boot vs. an idle eviction vs.
/// memory-pressure relief vs. an explicit operator unload — so an
/// operator looking at `/healthz` after the fact can tell whether
/// the governor moved a module or someone did. nil while loaded
/// (or in the initial post-boot pre-load window).
public struct ModuleSnapshot: Sendable, Codable {
    public let id: ModuleID
    public let state: ModuleState
    public let residentBytes: Int
    public let evictable: Bool
    public let unloadedReason: UnloadedReason?
    /// ADR 023 G3 — whether `residentBytes` is a **measured** footprint (the
    /// load-time probe delta reconciled in via M5.1/M5.4, `learnedFootprint`)
    /// rather than the pre-load **estimate**. `false` before the first reconcile
    /// or when a concurrent-teardown deflated probe made the reconcile fall back
    /// to the estimate (E12). Lets `athena ps`/healthz say whether a per-tenant
    /// number is real or a guess. (Honesty boundary, ADR 023: this is the
    /// *active* per-tenant measurement; the reclaimable MLX cache is a single
    /// global number, never per-tenant attributed.)
    public let measured: Bool

    public init(
        id: ModuleID, state: ModuleState, residentBytes: Int,
        evictable: Bool, unloadedReason: UnloadedReason?,
        measured: Bool = false
    ) {
        self.id = id
        self.state = state
        self.residentBytes = residentBytes
        self.evictable = evictable
        self.unloadedReason = unloadedReason
        self.measured = measured
    }
}

public struct GovernorSnapshot: Sendable, Codable {
    public let totalBudgetBytes: Int
    public let residentBytes: Int
    public let freeBytes: Int
    public let promptCacheCapBytes: Int
    public let modules: [ModuleSnapshot]
    /// ADR 023 G2 — the active admission accounting mode. `"footprint"` ⇒
    /// `freeBytes` is `budget − max(committed, reserved)` (the live Metal
    /// footprint governs admission); `"estimate"` ⇒ the pre-G2 reservation-only
    /// denominator (the revert switch). Surfaced so an operator can confirm from
    /// `/healthz` whether the truthful-accounting path is active.
    public let admissionMode: String

    public init(
        totalBudgetBytes: Int, residentBytes: Int, freeBytes: Int,
        promptCacheCapBytes: Int, modules: [ModuleSnapshot],
        admissionMode: String = GovernorMemory.AdmissionMode.footprint.rawValue
    ) {
        self.totalBudgetBytes = totalBudgetBytes
        self.residentBytes = residentBytes
        self.freeBytes = freeBytes
        self.promptCacheCapBytes = promptCacheCapBytes
        self.modules = modules
        self.admissionMode = admissionMode
    }
}

/// M46.5 — last-known reason a module's slot is `.unloaded`. nil ⇒
/// module is currently loaded (or has never been touched). The serve
/// path emits an audit-log row on every transition so an operator who
/// needs the timing as well as the cause can cross-reference.
public enum UnloadedReason: String, Sendable, Codable {
    /// Pressure-relief eviction by the governor — another module's
    /// load (or reconcile) pushed the budget over and this slot was
    /// the LRU victim.
    case memoryPressure = "memory_pressure"
    /// Explicit unload via `/api/models/unload` or `athena models
    /// unload` — operator-initiated.
    case operatorUnload = "operator_unload"
    /// Load attempted but the module threw (HF download failed,
    /// disk write failed, etc.). Distinct from a clean unload so
    /// the operator knows the module wasn't deliberately removed.
    case loadFailed = "load_failed"
}

/// The single Metal/MLX memory governor. Every inference module shares one
/// global budget; the governor performs admission control, on-demand load,
/// and LRU eviction of evictable modules so modules cooperate instead of
/// racing the box into a process-fatal Metal OOM.
///
/// The governor is the single source of truth for accounting: it records the
/// reservation it issued, so admission/eviction never depends on polling a
/// module's isolated internal state.
public actor MemoryGovernor {
    private struct Entry {
        let module: any InferenceModule
        var state: ModuleState
        var reservation: MemoryReservation?
        let evictable: Bool
        var lastUsed: Date
        /// M46.5 — last-known reason this slot transitioned to
        /// `.unloaded`. Set at each unload site; cleared back to nil
        /// when a successful load returns. Surfaced in
        /// `/healthz.modules[*].unloadedReason` so an operator can
        /// tell why a slot is empty (idle eviction vs. operator
        /// unload vs. load failure) without grepping the audit log.
        var unloadedReason: UnloadedReason? = nil
    }

    /// Reads process-global Metal/MLX active bytes. Injected so
    /// `AthenaCore` stays substrate-agnostic — the `athena` target
    /// backs it with `MLX.Memory`. nil ⇒ estimate-only (pre-M5).
    public typealias MemoryProbe = @Sendable () -> Int
    /// ADR 023 G2 — reads the live process footprint + the reclaimable MLX
    /// buffer cache so admission can meter `committed = phys_footprint − cache`
    /// (the genuinely-pinned memory) instead of trusting reservation estimates
    /// alone. Injected (like `memoryProbe`) so AthenaCore stays
    /// substrate-agnostic; the serve path backs it with
    /// `ProcessMemory.sample().physFootprint` + `MLX.Memory.cacheMemory`.
    /// `nil` (or a Mach-failure `(0, …)`) ⇒ admission degrades to the
    /// reservation-only denominator — byte-identical to the pre-G2 path and the
    /// path taken under `swift test` (no MLX). Distinct from `memoryProbe`
    /// (RSS, drives the relief high-water + reconcile), which is unchanged.
    public typealias FootprintProbe = @Sendable () -> (physFootprint: Int, cacheBytes: Int)?
    /// Called after a module finishes `unload()`, so the substrate's
    /// own buffer pool can be trimmed (the freed bytes otherwise stay
    /// cached). Injected — keeps AthenaCore substrate-agnostic.
    public typealias UnloadHook = @Sendable () -> Void
    /// ADR 023 G2 — reclaim the reclaimable MLX buffer cache (`clearCache`) as
    /// the first rung of the admission ladder, BEFORE evicting a tenant: the
    /// cache is headroom, not pinned memory, so reclaiming it may let the load
    /// fit without disrupting a resident module. Injected. `nil` ⇒ skip the
    /// reclaim rung (the reservation-only path).
    /// ADR 029 WP1 — `async` because the serve-side hook runs its MLX free
    /// (`clearCache`) inside the process-global `InferenceGate`, so it can never
    /// tear down the buffer pool while a decode holds the device. The governor
    /// `await`s it, so the reclaim's effect is still observed synchronously by
    /// the re-gate that follows — it just waits for any in-flight forward first.
    public typealias ReclaimCacheHook = @Sendable () async -> Void
    /// Model-lifecycle observer (loading/loaded/evicted/unloaded/load
    /// failed), keyed by module. Injected so AthenaCore needs no
    /// logging dependency; the `athena` target maps it to a per-module
    /// unified-log category.
    public typealias EventHook = @Sendable (ModuleID, String) -> Void
    /// M59.2 — sync read of the prompt-prefix KV pool's (bytes, entries) for
    /// the snapshot. Injected so AthenaCore stays free of any AthenaLLM
    /// dependency. NOTE: the M59 prompt cache was removed in the S0 de-vendor,
    /// so this hook is currently unwired (always nil) — retained as the
    /// injection point for a future substrate prefix-cache seam (mlx-tracker
    /// #24). nil ⇒ pool disabled / not wired.
    public typealias PromptCachePoolProbe = @Sendable () -> (bytes: Int, entries: Int)
    /// M59.2 — shed the prompt-prefix KV pool (drop entries not in use) as a
    /// cheap reclaim BEFORE evicting a module or refusing a load. Injected.
    /// nil ⇒ nothing to shed. ADR 029 WP1 — `async` for the same reason as
    /// `ReclaimCacheHook`: `flushIdle`/`clearCache` are Metal-touching, so the
    /// serve-side hook runs them inside the `InferenceGate`; the governor awaits.
    public typealias PromptCacheReliefHook = @Sendable () async -> Void

    public let totalBudgetBytes: Int
    /// Governor-owned global prompt-cache byte cap (brief 4b). The LLM
    /// module reads this to refuse over-cap prompts.
    public let promptCacheCapBytes: Int
    private var entries: [ModuleID: Entry] = [:]
    private var residentBytes: Int = 0
    /// Coalesces concurrent `ensureLoaded` callers onto one load.
    private var inFlight: [ModuleID: Task<Void, Error>] = [:]
    /// M68.1 (E2) — generation token for the in-flight load currently
    /// registered under each id. `Task` is a non-identity value type, so a
    /// detached cleanup can't compare itself by reference to decide whether
    /// it still owns `inFlight[id]`; the token lets `clearInFlight` wipe the
    /// slot ONLY when it still holds the same generation it was spawned for,
    /// instead of clobbering a newer in-flight task that replaced it.
    private var inFlightToken: [ModuleID: UInt64] = [:]
    /// Monotonic source for `inFlightToken` (and, transitively, an ordering
    /// witness for load generations). Wraps with `&+=`; collisions across a
    /// 2^64 wrap are not reachable in any real lifetime.
    private var loadSeq: UInt64 = 0
    /// M68.1 (NE1) — in-flight teardown (`module.unload()`) tasks, keyed by
    /// module. A slot moves to `.unloading` and runs `unload()` in a task; a
    /// concurrent reload of the SAME slot must await this teardown before it
    /// calls `module.load()`, or load() and unload() race on the module actor
    /// with undefined order (load lands first, the still-pending unload tears
    /// it down → the governor records `.loaded` over an empty container and
    /// `/healthz` lies). Mirror of the `inFlight` load handle, for the unload
    /// direction. Cleared in `markUnloaded` when the teardown completes.
    private var teardown: [ModuleID: Task<Void, Never>] = [:]
    /// M62 — the error from the most recent FAILED load, kept so the
    /// non-blocking `beginLoadIfNeeded` path can surface the real reason to
    /// the next caller instead of silently kicking another doomed load and
    /// returning a perpetual `module_loading` 503. Set in `performLoad`'s
    /// failure path, cleared on a successful load or once surfaced (so a
    /// retry can still happen). `ensureLoaded` already throws directly, so it
    /// doesn't consume this.
    private var lastLoadError: [ModuleID: Error] = [:]
    /// M54.3 — modules with an operator-action model pull in flight
    /// (startup / allowlist-add). While set, `beginLoadIfNeeded` returns
    /// `.loading` (→ 503) WITHOUT attempting a load, so an inference
    /// request waits for the pull to materialize the weights instead of
    /// triggering a Hub download or churning eviction on a fail-fast load.
    private var pulling: Set<ModuleID> = []
    private let memoryProbe: MemoryProbe?
    private let onUnloaded: UnloadHook?
    private let onEvent: EventHook?
    private let promptCachePoolProbe: PromptCachePoolProbe?
    private let promptCacheRelief: PromptCacheReliefHook?
    /// ADR 023 G2 — live-footprint probe + cache reclaim hook + the accounting
    /// mode. When `admissionMode == .footprint` and `footprintProbe` returns a
    /// usable sample, admission meters `max(committed, reserved)`; otherwise it
    /// degrades to the reservation-only denominator (pre-G2 behavior).
    private let footprintProbe: FootprintProbe?
    private let reclaimCache: ReclaimCacheHook?
    private let admissionMode: GovernorMemory.AdmissionMode
    /// M5.4: real footprint observed on a prior load. Subsequent
    /// admissions use this instead of the static `memoryEstimate()`, so
    /// an evicted-then-reloaded module is admitted on its true cost.
    private var learnedFootprint: [ModuleID: Int] = [:]

    public init(
        totalBudgetBytes: Int, memoryProbe: MemoryProbe? = nil,
        onUnloaded: UnloadHook? = nil,
        onEvent: EventHook? = nil,
        promptCacheCapBytes: Int? = nil,
        promptCachePoolProbe: PromptCachePoolProbe? = nil,
        promptCacheRelief: PromptCacheReliefHook? = nil,
        footprintProbe: FootprintProbe? = nil,
        reclaimCache: ReclaimCacheHook? = nil,
        admissionMode: GovernorMemory.AdmissionMode = .footprint
    ) {
        let budget = Self.safeBudget(totalBudgetBytes)
        self.totalBudgetBytes = budget
        self.memoryProbe = memoryProbe
        self.onUnloaded = onUnloaded
        self.onEvent = onEvent
        self.promptCacheCapBytes =
            promptCacheCapBytes ?? (budget / 4)
        self.promptCachePoolProbe = promptCachePoolProbe
        self.promptCacheRelief = promptCacheRelief
        self.footprintProbe = footprintProbe
        self.reclaimCache = reclaimCache
        self.admissionMode = admissionMode
    }

    public init(
        config: GovernorConfig, memoryProbe: MemoryProbe? = nil,
        onUnloaded: UnloadHook? = nil,
        onEvent: EventHook? = nil,
        promptCachePoolProbe: PromptCachePoolProbe? = nil,
        promptCacheRelief: PromptCacheReliefHook? = nil,
        footprintProbe: FootprintProbe? = nil,
        reclaimCache: ReclaimCacheHook? = nil,
        admissionMode: GovernorMemory.AdmissionMode = .footprint
    ) {
        let budget = Self.safeBudget(config.totalBudgetBytes)
        self.totalBudgetBytes = budget
        self.memoryProbe = memoryProbe
        self.onUnloaded = onUnloaded
        self.onEvent = onEvent
        // E5 — derive the cap off the clamped budget too, so a non-positive
        // configured budget doesn't leak a non-positive prompt-cache cap.
        self.promptCacheCapBytes =
            config.promptCacheCapBytes > 0
            ? config.promptCacheCapBytes : (budget / 4)
        self.promptCachePoolProbe = promptCachePoolProbe
        self.promptCacheRelief = promptCacheRelief
        self.footprintProbe = footprintProbe
        self.reclaimCache = reclaimCache
        self.admissionMode = admissionMode
    }

    /// M68.1 (E5) — a non-positive budget (a misconfigured `totalBudgetBytes`,
    /// or an overflowed/garbage value) would make `makeRoom` refuse EVERY load
    /// (`residentBytes + estimate <= 0` is never true for a real model) and
    /// give `relievePressure` a nonsense high-water mark — a daemon that 503s
    /// every inference. Clamp to the physical-memory-derived default so a bad
    /// config degrades to the standard budget instead of total refusal.
    private static func safeBudget(_ b: Int) -> Int {
        b > 0
            ? b
            : Int(Double(ProcessInfo.processInfo.physicalMemory) * 0.75)
    }

    /// Register a module instance under its id. `evictable` controls whether
    /// the governor may unload it to admit another module under budget
    /// pressure.
    public func register(_ module: any InferenceModule, evictable: Bool) {
        entries[module.id] = Entry(
            module: module,
            state: .unloaded,
            reservation: nil,
            evictable: evictable,
            lastUsed: .distantPast
        )
    }

    /// M43.2 — outcome of `beginLoadIfNeeded` for request-thread call
    /// sites that must not block on a multi-GB cold-load.
    ///   - `.loaded`  ⇒ slot is hot; caller may proceed with inference.
    ///   - `.loading` ⇒ a background load is in flight (just kicked off,
    ///     or already running); caller should respond `503` with a
    ///     `Retry-After` header so the client retries shortly.
    public enum LoadStatus: Sendable {
        case loaded
        case loading
    }

    /// M43.2 — non-blocking variant of `ensureLoaded` for sync inference
    /// handlers. Returns `.loaded` if the slot is hot, otherwise starts
    /// a background load (if one isn't already in flight) and returns
    /// `.loading` immediately. The cold-load runs detached and is
    /// joinable by any concurrent `ensureLoaded` (which keeps blocking
    /// semantics — used by the queue worker and the preload-on-start
    /// path, both legitimately off the request thread).
    public func beginLoadIfNeeded(_ id: ModuleID) async throws -> LoadStatus {
        guard entries[id] != nil else {
            throw AthenaError.moduleNotRegistered(id)
        }
        await relievePressure(except: id)
        // E1 — read the slot's CURRENT state, not a value-copy captured
        // before `relievePressure` (which can mutate this actor's state).
        if entries[id]?.state == .loaded {
            entries[id]?.lastUsed = Date()
            return .loaded
        }
        // M54.3 — an operator-action pull is materializing the weights;
        // 503 until it lands rather than attempt a (doomed, churning) load.
        if pulling.contains(id) { return .loading }
        if inFlight[id] != nil { return .loading }
        // M62 — a prior load FAILED and nothing is in flight: surface the
        // real error to THIS caller instead of silently launching another
        // doomed load (which would just 503 `module_loading` forever — e.g. a
        // model dir missing config.json). NE2 (M68.1) widens this to also
        // capture admission (makeRoom) failures, so a too-large model 503s
        // with the real `memory_budget_exceeded` instead of perpetual
        // `module_loading`. Cleared as we surface it so the next request
        // retries fresh rather than wedging on a transient.
        if let err = lastLoadError[id] {
            lastLoadError[id] = nil
            throw err
        }
        loadSeq &+= 1
        let token = loadSeq
        let task = Task<Void, Error> { try await self.performLoad(id) }
        inFlight[id] = task
        inFlightToken[id] = token
        Task { [weak self] in
            _ = try? await task.value
            await self?.clearInFlight(id, token: token)
        }
        return .loading
    }

    /// E2 — clear the in-flight slot ONLY if it still holds the generation
    /// this cleanup was spawned for; a newer load that already replaced it
    /// (after this one finished) must not be wiped.
    private func clearInFlight(_ id: ModuleID, token: UInt64) {
        if inFlightToken[id] == token {
            inFlight[id] = nil
            inFlightToken[id] = nil
        }
    }

    /// M54.3 — mark/clear an operator-action pull in flight for `id`. While
    /// marked, `beginLoadIfNeeded` 503s (the weights aren't local yet).
    public func setPulling(_ id: ModuleID, _ inFlight: Bool) {
        if inFlight { pulling.insert(id) } else { pulling.remove(id) }
    }

    /// Ensure the module is loaded and bill its memory. Concurrent callers
    /// for the same module await a single shared load. Throws
    /// `AthenaError.memoryBudgetExceeded` (→ 503) when admission fails.
    public func ensureLoaded(_ id: ModuleID) async throws {
        guard entries[id] != nil else {
            throw AthenaError.moduleNotRegistered(id)
        }
        await relievePressure(except: id)
        // E1 — decide on the slot's CURRENT state, re-read after the
        // (synchronous, but state-mutating) `relievePressure` above rather
        // than a value-copy taken before it.
        if entries[id]?.state == .loaded {
            entries[id]?.lastUsed = Date()
            return
        }
        if let existing = inFlight[id] {
            try await existing.value
            return
        }
        loadSeq &+= 1
        let token = loadSeq
        let task = Task<Void, Error> { try await self.performLoad(id) }
        inFlight[id] = task
        inFlightToken[id] = token
        // E2 — clear only our own generation; a load that replaced us while
        // we awaited must survive this defer.
        defer {
            if inFlightToken[id] == token {
                inFlight[id] = nil
                inFlightToken[id] = nil
            }
        }
        try await task.value
    }

    /// ADR 015 — blocking-with-budget variant of `beginLoadIfNeeded` for the
    /// request path: wait up to `seconds` for an in-progress LOCAL load, then
    /// serve. Returns:
    ///   - `.loaded`  ⇒ slot went hot within the budget; proceed to inference.
    ///   - `.loading` ⇒ the budget elapsed OR an operator pull (download) is in
    ///     flight; caller responds `503` + `Retry-After` exactly as today.
    /// A failed / over-budget load throws its real error IMMEDIATELY (never
    /// waited on), identical to `beginLoadIfNeeded`. `seconds <= 0` ⇒ no wait
    /// (legacy immediate-503 behavior; the revert switch).
    ///
    /// Unlike `ensureLoaded`, a timeout does NOT cancel the shared load — it
    /// keeps running detached so the next request finds the slot resident. A
    /// request-side cancellation (client disconnect) ends the wait gracefully
    /// (returns `.loading`) and likewise leaves the load running. Downloads are
    /// minutes/GB and operator-initiated, so they are never waited on (the same
    /// boundary M43.2 drew — see ADR 015).
    public func awaitLoad(_ id: ModuleID, within seconds: Double) async throws
        -> LoadStatus
    {
        guard entries[id] != nil else {
            throw AthenaError.moduleNotRegistered(id)
        }
        await relievePressure(except: id)
        // E1 — read CURRENT state after the (state-mutating) relief above.
        if entries[id]?.state == .loaded {
            entries[id]?.lastUsed = Date()
            return .loaded
        }
        // A download is materializing the weights: 503 now, do not wait.
        if pulling.contains(id) { return .loading }
        // Surface a real prior failure at once (same as `beginLoadIfNeeded`).
        if let err = lastLoadError[id] {
            lastLoadError[id] = nil
            throw err
        }
        // Ensure exactly one load is in flight: join an existing one or start it.
        if inFlight[id] == nil {
            loadSeq &+= 1
            let token = loadSeq
            let started = Task<Void, Error> { try await self.performLoad(id) }
            inFlight[id] = started
            inFlightToken[id] = token
            Task { [weak self] in
                _ = try? await started.value
                await self?.clearInFlight(id, token: token)
            }
        }
        // `seconds <= 0` ⇒ don't wait; behave like the legacy non-blocking gate.
        guard seconds > 0 else { return .loading }
        // Poll the slot until it goes hot, fails, or the budget elapses. Each
        // `Task.sleep` suspends THIS actor method so `performLoad` (also an
        // actor method) can make progress; on resume we re-read live state.
        let deadline = Date().addingTimeInterval(seconds)
        let pollNanos = UInt64(loadPollIntervalMillis) * 1_000_000
        while Date() < deadline {
            do {
                try await Task.sleep(nanoseconds: pollNanos)
            } catch {
                // Request cancelled (client disconnect): stop waiting; leave the
                // detached load running for the next caller.
                return .loading
            }
            if entries[id]?.state == .loaded {
                entries[id]?.lastUsed = Date()
                return .loaded
            }
            if let err = lastLoadError[id] {
                lastLoadError[id] = nil
                throw err
            }
        }
        // Budget elapsed with the load still running — 503 + Retry-After.
        return .loading
    }

    /// ADR 015 — a streaming caller's pre-flight read: decide, WITHOUT
    /// mutating state or starting a load, whether to serve immediately, return
    /// a clean `503` (a download is materializing the weights), or start an
    /// SSE response that emits `: loading` keep-alives while `awaitLoad` runs.
    ///   - `.loaded`    ⇒ slot is hot; stream normally (no keep-alives).
    ///   - `.pulling`   ⇒ operator pull in flight; caller 503s cleanly (no SSE).
    ///   - `.needsLoad` ⇒ a local load is needed; caller opens the SSE stream
    ///     and drives the wait inside it (a prior load failure also lands here
    ///     and surfaces as an in-stream error when the caller then `awaitLoad`s).
    public enum LoadPeek: Sendable {
        case loaded
        case pulling
        case needsLoad
    }
    public func peekLoad(_ id: ModuleID) -> LoadPeek {
        if entries[id]?.state == .loaded { return .loaded }
        if pulling.contains(id) { return .pulling }
        return .needsLoad
    }

    /// ADR 015 — poll cadence for `awaitLoad`. Coarse enough to avoid needless
    /// wakeups during a multi-second load, fine enough that "ready" latency is
    /// imperceptible. Not configurable (an implementation detail, not a knob).
    private let loadPollIntervalMillis = 100

    /// M5.3 OOM guard: relief based on the LIVE substrate footprint,
    /// not the bookkeeping (which the static estimates can under-count
    /// even when reconciled). If actual MLX memory is above the
    /// high-water mark, shed the LRU evictable module (≠ the one being
    /// requested) — its `unload` + the clear-cache hook reclaim the
    /// pool. Best-effort and progressive: sustained pressure across
    /// requests sheds further victims one at a time.
    private func relievePressure(except keep: ModuleID) async {
        guard let memoryProbe else { return }
        let highWater = totalBudgetBytes / 10 * 9  // 90%
        guard memoryProbe() > highWater else { return }
        // M59.2 — shed the prompt-prefix KV pool first: it's a pure
        // performance cache (every entry is reconstructible by a cold
        // prefill), so dropping it is a cheaper, less disruptive reclaim than
        // evicting a loaded module. Best-effort; the freed MLX buffers return
        // to the budget on the next clearCache (the unload hook).
        // ADR 029 WP1 — awaited: the hook runs its MLX free under the
        // InferenceGate, so it can't race an in-flight decode. The re-read
        // below sees its effect once it lands.
        await promptCacheRelief?()
        if memoryProbe() <= highWater { return }
        let victim =
            entries
            .filter {
                $0.key != keep && $0.value.evictable
                    && $0.value.state == .loaded
            }
            .min { $0.value.lastUsed < $1.value.lastUsed }?
            .key
        if let victim { evictSync(victim) }
    }

    /// M60.6 — shed the prompt-prefix KV pool if the process footprint is over
    /// the high-water mark, INDEPENDENT of a model-load admission. Called after
    /// each generation so a pool that grew during sustained decode is reclaimed
    /// instead of staying pinned over budget until the next load (`relievePressure`
    /// above only runs on the load path). Prompt-cache only — it never evicts a
    /// loaded module mid-serving.
    ///
    /// Deliberately uses **phys_footprint**, not the RSS `memoryProbe`: the
    /// retained KV pool lives in Metal/GPU buffers that RSS under-counts (M55),
    /// so an RSS check would miss the very memory we need to shed.
    public func relievePromptCachePressureIfNeeded() async {
        guard promptCacheRelief != nil else { return }
        let highWater = totalBudgetBytes / 10 * 9  // 90%
        guard ProcessMemory.sample().physFootprint > highWater else { return }
        // ADR 029 WP1 — gated hook (awaited): shed the pool without racing a
        // concurrent tenant's decode.
        await promptCacheRelief?()
    }

    private func performLoad(_ id: ModuleID) async throws {
        guard let module = entries[id]?.module else {
            throw AthenaError.moduleNotRegistered(id)
        }
        if entries[id]?.state == .loaded { return }

        // NE1 (M68.1) — if this slot is mid-teardown (just evicted or
        // explicitly unloaded), wait for the pending `module.unload()` to
        // finish before re-loading. Otherwise load() and the detached
        // unload() race on the module actor with undefined order. After the
        // wait, re-read state (E1): a concurrent path may have loaded it.
        if let pending = teardown[id] {
            await pending.value
            if entries[id]?.state == .loaded { return }
        }

        // M5.4: prefer a previously-observed real footprint.
        let estimate: Int
        if let learned = learnedFootprint[id] {
            estimate = learned
        } else {
            estimate = await module.memoryEstimate()
            // E1 — re-read after the `memoryEstimate()` suspension.
            if entries[id]?.state == .loaded { return }
        }
        // NE2 (M68.1) — admission can fail (model larger than the budget, or
        // the budget perpetually over after eviction). Record it in
        // `lastLoadError` exactly like a `module.load()` throw, so the
        // non-blocking `beginLoadIfNeeded` path surfaces the real
        // `memory_budget_exceeded` 503 to the next caller instead of kicking
        // another doomed background load and 503-ing `module_loading` forever
        // (the detached cleanup swallows this throw via `try?`).
        do {
            try await makeRoom(for: estimate, requestedBy: id)
        } catch {
            let classified = AthenaError.classify(error, module: id)
            lastLoadError[id] = classified
            entries[id]?.unloadedReason = .loadFailed
            throw classified
        }

        let reservation = MemoryReservation(module: id, bytes: estimate)
        residentBytes += estimate
        entries[id]?.state = .loading
        entries[id]?.reservation = reservation

        let before = memoryProbe?()
        let started = Date()
        onEvent?(id, "loading (estimate \(Self.fmtBytes(estimate)))")
        do {
            try await module.load(reservation: reservation)
        } catch {
            // E6 (M68.1) — return the bytes ONLY if our reservation is still
            // on the books. A concurrent `releaseSlot` (allowlist drop) during
            // the load `await` may have already reclaimed it; a second
            // subtract here would corrupt `residentBytes` downward. Subtract
            // the reservation's own bytes, then clear it.
            if let res = entries[id]?.reservation {
                residentBytes -= res.bytes
                entries[id]?.reservation = nil
            }
            // Don't clobber a state another path set (e.g. `releaseSlot` →
            // `.unloaded`); only fall back to `.unloaded` if we still own the
            // `.loading` transition.
            if entries[id]?.state == .loading {
                entries[id]?.state = .unloaded
            }
            entries[id]?.unloadedReason = .loadFailed
            onEvent?(id, "load failed after \(Self.ms(since: started)): \(error)")
            // A Metal/MLX OOM during load is classified to 503, not a
            // bare 500 (brief item 4a).
            let classified = AthenaError.classify(error, module: id)
            // M62 — remember it so the non-blocking `beginLoadIfNeeded` path
            // surfaces the real reason to the next caller instead of a
            // perpetual `module_loading` 503.
            lastLoadError[id] = classified
            throw classified
        }
        entries[id]?.state = .loaded
        entries[id]?.lastUsed = Date()
        // M46.5 — clear the reason once a load succeeds; nil while
        // loaded is the documented "currently resident" signal.
        entries[id]?.unloadedReason = nil
        lastLoadError[id] = nil  // M62 — a success clears the prior failure

        // M56 — name WHICH model loaded, its real footprint, and how long
        // it took, so a sysadmin sees "loaded <id> (<bytes>) in <ms>"
        // instead of a bare "loaded". The specific model id comes from the
        // module (the governor keys by module class); bytes from the same
        // probe the reconcile uses.
        let after = memoryProbe?()
        var observed: Int =
            (before != nil && after != nil) ? max(after! - before!, 0) : 0
        // E12 (M68.1) — the process-global probe delta UNDERCOUNTS when a
        // concurrent teardown (another module's eviction) freed bytes between
        // `before` and `after`: the delta can fall below this module's real
        // footprint, even to ≤0, so reconcile is skipped and
        // `learnedFootprint[id]` is never recorded → the next admission keeps
        // using the static estimate. When the delta lands below the estimate
        // (the tell-tale of a deflated probe), fall back to the module's OWN
        // resident self-report so the reconcile still fires. `max(...)` keeps
        // the normal single-load case (delta ≥ estimate) byte-unchanged, and a
        // module that self-reports 0 leaves `observed` as the probe delta.
        if observed < estimate {
            observed = max(observed, await module.residentBytes)
        }
        let modelId =
            await (module as? any ModelSelectable)?.residentModelId()
        onEvent?(
            id,
            "loaded \(modelId ?? id.rawValue) "
                + "(\(Self.fmtBytes(observed > 0 ? observed : estimate))) "
                + "in \(Self.ms(since: started))")
        // M5.1: reconcile the static estimate to the real Metal/MLX
        // footprint this load actually consumed. Best-effort
        // attribution (the probe is process-global; the actor +
        // inFlight coalescing serialise loads). Honest accounting here
        // makes the NEXT admission correctly 503/evict; an over-budget
        // reconciliation also sheds other evictable modules now.
        if observed > 0 {
            reconcile(id, estimate: estimate, observed: observed)
        }
    }

    /// Human bytes for log events (GB ≥ 1 GB, else MB).
    private static func fmtBytes(_ b: Int) -> String {
        b >= 1_000_000_000
            ? String(format: "%.2fGB", Double(b) / 1e9)
            : String(format: "%.0fMB", Double(b) / 1e6)
    }
    /// Elapsed since `start` as a `<n>ms` / `<n.n>s` string.
    private static func ms(since start: Date) -> String {
        let s = Date().timeIntervalSince(start)
        return s >= 1
            ? String(format: "%.1fs", s)
            : String(format: "%.0fms", s * 1000)
    }

    /// Replace `id`'s estimate-based reservation with the observed
    /// footprint and, if that pushes the budget over, evict other
    /// evictable modules LRU-first to get back under.
    private func reconcile(
        _ id: ModuleID, estimate: Int, observed: Int
    ) {
        guard observed > 0,
            entries[id]?.state == .loaded,
            entries[id]?.reservation != nil
        else { return }
        residentBytes += observed - estimate
        entries[id]?.reservation = MemoryReservation(
            module: id, bytes: observed)
        learnedFootprint[id] = observed  // M5.4
        guard residentBytes > totalBudgetBytes else { return }
        let victims =
            entries
            .filter {
                $0.key != id && $0.value.evictable
                    && $0.value.state == .loaded
            }
            .sorted { $0.value.lastUsed < $1.value.lastUsed }
        for (victimID, _) in victims {
            if residentBytes <= totalBudgetBytes { break }
            evictSync(victimID)
        }
    }

    /// ADR 023 G2 — the admission denominator: the larger of the live
    /// footprint ceiling (`committed = phys_footprint − reclaimable cache`) and
    /// the reservation floor (`residentBytes`). Returns `residentBytes` verbatim
    /// when the mode is `.estimate`, the footprint probe is absent, or the probe
    /// failed (`physFootprint == 0`) — so the pre-G2 path and the `swift test`
    /// path (no MLX) are byte-identical. The decision algebra itself is the
    /// MLX-free unit-pinned `GovernorMemory` seam; this only supplies the probe.
    private func admissionDenominator() -> Int {
        guard admissionMode == .footprint,
            let sample = footprintProbe?(),
            sample.physFootprint > 0
        else { return residentBytes }
        let committed = GovernorMemory.committedBytes(
            physFootprint: sample.physFootprint,
            reclaimableCache: sample.cacheBytes)
        return GovernorMemory.admissionDenominator(
            mode: .footprint, committed: committed, reserved: residentBytes)
    }

    /// ADR 039 S2 — the live admission inputs the batch scheduler meters
    /// per-sequence KV reservations against: the current ADR-023 denominator
    /// (`max(committed, reserved)`) and the total budget. Reuses the same probe
    /// as request admission so batch rows and module loads compete on one number.
    public func admissionInputs() -> (denominator: Int, budget: Int) {
        (admissionDenominator(), totalBudgetBytes)
    }

    /// Free budget for `estimate` bytes, evicting evictable loaded modules
    /// LRU-first. Throws if it still cannot fit after exhausting eviction.
    private func makeRoom(for estimate: Int, requestedBy id: ModuleID) async throws {
        // ADR 023 G2 — front-door gate on the LIVE footprint, not the
        // reservation sum alone, so the genuinely-pinned resident footprint the
        // estimates were blind to can't be overcommitted.
        // `admissionDenominator()` == `residentBytes` when the probe is absent
        // or the mode is `.estimate`, so this line is the pre-G2 admit check
        // unchanged on that path.
        if GovernorMemory.fits(
            request: estimate, denominator: admissionDenominator(),
            budget: totalBudgetBytes)
        {
            return
        }

        // Rung 1 — reclaim reconstructible headroom WITHOUT evicting a tenant:
        // shed the prompt-prefix KV pool (live MLX memory ⇒ counted in
        // `committed`, so this lowers the admission ceiling) and trim the
        // reclaimable MLX buffer cache (keeps actual `phys_footprint` under the
        // hard MLX limit as the new model allocates). Then re-gate — if the pool
        // was the bloat, the load now fits without disrupting a resident module.
        // No-ops on the pure path (both hooks nil), so the re-gate just repeats
        // the front-door verdict there.
        // ADR 029 WP1 — both hooks are awaited: they run their MLX frees under
        // the InferenceGate, so this rung waits for any in-flight decode rather
        // than tearing down the buffer pool beneath it. The re-gate below still
        // sees the reclaim's effect (the hook completes before we return).
        await promptCacheRelief?()
        await reclaimCache?()
        if GovernorMemory.fits(
            request: estimate, denominator: admissionDenominator(),
            budget: totalBudgetBytes)
        {
            return
        }

        // Rung 2 — evict evictable modules LRU-first. The loop meters the
        // SYNCHRONOUS reservation accounting (`residentBytes`): `evictSync`
        // returns a victim's bytes immediately, whereas the live footprint only
        // falls once the detached `unload()` runs, so reservations are the
        // reliable within-call progress signal here.
        let candidates =
            entries
            .filter { $0.key != id && $0.value.evictable && $0.value.state == .loaded }
            .sorted { $0.value.lastUsed < $1.value.lastUsed }

        for (victimID, _) in candidates {
            if residentBytes + estimate <= totalBudgetBytes { break }
            evictSync(victimID)
        }

        // Reject if we still can't fit. After eviction the reservation gate is
        // the live signal (`committed` lags the async unloads we just issued,
        // so re-gating on it would spuriously reject loads that WILL fit once
        // the unloads complete). But when NOTHING was evictable, eviction
        // changed nothing and there is no lag — so the `committed` ceiling is
        // authoritative and rejects an overcommit the box genuinely can't fit
        // (the G2 point: stop admitting work the reservation count says fits but
        // the real footprint does not).
        let reservedOver = residentBytes + estimate > totalBudgetBytes
        let committedOver =
            candidates.isEmpty
            && !GovernorMemory.fits(
                request: estimate, denominator: admissionDenominator(),
                budget: totalBudgetBytes)
        if reservedOver || committedOver {
            throw AthenaError.memoryBudgetExceeded(
                requested: estimate,
                available: totalBudgetBytes - residentBytes,
                module: id
            )
        }
    }

    /// Synchronously drop a victim's reservation, then unload it detached.
    /// The bytes are returned to the budget immediately so the admitting
    /// caller can proceed without waiting on substrate teardown.
    private func evictSync(_ id: ModuleID) {
        guard let entry = entries[id], let reservation = entry.reservation
        else { return }
        residentBytes -= reservation.bytes
        entries[id]?.state = .unloading
        entries[id]?.reservation = nil
        // M46.5 — eviction is always memory-pressure-driven (this
        // function is reached from `makeRoom`/`reconcile`/
        // `relievePressure`); operator-initiated unloads go through
        // `unload(_:)`, allowlist-drop reclaims go through
        // `releaseSlot(_:)`. Each setter records its own reason.
        entries[id]?.unloadedReason = .memoryPressure
        onEvent?(id, "evicted (budget pressure)")
        let module = entry.module
        let hook = onUnloaded
        // NE1 (M68.1) — register the teardown so a concurrent reload of THIS
        // slot (`performLoad`) awaits `module.unload()` before calling
        // `module.load()`, instead of racing it on the module actor.
        teardown[id] = Task { [weak self] in
            // ADR 029 WP1 — `module.unload()` frees weights on the Metal pool and
            // `hook` trims the buffer cache; both must not run while a decode
            // holds the device. Gate the whole teardown span (the hook stays a
            // plain sync closure — it's serialized here, not self-gated).
            try? await InferenceGate.shared.withExclusiveExecution {
                await module.unload()
                hook?()
            }
            await self?.markUnloaded(id)
        }
    }

    private func markUnloaded(_ id: ModuleID) {
        if entries[id]?.state == .unloading {
            entries[id]?.state = .unloaded
        }
        // NE1 — the teardown for this slot is done; drop the handle so the
        // next reload doesn't await a completed task (harmless) and so the
        // map doesn't leak entries.
        teardown[id] = nil
    }

    /// M43.1 — reconcile the governor when a module drops its resident
    /// container outside the load/unload code paths (the allowlist-drop
    /// case in `setAllowedModelIds`: the module nils `container` directly
    /// when the resident id falls out of the new allowlist, so the
    /// governor's reservation + state would otherwise stay stale and
    /// `/healthz` would lie). Idempotent: a slot already at `.unloaded`
    /// with no reservation is a no-op.
    public func releaseSlot(_ id: ModuleID) {
        guard let entry = entries[id] else { return }
        if let reservation = entry.reservation {
            residentBytes -= reservation.bytes
        }
        entries[id]?.reservation = nil
        entries[id]?.state = .unloaded
        // M46.5 — allowlist-drop is operator intent (they changed the
        // allowlist; the module is reacting). Classifying it as
        // `.operatorUnload` matches `/api/models/unload` behaviour.
        entries[id]?.unloadedReason = .operatorUnload
        onEvent?(id, "evicted (allowlist drop)")
    }

    /// Explicitly unload a module and return its bytes to the budget.
    public func unload(_ id: ModuleID) async {
        guard let entry = entries[id] else { return }
        if let reservation = entry.reservation {
            residentBytes -= reservation.bytes
        }
        entries[id]?.state = .unloading
        entries[id]?.reservation = nil
        // M46.5 — `unload(_:)` is reached only from operator paths
        // (`/api/models/unload` + the `athena models unload` CLI).
        entries[id]?.unloadedReason = .operatorUnload
        onEvent?(id, "unloaded")
        // NE1 (M68.1) — route the teardown through the same handle as
        // `evictSync`, then await it. Two wins over the pre-fix inline
        // `await entry.module.unload(); state = .unloaded`:
        //   1. a concurrent reload (`performLoad`) awaits this teardown
        //      before calling `module.load()`, so load() can't race the
        //      still-pending unload() on the module actor; and
        //   2. the final `.unloaded` write is state-guarded inside
        //      `markUnloaded` (`if state == .unloading`), so a reload that
        //      already moved the slot to `.loaded` between `module.unload()`
        //      and the state write is no longer clobbered back to `.unloaded`.
        let module = entry.module
        let hook = onUnloaded
        let task = Task { [weak self] in
            // ADR 029 WP1 — gate the teardown span (see `evictSync`).
            try? await InferenceGate.shared.withExclusiveExecution {
                await module.unload()
                hook?()
            }
            await self?.markUnloaded(id)
        }
        teardown[id] = task
        await task.value
    }

    public func snapshot() -> GovernorSnapshot {
        let mods = entries.values.map {
            ModuleSnapshot(
                id: $0.module.id,
                state: $0.state,
                residentBytes: $0.reservation?.bytes ?? 0,
                evictable: $0.evictable,
                unloadedReason:
                    $0.state == .loaded ? nil : $0.unloadedReason,
                // ADR 023 G3: `learnedFootprint` is set only by a successful
                // reconcile, so its presence ⇒ residentBytes is measured.
                measured: learnedFootprint[$0.module.id] != nil
            )
        }
        .sorted { $0.id.rawValue < $1.id.rawValue }
        // ADR 023 G2 — report the HONEST free budget: `budget − max(committed,
        // reserved)` so `/healthz`/`athena ps` reflect what the box can actually
        // fit, not `budget − estimates`. Degrades to `budget − residentBytes`
        // (clamped ≥ 0) when the probe is absent or the mode is `.estimate`.
        // `residentBytes` itself stays the reservation sum (per-tenant
        // attribution + the G3 `measured` flag) — the cache is never attributed
        // to a tenant (honesty boundary).
        return GovernorSnapshot(
            totalBudgetBytes: totalBudgetBytes,
            residentBytes: residentBytes,
            freeBytes: GovernorMemory.freeBytes(
                budget: totalBudgetBytes, denominator: admissionDenominator()),
            promptCacheCapBytes: promptCacheCapBytes,
            modules: mods,
            admissionMode: admissionMode.rawValue
        )
    }
}
