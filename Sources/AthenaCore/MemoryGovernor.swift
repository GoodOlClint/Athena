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
}

public struct GovernorSnapshot: Sendable, Codable {
    public let totalBudgetBytes: Int
    public let residentBytes: Int
    public let freeBytes: Int
    public let promptCacheCapBytes: Int
    public let modules: [ModuleSnapshot]
    /// M59.2 — cross-request prompt-prefix KV pool (the persistent reuse
    /// cache, distinct from `promptCacheCapBytes` which is the per-request
    /// admission guard). Bytes are the pool's own estimate of its live KV
    /// snapshots; entries is the count. Both 0 when the pool is disabled or
    /// empty. Deliberately NOT folded into `residentBytes` — the live memory
    /// probe already sees these MLX buffers, so adding them there would
    /// double-count against module admission.
    public let promptCachePoolBytes: Int
    public let promptCachePoolEntries: Int

    public init(
        totalBudgetBytes: Int, residentBytes: Int, freeBytes: Int,
        promptCacheCapBytes: Int, modules: [ModuleSnapshot],
        promptCachePoolBytes: Int = 0, promptCachePoolEntries: Int = 0
    ) {
        self.totalBudgetBytes = totalBudgetBytes
        self.residentBytes = residentBytes
        self.freeBytes = freeBytes
        self.promptCacheCapBytes = promptCacheCapBytes
        self.modules = modules
        self.promptCachePoolBytes = promptCachePoolBytes
        self.promptCachePoolEntries = promptCachePoolEntries
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
    /// Called after a module finishes `unload()`, so the substrate's
    /// own buffer pool can be trimmed (the freed bytes otherwise stay
    /// cached). Injected — keeps AthenaCore substrate-agnostic.
    public typealias UnloadHook = @Sendable () -> Void
    /// Model-lifecycle observer (loading/loaded/evicted/unloaded/load
    /// failed), keyed by module. Injected so AthenaCore needs no
    /// logging dependency; the `athena` target maps it to a per-module
    /// unified-log category.
    public typealias EventHook = @Sendable (ModuleID, String) -> Void
    /// M59.2 — sync read of the prompt-prefix KV pool's (bytes, entries) for
    /// the snapshot. Injected so AthenaCore stays free of any AthenaLLM
    /// dependency; the serve path backs it with the shared `PrefixKVCache`
    /// (a lock-guarded class, so this is safe to call synchronously from
    /// `snapshot()`). nil ⇒ pool disabled / not wired.
    public typealias PromptCachePoolProbe = @Sendable () -> (bytes: Int, entries: Int)
    /// M59.2 — shed the prompt-prefix KV pool (drop entries not in use) as a
    /// cheap reclaim BEFORE evicting a module or refusing a load. Injected,
    /// sync. nil ⇒ nothing to shed.
    public typealias PromptCacheReliefHook = @Sendable () -> Void

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
        promptCacheRelief: PromptCacheReliefHook? = nil
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
    }

    public init(
        config: GovernorConfig, memoryProbe: MemoryProbe? = nil,
        onUnloaded: UnloadHook? = nil,
        onEvent: EventHook? = nil,
        promptCachePoolProbe: PromptCachePoolProbe? = nil,
        promptCacheRelief: PromptCacheReliefHook? = nil
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
    public func beginLoadIfNeeded(_ id: ModuleID) throws -> LoadStatus {
        guard entries[id] != nil else {
            throw AthenaError.moduleNotRegistered(id)
        }
        relievePressure(except: id)
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
        relievePressure(except: id)
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

    /// M5.3 OOM guard: relief based on the LIVE substrate footprint,
    /// not the bookkeeping (which the static estimates can under-count
    /// even when reconciled). If actual MLX memory is above the
    /// high-water mark, shed the LRU evictable module (≠ the one being
    /// requested) — its `unload` + the clear-cache hook reclaim the
    /// pool. Best-effort and progressive: sustained pressure across
    /// requests sheds further victims one at a time.
    private func relievePressure(except keep: ModuleID) {
        guard let memoryProbe else { return }
        let highWater = totalBudgetBytes / 10 * 9  // 90%
        guard memoryProbe() > highWater else { return }
        // M59.2 — shed the prompt-prefix KV pool first: it's a pure
        // performance cache (every entry is reconstructible by a cold
        // prefill), so dropping it is a cheaper, less disruptive reclaim than
        // evicting a loaded module. Best-effort; the freed MLX buffers return
        // to the budget on the next clearCache (the unload hook).
        promptCacheRelief?()
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
    public func relievePromptCachePressureIfNeeded() {
        guard promptCacheRelief != nil else { return }
        let highWater = totalBudgetBytes / 10 * 9  // 90%
        guard ProcessMemory.sample().physFootprint > highWater else { return }
        promptCacheRelief?()
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
            try makeRoom(for: estimate, requestedBy: id)
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
        let victims = entries
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

    /// Free budget for `estimate` bytes, evicting evictable loaded modules
    /// LRU-first. Throws if it still cannot fit after exhausting eviction.
    private func makeRoom(for estimate: Int, requestedBy id: ModuleID) throws {
        if residentBytes + estimate <= totalBudgetBytes { return }

        let candidates = entries
            .filter { $0.key != id && $0.value.evictable && $0.value.state == .loaded }
            .sorted { $0.value.lastUsed < $1.value.lastUsed }

        for (victimID, _) in candidates {
            if residentBytes + estimate <= totalBudgetBytes { break }
            evictSync(victimID)
        }

        if residentBytes + estimate > totalBudgetBytes {
            // M59.2 — last resort before refusing the load: shed the
            // prompt-prefix KV pool (a reconstructible perf cache). It isn't
            // in `residentBytes` (the live probe sees it instead), so this
            // doesn't change the arithmetic here, but it frees real Metal
            // bytes so the load that follows this admission has headroom.
            promptCacheRelief?()
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
            await module.unload()
            hook?()
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
            await module.unload()
            hook?()
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
                    $0.state == .loaded ? nil : $0.unloadedReason
            )
        }
        .sorted { $0.id.rawValue < $1.id.rawValue }
        let pool = promptCachePoolProbe?() ?? (bytes: 0, entries: 0)
        return GovernorSnapshot(
            totalBudgetBytes: totalBudgetBytes,
            residentBytes: residentBytes,
            freeBytes: totalBudgetBytes - residentBytes,
            promptCacheCapBytes: promptCacheCapBytes,
            modules: mods,
            promptCachePoolBytes: pool.bytes,
            promptCachePoolEntries: pool.entries
        )
    }
}
