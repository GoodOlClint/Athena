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

    public let totalBudgetBytes: Int
    /// Governor-owned global prompt-cache byte cap (brief 4b). The LLM
    /// module reads this to refuse over-cap prompts.
    public let promptCacheCapBytes: Int
    private var entries: [ModuleID: Entry] = [:]
    private var residentBytes: Int = 0
    /// Coalesces concurrent `ensureLoaded` callers onto one load.
    private var inFlight: [ModuleID: Task<Void, Error>] = [:]
    private let memoryProbe: MemoryProbe?
    private let onUnloaded: UnloadHook?
    private let onEvent: EventHook?
    /// M5.4: real footprint observed on a prior load. Subsequent
    /// admissions use this instead of the static `memoryEstimate()`, so
    /// an evicted-then-reloaded module is admitted on its true cost.
    private var learnedFootprint: [ModuleID: Int] = [:]

    public init(
        totalBudgetBytes: Int, memoryProbe: MemoryProbe? = nil,
        onUnloaded: UnloadHook? = nil,
        onEvent: EventHook? = nil,
        promptCacheCapBytes: Int? = nil
    ) {
        self.totalBudgetBytes = totalBudgetBytes
        self.memoryProbe = memoryProbe
        self.onUnloaded = onUnloaded
        self.onEvent = onEvent
        self.promptCacheCapBytes =
            promptCacheCapBytes ?? (totalBudgetBytes / 4)
    }

    public init(
        config: GovernorConfig, memoryProbe: MemoryProbe? = nil,
        onUnloaded: UnloadHook? = nil,
        onEvent: EventHook? = nil
    ) {
        self.totalBudgetBytes = config.totalBudgetBytes
        self.memoryProbe = memoryProbe
        self.onUnloaded = onUnloaded
        self.onEvent = onEvent
        self.promptCacheCapBytes = config.promptCacheCapBytes
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
        guard let entry = entries[id] else {
            throw AthenaError.moduleNotRegistered(id)
        }
        relievePressure(except: id)
        if entry.state == .loaded {
            entries[id]?.lastUsed = Date()
            return .loaded
        }
        if inFlight[id] != nil { return .loading }
        let task = Task<Void, Error> { try await self.performLoad(id) }
        inFlight[id] = task
        Task { [weak self] in
            _ = try? await task.value
            await self?.clearInFlight(id)
        }
        return .loading
    }

    private func clearInFlight(_ id: ModuleID) {
        inFlight[id] = nil
    }

    /// Ensure the module is loaded and bill its memory. Concurrent callers
    /// for the same module await a single shared load. Throws
    /// `AthenaError.memoryBudgetExceeded` (→ 503) when admission fails.
    public func ensureLoaded(_ id: ModuleID) async throws {
        guard let entry = entries[id] else {
            throw AthenaError.moduleNotRegistered(id)
        }
        relievePressure(except: id)
        if entry.state == .loaded {
            entries[id]?.lastUsed = Date()
            return
        }
        if let existing = inFlight[id] {
            try await existing.value
            return
        }
        let task = Task<Void, Error> { try await self.performLoad(id) }
        inFlight[id] = task
        defer { inFlight[id] = nil }
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

    private func performLoad(_ id: ModuleID) async throws {
        guard let entry = entries[id] else {
            throw AthenaError.moduleNotRegistered(id)
        }
        if entry.state == .loaded { return }

        // M5.4: prefer a previously-observed real footprint.
        let estimate: Int
        if let learned = learnedFootprint[id] {
            estimate = learned
        } else {
            estimate = await entry.module.memoryEstimate()
        }
        try makeRoom(for: estimate, requestedBy: id)

        let reservation = MemoryReservation(module: id, bytes: estimate)
        residentBytes += estimate
        entries[id]?.state = .loading
        entries[id]?.reservation = reservation

        let before = memoryProbe?()
        let started = Date()
        onEvent?(id, "loading (estimate \(Self.fmtBytes(estimate)))")
        do {
            try await entry.module.load(reservation: reservation)
        } catch {
            residentBytes -= estimate
            entries[id]?.state = .unloaded
            entries[id]?.reservation = nil
            entries[id]?.unloadedReason = .loadFailed
            onEvent?(id, "load failed after \(Self.ms(since: started)): \(error)")
            // A Metal/MLX OOM during load is classified to 503, not a
            // bare 500 (brief item 4a).
            throw AthenaError.classify(error, module: id)
        }
        entries[id]?.state = .loaded
        entries[id]?.lastUsed = Date()
        // M46.5 — clear the reason once a load succeeds; nil while
        // loaded is the documented "currently resident" signal.
        entries[id]?.unloadedReason = nil
        // M56 — name WHICH model loaded, its real footprint, and how long
        // it took, so a sysadmin sees "loaded <id> (<bytes>) in <ms>"
        // instead of a bare "loaded". The specific model id comes from the
        // module (the governor keys by module class); bytes from the same
        // probe the reconcile uses.
        let after = memoryProbe?()
        let observed: Int =
            (before != nil && after != nil) ? max(after! - before!, 0) : 0
        let modelId =
            await (entry.module as? any ModelSelectable)?.residentModelId()
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
        Task { [weak self] in
            await module.unload()
            hook?()
            await self?.markUnloaded(id)
        }
    }

    private func markUnloaded(_ id: ModuleID) {
        if entries[id]?.state == .unloading {
            entries[id]?.state = .unloaded
        }
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
        await entry.module.unload()
        onUnloaded?()
        onEvent?(id, "unloaded")
        entries[id]?.state = .unloaded
        // M46.5 — `unload(_:)` is reached only from operator paths
        // (`/api/models/unload` + the `athena models unload` CLI).
        entries[id]?.unloadedReason = .operatorUnload
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
        return GovernorSnapshot(
            totalBudgetBytes: totalBudgetBytes,
            residentBytes: residentBytes,
            freeBytes: totalBudgetBytes - residentBytes,
            promptCacheCapBytes: promptCacheCapBytes,
            modules: mods
        )
    }
}
