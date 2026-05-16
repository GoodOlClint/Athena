import Foundation

/// Point-in-time view of the global budget, exposed by the health endpoint
/// and the `athena ps` CLI.
public struct ModuleSnapshot: Sendable, Codable {
    public let id: ModuleID
    public let state: ModuleState
    public let reservedBytes: Int
    public let evictable: Bool
}

public struct GovernorSnapshot: Sendable, Codable {
    public let totalBudgetBytes: Int
    public let reservedBytes: Int
    public let freeBytes: Int
    public let modules: [ModuleSnapshot]
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
    }

    public let totalBudgetBytes: Int
    private var entries: [ModuleID: Entry] = [:]
    private var reservedBytes: Int = 0
    /// Coalesces concurrent `ensureLoaded` callers onto one load.
    private var inFlight: [ModuleID: Task<Void, Error>] = [:]

    public init(totalBudgetBytes: Int) {
        self.totalBudgetBytes = totalBudgetBytes
    }

    public init(config: GovernorConfig) {
        self.totalBudgetBytes = config.totalBudgetBytes
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

    /// Ensure the module is loaded and bill its memory. Concurrent callers
    /// for the same module await a single shared load. Throws
    /// `AthenaError.memoryBudgetExceeded` (→ 503) when admission fails.
    public func ensureLoaded(_ id: ModuleID) async throws {
        guard let entry = entries[id] else {
            throw AthenaError.moduleNotRegistered(id)
        }
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

    private func performLoad(_ id: ModuleID) async throws {
        guard let entry = entries[id] else {
            throw AthenaError.moduleNotRegistered(id)
        }
        if entry.state == .loaded { return }

        let estimate = await entry.module.memoryEstimate()
        try makeRoom(for: estimate, requestedBy: id)

        let reservation = MemoryReservation(module: id, bytes: estimate)
        reservedBytes += estimate
        entries[id]?.state = .loading
        entries[id]?.reservation = reservation

        do {
            try await entry.module.load(reservation: reservation)
        } catch {
            reservedBytes -= estimate
            entries[id]?.state = .unloaded
            entries[id]?.reservation = nil
            throw AthenaError.moduleLoadFailed(
                id, reason: String(describing: error))
        }
        entries[id]?.state = .loaded
        entries[id]?.lastUsed = Date()
    }

    /// Free budget for `estimate` bytes, evicting evictable loaded modules
    /// LRU-first. Throws if it still cannot fit after exhausting eviction.
    private func makeRoom(for estimate: Int, requestedBy id: ModuleID) throws {
        if reservedBytes + estimate <= totalBudgetBytes { return }

        let candidates = entries
            .filter { $0.key != id && $0.value.evictable && $0.value.state == .loaded }
            .sorted { $0.value.lastUsed < $1.value.lastUsed }

        for (victimID, _) in candidates {
            if reservedBytes + estimate <= totalBudgetBytes { break }
            evictSync(victimID)
        }

        if reservedBytes + estimate > totalBudgetBytes {
            throw AthenaError.memoryBudgetExceeded(
                requested: estimate,
                available: totalBudgetBytes - reservedBytes,
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
        reservedBytes -= reservation.bytes
        entries[id]?.state = .unloading
        entries[id]?.reservation = nil
        let module = entry.module
        Task { [weak self] in
            await module.unload()
            await self?.markUnloaded(id)
        }
    }

    private func markUnloaded(_ id: ModuleID) {
        if entries[id]?.state == .unloading {
            entries[id]?.state = .unloaded
        }
    }

    /// Explicitly unload a module and return its bytes to the budget.
    public func unload(_ id: ModuleID) async {
        guard let entry = entries[id] else { return }
        if let reservation = entry.reservation {
            reservedBytes -= reservation.bytes
        }
        entries[id]?.state = .unloading
        entries[id]?.reservation = nil
        await entry.module.unload()
        entries[id]?.state = .unloaded
    }

    public func snapshot() -> GovernorSnapshot {
        let mods = entries.values.map {
            ModuleSnapshot(
                id: $0.module.id,
                state: $0.state,
                reservedBytes: $0.reservation?.bytes ?? 0,
                evictable: $0.evictable
            )
        }
        .sorted { $0.id.rawValue < $1.id.rawValue }
        return GovernorSnapshot(
            totalBudgetBytes: totalBudgetBytes,
            reservedBytes: reservedBytes,
            freeBytes: totalBudgetBytes - reservedBytes,
            modules: mods
        )
    }
}
