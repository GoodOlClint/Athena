/// Pure model-choice bookkeeping for the single embedding slot (#236), kept
/// MLX-free so the decision is unit-testable (ADR 008/009).
///
/// Two inputs choose what a (re)load binds: `coldTarget`, the model the
/// request that triggered the load named, and `lastServed`, the last model that
/// actually loaded (NI3: a governor evict→reload restores it). Neither is
/// recorded before a load succeeds, so a model that fails to load is never
/// retried on behalf of a request that names a different model.
struct EmbeddingSlotPlan: Equatable {
    private(set) var lastServed: String?
    private(set) var coldTarget: String?

    mutating func stageColdTarget(_ id: String) { coldTarget = id }

    func loadTarget(defaultModel: () throws -> String) rethrows -> String {
        try coldTarget ?? lastServed ?? defaultModel()
    }

    mutating func loadSucceeded(_ id: String) {
        lastServed = id
        coldTarget = nil
    }

    mutating func loadFailed() { coldTarget = nil }
}
