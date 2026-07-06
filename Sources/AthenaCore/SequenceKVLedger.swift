import Foundation

/// ADR 039 — the batch's outstanding per-sequence KV reservations.
///
/// A pure value type (unit-pinned, MLX-free per ADR 008/009): it tracks how many
/// worst-case KV bytes each live batched row has reserved, and admits a new row
/// only if its reservation fits on top of the governor's live denominator plus
/// the batch's already-reserved KV. The batch scheduler (ADR 039 S2) owns one
/// instance under its actor and drives admit/release as rows join and finish; the
/// governor's `denominator`/`budget` are passed in at each decision so the ledger
/// stays free of any MLX/Mach probe.
///
/// Why worst-case up front: reserving `maxTokens × perTokenKVBytes` at admission
/// means the batch's committed KV can never outgrow what admission approved — the
/// ADR-023 mid-batch budget blowout this ledger exists to prevent. Denied rows
/// reserve nothing (they fall back to the queue / serial path), so a full batch
/// simply stops admitting rather than overcommitting the Metal budget.
public struct SequenceKVLedger: Sendable {
    /// uid → reserved worst-case KV bytes for that live row.
    private var reservations: [Int: Int] = [:]

    public init() {}

    /// Total KV bytes currently reserved across all live rows.
    public var totalReservedBytes: Int { reservations.values.reduce(0, +) }

    /// Number of live rows holding a reservation.
    public var activeCount: Int { reservations.count }

    /// Attempt to admit `uid` reserving `rowKVBytes`, given the governor's current
    /// `denominator` (ADR-023 `max(committed, reserved)`) and `budget`. On success
    /// records the reservation and returns `true`; on a budget miss records
    /// nothing and returns `false` (the caller denies the batch seat). A duplicate
    /// `uid` is treated as already-admitted (idempotent success) without
    /// double-counting.
    @discardableResult
    public mutating func admit(
        uid: Int, rowKVBytes: Int, denominator: Int, budget: Int
    ) -> Bool {
        if reservations[uid] != nil { return true }
        guard
            GovernorMemory.admitsSequence(
                rowKVBytes: rowKVBytes,
                activeSequenceKVBytes: totalReservedBytes,
                denominator: denominator, budget: budget)
        else { return false }
        reservations[uid] = rowKVBytes
        return true
    }

    /// Release a completed or cancelled row's reservation. Idempotent; returns the
    /// bytes freed (0 if the uid held nothing).
    @discardableResult
    public mutating func release(uid: Int) -> Int {
        reservations.removeValue(forKey: uid) ?? 0
    }
}
