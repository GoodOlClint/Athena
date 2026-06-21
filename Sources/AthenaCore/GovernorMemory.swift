import Foundation

/// MLX-free governor memory-accounting helpers (ADR 023). Pure decision logic so
/// it's unit-pinned (ADR 008/009); the MLX/OS probes live at the call sites.
public enum GovernorMemory {
    /// G1 — resolve the effective serve-path MLX buffer-cache limit in bytes.
    ///
    /// - A configured value wins (CLI flag or TOML `mlx_cache_limit_bytes`).
    /// - `nil` (absent) ⇒ a fraction of the Metal budget (default ~⅓), large
    ///   enough to avoid alloc/free churn but bounded so the cache can't grow to
    ///   fill the whole budget (the ADR-023 field finding).
    /// - `≤ 0` ⇒ `nil` (unbounded — today's behavior, an explicit opt-out).
    /// - A non-positive budget with no configured value ⇒ `nil` (can't size a
    ///   fraction of nothing; leave MLX's default).
    ///
    /// Returns the limit to set via `MLX.Memory.cacheLimit`, or `nil` to leave
    /// MLX's default (unbounded) in place.
    public static func resolveCacheLimit(
        configured: Int?, budgetBytes: Int, fractionDenominator: Int = 3
    ) -> Int? {
        if let configured { return configured > 0 ? configured : nil }
        guard budgetBytes > 0, fractionDenominator > 0 else { return nil }
        return budgetBytes / fractionDenominator
    }

    // MARK: - G2 — admission against the real footprint

    /// How the governor sizes the admission denominator (ADR 023 G2).
    ///
    /// - `.footprint` (default): admit against `max(committed, reserved)`, where
    ///   `committed = phys_footprint − reclaimable MLX cache` is the live ceiling
    ///   and `reserved` (the reservation sum) is the warmup-window floor. Catches
    ///   the ungoverned cache + the real resident footprint the pre-G2 estimate
    ///   math was blind to.
    /// - `.estimate`: the pre-G2 path — admit against the reservation sum alone.
    ///   The revert switch (`governor_admission_mode = "estimate"`).
    public enum AdmissionMode: String, Sendable, Codable {
        case footprint
        case estimate

        /// Parse the `governor_admission_mode` config string. Absent / unknown /
        /// empty ⇒ `.footprint` (the correctness default; the knob is an escape
        /// hatch, not an opt-in gate).
        public static func parse(_ raw: String?) -> AdmissionMode {
            switch raw?.lowercased() {
            case "estimate": return .estimate
            case "footprint": return .footprint
            default: return .footprint
            }
        }
    }

    /// The genuinely-pinned memory: `phys_footprint` minus the reclaimable MLX
    /// buffer cache (MLX active + mmap'd weight pages). The cache is reclaimable
    /// headroom, so it is excluded from the committed total — admission reclaims
    /// it (clearCache) before evicting a tenant. Clamped ≥ 0 so a probe race
    /// (cache momentarily reported larger than the footprint) can't go negative.
    public static func committedBytes(
        physFootprint: Int, reclaimableCache: Int
    ) -> Int {
        max(physFootprint - reclaimableCache, 0)
    }

    /// The number admission meters `request + denominator ≤ budget` against.
    ///
    /// - `.footprint`: `max(committed, reserved)` — the live footprint is the
    ///   ceiling, the reservation sum is the floor during the lazy-mmap fault-in
    ///   window (so a just-loaded-but-cold model can't be transiently
    ///   double-admitted before its weights fault in and lift `committed`).
    /// - `.estimate`: `reserved` — the pre-G2 reservation-only denominator.
    ///
    /// Call-site contract: when the footprint probe is unavailable (nil — e.g.
    /// under `swift test`, or a Mach failure), pass `mode: .estimate` so the
    /// denominator is the reservation sum and the result is byte-identical to
    /// the pre-G2 path.
    public static func admissionDenominator(
        mode: AdmissionMode, committed: Int, reserved: Int
    ) -> Int {
        switch mode {
        case .estimate: return reserved
        case .footprint: return max(committed, reserved)
        }
    }

    /// Honest free budget for the snapshot/healthz: `budget − denominator`,
    /// clamped ≥ 0 (an over-budget footprint reports 0 free, not a negative).
    public static func freeBytes(budget: Int, denominator: Int) -> Int {
        max(budget - denominator, 0)
    }

    /// Admission predicate: does `request` fit on top of `denominator` within
    /// `budget`? The whole admission decision reduces to this total function
    /// over plain Ints — no MLX/Mach — so it is unit-pinned (ADR 008/009).
    public static func fits(request: Int, denominator: Int, budget: Int) -> Bool {
        denominator + request <= budget
    }
}
