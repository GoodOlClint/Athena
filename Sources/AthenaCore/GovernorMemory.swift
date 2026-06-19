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
}
