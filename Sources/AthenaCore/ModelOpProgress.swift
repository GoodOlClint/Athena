import Foundation

/// Progress payload for the long model ops (`pull`/`convert`) — usability audit
/// 2026-07-02 §2/§3. Replaces the bare `(Double) -> Void` fraction closure so
/// the daemon and the in-process CLI can report *per-file* download rows, the
/// convert *phase*, and the minutes-long *quantize* loop's `i/N` — not just one
/// aggregate bar that sits silent through quantization.
///
/// MLX-free (ADR 008/009); the SSE frame encoding + throttle live in
/// AthenaServerKit and are unit-pinned.
public enum ModelOpProgress: Sendable {
    /// Aggregate download progress. `fraction` 0…1; `bytes`/`total` are the
    /// byte-weighted totals across all files (0 until known).
    case download(fraction: Double, bytes: Int64, total: Int64)
    /// One file's download progress within a multi-shard snapshot. `index` is
    /// 1-based over `count` files; `done` once fully written.
    case file(
        name: String, index: Int, count: Int,
        bytes: Int64, total: Int64, done: Bool)
    /// A coarse named phase with no sub-progress (`download`, `load`,
    /// `quantize`, `write`).
    case phase(String)
    /// The quantize materialize loop: tensor `index` of `count` (1-based).
    case quantize(index: Int, count: Int)
}
