import AthenaCore
import Foundation

/// SSE frame encoding + throttle for the model-op progress stream (usability
/// audit 2026-07-02 §2/§3). Pure and MLX-free (ADR 008/009) so the wire format
/// and the emit-rate decision are unit-pinned off the daemon graph.
///
/// Frames are additive over the legacy `{"event":"progress","fraction":F}`:
/// old clients ignore unknown `event`s (verified: RemoteModels `default: break`),
/// and the new client tolerates a daemon that only sends `progress` (falls back
/// to a single bar). That two-way compatibility is what the frame tests pin.
public enum ModelOpProgressFrame {
    /// Encode one progress payload as the JSON body of an SSE `data:` frame.
    /// Keys are stable; new fields are additive.
    public static func json(_ p: ModelOpProgress) -> Data {
        let obj: [String: Any]
        switch p {
        case let .download(fraction, bytes, total):
            obj = [
                "event": "progress", "fraction": fraction,
                "bytes": bytes, "total": total,
            ]
        case let .file(name, index, count, bytes, total, done):
            obj = [
                "event": "file", "name": name, "index": index,
                "count": count, "bytes": bytes, "total": total, "done": done,
            ]
        case let .phase(name):
            obj = ["event": "phase", "phase": name]
        case let .quantize(index, count):
            obj = ["event": "quantize", "index": index, "count": count]
        }
        return
            (try? JSONSerialization.data(
                withJSONObject: obj, options: [.sortedKeys]))
            ?? Data(#"{"event":"progress","fraction":0}"#.utf8)
    }

    /// Per-key emit throttle: emit at most once per `minIntervalMs`, unless the
    /// fraction moved by ≥ `minFractionDelta` (so completion/first frames aren't
    /// swallowed) or it's a terminal `done` frame. Pure — the caller holds the
    /// `last*` state per key (per file path / the aggregate).
    public static func shouldEmit(
        nowMs: Int, lastMs: Int?, fraction: Double, lastFraction: Double?,
        done: Bool = false,
        minIntervalMs: Int = 500, minFractionDelta: Double = 0.01
    ) -> Bool {
        if done { return true }
        guard let lastMs, let lastFraction else { return true }  // first frame
        if fraction - lastFraction >= minFractionDelta { return true }
        return nowMs - lastMs >= minIntervalMs
    }
}
