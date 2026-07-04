import Foundation

/// Fused Multi-Token-Prediction detection for a servable LLM checkpoint
/// (usability audit 2026-07-02 §4). `ModelSupport` (ADR 021) classifies a
/// Qwen3.5 `-mtp` checkpoint as `.llm` like any other — the fused MTP head is
/// an *attribute of a servable LLM*, not a modality — and config alone can't
/// tell a `-mtp` build from a stock same-family checkpoint: `mtp_num_hidden_layers`
/// is an architectural constant present even in checkpoints that ship NO
/// `mtp.*` weights. The only reliable signal is the weight index.
///
/// Relocated here from `AthenaLLM.AthenaModelRegistration` (which still calls
/// it at load time to decide whether to construct the MTP head) so it is
/// MLX-free and unit-pinnable under `swift test` (ADR 008/009). The
/// honesty-boundary note (ADR 021 decision 4, amended for this probe): this
/// reads the safetensors weight index — **one step past config-only** — but
/// stays pure filesystem I/O, no MLX, no model load.
public enum MTPCheckpoint {
    /// True iff the checkpoint at `dir` actually contains `mtp.*` weights.
    /// nil dir ⇒ true (defer to config; the load path always sets the dir).
    public static func checkpointHasMTP(_ dir: URL?) -> Bool {
        guard let dir else { return true }
        let index = dir.appendingPathComponent("model.safetensors.index.json")
        if let data = try? Data(contentsOf: index),
            let obj = try? JSONSerialization.jsonObject(with: data)
                as? [String: Any],
            let weightMap = obj["weight_map"] as? [String: Any]
        {
            return weightMap.keys.contains { $0.contains("mtp.") }
        }
        // No index (single-file checkpoint): scan safetensors headers.
        let entries =
            (try? FileManager.default.contentsOfDirectory(
                at: dir, includingPropertiesForKeys: nil)) ?? []
        for url in entries where url.pathExtension == "safetensors" {
            if safetensorsHeaderHasMTP(url) { return true }
        }
        return entries.contains { $0.pathExtension == "safetensors" }
            ? false : true
    }

    private static func safetensorsHeaderHasMTP(_ url: URL) -> Bool {
        guard let fh = try? FileHandle(forReadingFrom: url) else {
            return false
        }
        defer { try? fh.close() }
        guard let lenData = try? fh.read(upToCount: 8), lenData.count == 8
        else { return false }
        let headerLen = lenData.withUnsafeBytes {
            $0.load(as: UInt64.self).littleEndian
        }
        guard headerLen > 0, headerLen < 50_000_000,
            let header = try? fh.read(upToCount: Int(headerLen)),
            let json = String(data: header, encoding: .utf8)
        else { return false }
        return json.contains("mtp.")
    }

    /// A servable LLM's fused-MTP attribute (audit §4): does the weight index
    /// actually carry `mtp.*` tensors? The **weight index is the authority** —
    /// config is not a reliable gate (real Qwen3.5 `-mtp` converts drop
    /// `mtp_num_hidden_layers` entirely, and the substring matches the real
    /// `language_model.mtp.*` prefix), so this is a direct index scan.
    /// The caller restricts it to the `.llm` modality, so no non-generative
    /// arch is ever probed. Symlink-resolved like the other detectors; reads
    /// one small JSON, no MLX, no model load.
    public static func hasFusedMTP(in dir: URL) -> Bool {
        checkpointHasMTP(dir.resolvingSymlinksInPath())
    }
}
