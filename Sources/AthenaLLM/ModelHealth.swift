import Foundation

/// Offline structural integrity check for a stored model directory —
/// no model load, no MLX. Shared by `athena verify` (M9.5b) and
/// `athena prune` (M9.5c). A dangling `pull` symlink (HF cache / SSD
/// gone) reports as `missing` because `fileExists` follows symlinks.
public enum ModelHealth {
    /// Empty ⇒ healthy. Each string is one human-readable problem.
    public static func check(_ entry: URL) -> [String] {
        let fm = FileManager.default
        // Resolve a symlinked model dir (how `pull` and `cp` alias
        // land) so enumeration sees the real contents; a dangling
        // link resolves to a non-existent path ⇒ still "missing".
        let dir = entry.resolvingSymlinksInPath()
        var isDir: ObjCBool = false
        guard fm.fileExists(atPath: dir.path, isDirectory: &isDir),
            isDir.boolValue
        else { return ["missing (absent or dangling symlink)"] }

        var problems: [String] = []

        let cfg = dir.appendingPathComponent("config.json")
        if !fm.fileExists(atPath: cfg.path) {
            problems.append("no config.json")
        } else if let d = try? Data(contentsOf: cfg),
            (try? JSONSerialization.jsonObject(with: d)) as? [String: Any]
                == nil
        {
            problems.append("invalid config.json")
        }

        problems += safetensorsProblems(dir)

        let tok = ["tokenizer.json", "tokenizer_config.json",
            "tokenizer.model", "vocab.json"]
        if !tok.contains(where: {
            fm.fileExists(
                atPath: dir.appendingPathComponent($0).path)
        }) {
            problems.append("no tokenizer")
        }
        return problems
    }

    private static func safetensorsProblems(_ dir: URL) -> [String] {
        let fm = FileManager.default
        let entries =
            (try? fm.contentsOfDirectory(
                at: dir, includingPropertiesForKeys: nil)) ?? []
        let index = dir.appendingPathComponent(
            "model.safetensors.index.json")

        if fm.fileExists(atPath: index.path) {
            // Sharded: validate the index + every referenced shard.
            guard let d = try? Data(contentsOf: index),
                let obj = try? JSONSerialization.jsonObject(with: d)
                    as? [String: Any],
                let map = obj["weight_map"] as? [String: Any]
            else { return ["invalid model.safetensors.index.json"] }
            let shards = Set(map.values.compactMap { $0 as? String })
            guard !shards.isEmpty else {
                return ["empty weight_map in index"]
            }
            return shards.sorted().compactMap { shard in
                headerProblem(dir.appendingPathComponent(shard))
            }
        }

        let st = entries.filter { $0.pathExtension == "safetensors" }
        if st.isEmpty { return ["no safetensors"] }
        return st.compactMap { headerProblem($0) }
    }

    /// Cheap safetensors integrity: the 8-byte LE header length must be
    /// sane and the header must be a JSON object with ≥1 tensor — read
    /// without pulling tensor bytes, so this is fast on huge shards.
    private static func headerProblem(_ url: URL) -> String? {
        let name = url.lastPathComponent
        guard let fh = try? FileHandle(forReadingFrom: url) else {
            return "missing shard \(name)"
        }
        defer { try? fh.close() }
        guard let lead = try? fh.read(upToCount: 8), lead.count == 8
        else { return "truncated \(name)" }
        let headerLen = lead.reduce(into: UInt64(0)) { acc, b in
            acc = (acc >> 8) | (UInt64(b) << 56)
        }
        // Size from the OPEN handle, not `attributesOfItem`: `pull` stores
        // shards in the HF-cache layout where each shard is a SYMLINK to
        // ../../blobs/<sha>. `attributesOfItem` lstats the link (tens of
        // bytes) and would spuriously fail the bound below ("corrupt
        // header") on every pulled model; the handle follows the link to
        // the real blob, matching the bytes we actually read.
        let fileSize = Int((try? fh.seekToEnd()) ?? 0)
        try? fh.seek(toOffset: 8)
        // Compare in UInt64: a corrupt all-0xFF length is UInt64.max,
        // and `Int(UInt64.max)` traps — never convert before bounding.
        let avail = fileSize >= 8 ? UInt64(fileSize - 8) : 0
        guard headerLen > 1, headerLen <= avail else {
            return "corrupt header \(name)"
        }
        let hlen = Int(headerLen)  // safe: headerLen ≤ avail ≤ Int.max
        guard let hdr = try? fh.read(upToCount: hlen),
            hdr.count == hlen,
            let obj = try? JSONSerialization.jsonObject(with: hdr)
                as? [String: Any]
        else { return "unparseable header \(name)" }
        let tensors = obj.keys.filter { $0 != "__metadata__" }
        return tensors.isEmpty ? "no tensors in \(name)" : nil
    }
}
