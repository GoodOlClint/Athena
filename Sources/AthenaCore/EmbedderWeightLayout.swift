import Foundation

/// #205 — the substrate's Gemma 3 embedder looks its backbone weights up under
/// a `model.` prefix. The upstream sentence-transformers export stores them
/// at the top level (`embed_tokens.weight`, `layers.N.*`, `norm.weight`), and
/// the substrate does not remap them, so the load dies deep with `keyNotFound`.
public enum EmbedderWeightLayout {
    /// `model_type`s whose substrate loader expects the `model.` prefix.
    static let prefixedBackboneTypes: Set<String> = ["gemma3_text"]

    /// True when `keys` hold a backbone at the top level with no `model.` prefix.
    static func lacksModelPrefix(keys: [String]) -> Bool {
        keys.contains("embed_tokens.weight")
            && !keys.contains { $0.hasPrefix("model.") }
    }

    /// Weight keys of `<dir>/model.safetensors` (header read only) or, for a
    /// sharded checkpoint, of its index. nil when neither is readable.
    static func weightKeys(in dir: URL) -> [String]? {
        let index = dir.appendingPathComponent("model.safetensors.index.json")
        if let data = try? Data(contentsOf: index),
            let obj = try? JSONSerialization.jsonObject(with: data)
                as? [String: Any],
            let map = obj["weight_map"] as? [String: Any]
        {
            return Array(map.keys)
        }
        guard
            let fh = try? FileHandle(
                forReadingFrom:
                    dir.appendingPathComponent("model.safetensors"))
        else { return nil }
        defer { try? fh.close() }
        guard let lenData = try? fh.read(upToCount: 8), lenData.count == 8
        else { return nil }
        let headerLen = lenData.withUnsafeBytes {
            $0.loadUnaligned(as: UInt64.self)
        }
        guard headerLen > 0, headerLen < 50_000_000,
            let header = try? fh.read(upToCount: Int(headerLen)),
            let obj = try? JSONSerialization.jsonObject(with: header)
                as? [String: Any]
        else { return nil }
        return obj.keys.filter { $0 != "__metadata__" }
    }

    /// Whether `dir` holds a `modelType` checkpoint whose weights the
    /// substrate loader cannot find for lack of the `model.` prefix.
    public static func isUnprefixedExport(
        modelType: String?, in dir: URL
    ) -> Bool {
        guard let type = modelType?.lowercased(),
            prefixedBackboneTypes.contains(type),
            let keys = weightKeys(in: dir.resolvingSymlinksInPath())
        else { return false }
        return lacksModelPrefix(keys: keys)
    }
}
