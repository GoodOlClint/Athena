import Foundation

/// Pure model-store filesystem operations (M16.2), shared by the
/// `athena` CLI (`list`/`show`/`rm`/`cp`) and the native HTTP
/// `/api/models*` handlers — one implementation, no duplication.
/// Foundation-only, no MLX/model load. Long-running fetch/convert
/// (`pull`/`convert`) live elsewhere and are queue-dispatched (M16.3).
public enum ModelStoreOps {
    public struct Entry: Sendable {
        public let name: String
        public let bytes: Int
        public let modified: Date
    }

    public struct Detail: Sendable {
        public let name: String
        public let path: String
        public let bytes: Int
        /// Raw `config.json` bytes (the caller parses/embeds it).
        public let configJSON: Data
    }

    public enum OpError: Error, CustomStringConvertible {
        case invalidName(String)
        case notFound(String)
        case exists(String)
        case io(String)
        public var description: String {
            switch self {
            case .invalidName(let n): return "invalid model name '\(n)'"
            case .notFound(let n): return "no model '\(n)'"
            case .exists(let n): return "'\(n)' already exists"
            case .io(let m): return m
            }
        }
    }

    /// Sum of `*.safetensors` bytes, resolving symlinks. `pull` links a
    /// model dir to the HF cache `snapshots/<hash>/` whose entries are
    /// themselves symlinks into `blobs/`; the link's own size is ~0, so
    /// both the dir and each file must be symlink-resolved.
    public static func safetensorsSize(_ dir: URL) -> Int {
        let real = dir.resolvingSymlinksInPath()
        let entries =
            (try? FileManager.default.contentsOfDirectory(
                at: real, includingPropertiesForKeys: nil)) ?? []
        return
            entries
            .filter { $0.pathExtension == "safetensors" }
            .reduce(0) {
                let f = $1.resolvingSymlinksInPath()
                let s =
                    (try? f.resourceValues(forKeys: [.fileSizeKey]))?
                    .fileSize ?? 0
                return $0 + s
            }
    }

    public static func humanBytes(_ n: Int) -> String {
        let u = ["B", "KB", "MB", "GB", "TB"]
        var v = Double(n), i = 0
        while v >= 1024, i < u.count - 1 { v /= 1024; i += 1 }
        return String(format: i == 0 ? "%.0f %@" : "%.1f %@", v, u[i])
    }

    /// Reject anything that isn't a bare child of the store root (no
    /// path traversal / absolute escape).
    public static func isValidName(_ name: String) -> Bool {
        !name.isEmpty && !name.contains("/") && name != ".."
            && name != "."
    }

    /// Models in the store (a child dir with a `config.json`), sorted
    /// by name. Empty if the store dir is absent.
    public static func list(root: URL) -> [Entry] {
        let fm = FileManager.default
        guard
            let entries = try? fm.contentsOfDirectory(
                at: root,
                includingPropertiesForKeys: [
                    .isDirectoryKey, .contentModificationDateKey,
                ])
        else { return [] }
        var rows: [Entry] = []
        for dir in entries {
            let cfg = dir.appendingPathComponent("config.json")
            guard fm.fileExists(atPath: cfg.path) else { continue }
            let mod =
                (try? dir.resourceValues(forKeys: [
                    .contentModificationDateKey
                ]))?.contentModificationDate ?? .distantPast
            rows.append(
                Entry(
                    name: dir.lastPathComponent,
                    bytes: safetensorsSize(dir), modified: mod))
        }
        return rows.sorted { $0.name < $1.name }
    }

    /// A single model's path/size/config, or nil if it has no
    /// readable `config.json` under the store root.
    public static func show(root: URL, name: String) -> Detail? {
        guard isValidName(name) else { return nil }
        let dir = root.appendingPathComponent(
            name, isDirectory: true)
        let cfg = dir.appendingPathComponent("config.json")
        guard let data = try? Data(contentsOf: cfg) else {
            return nil
        }
        return Detail(
            name: name, path: dir.path,
            bytes: safetensorsSize(dir), configJSON: data)
    }

    /// Delete a model directory (a direct child of the store root).
    public static func remove(root: URL, name: String) throws {
        guard isValidName(name) else {
            throw OpError.invalidName(name)
        }
        let dir = root.appendingPathComponent(
            name, isDirectory: true)
        var isDir: ObjCBool = false
        guard
            FileManager.default.fileExists(
                atPath: dir.path, isDirectory: &isDir),
            isDir.boolValue
        else { throw OpError.notFound(name) }
        do {
            try FileManager.default.removeItem(at: dir)
        } catch {
            throw OpError.io("remove failed: \(error)")
        }
    }

    /// Alias (symlink) or deep-copy `src` to a new bare `dst` name.
    /// Mirrors `athena cp`: default symlink (no multi-GB copy), or a
    /// real copy with `deepCopy`. Returns the destination URL.
    @discardableResult
    public static func copy(
        root: URL, src: String, dst: String,
        deepCopy: Bool, force: Bool
    ) throws -> URL {
        let store = ModelStore(rootDirectory: root)
        let source = store.resolve(src)
        let fm = FileManager.default
        guard fm.fileExists(atPath: source.path) else {
            throw OpError.notFound(src)
        }
        guard isValidName(dst) else {
            throw OpError.invalidName(dst)
        }
        let dest = root.appendingPathComponent(
            dst, isDirectory: true)
        if fm.fileExists(atPath: dest.path)
            || (try? dest.checkResourceIsReachable()) == true
        {
            guard force else { throw OpError.exists(dst) }
            try? fm.removeItem(at: dest)
        } else {
            // Clear a dangling symlink at the destination too.
            try? fm.removeItem(at: dest)
        }
        do {
            try fm.createDirectory(
                at: root, withIntermediateDirectories: true)
            if deepCopy {
                try fm.copyItem(
                    at: source.resolvingSymlinksInPath(), to: dest)
            } else {
                try fm.createSymbolicLink(
                    at: dest, withDestinationURL: source)
            }
        } catch {
            throw OpError.io("cp failed: \(error)")
        }
        return dest
    }
}
