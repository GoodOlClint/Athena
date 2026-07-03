import Crypto
import Foundation

/// ADR 027 S2 — the on-disk KV-snapshot store: flat files in the data dir keyed
/// by rendered-prefix hash (not SQLite, per the recorded M59.5 home). Each file
/// is `KVSnapshotHeader.encode() ‖ envelope-sealed body`.
///
/// MLX-free and `Data`-level: the body bytes arrive already produced by
/// `KVByteCodec` (`AthenaLLM`, MLX-gated) at the call site, so the store —
/// file layout, atomic + 0600 writes, skip-on-skew load, and retention — is
/// unit-pinnable under `swift test` (ADR 008/009). The store does **no** policy
/// of its own about *when* to save/restore; that is the governor + cache wiring
/// (S3). It is purely a content-addressed, encrypted blob shelf.
///
/// Off by default at the config layer (ADR 027 / ADR 025): the wiring only
/// constructs a store when `prompt_cache_persist_to_disk` is on and the daemon
/// is not in a no-write loopback posture.
public struct KVSnapshotStore {

    /// The directory holding `<hex-prefix-hash>.kvs` files. Created 0700 on first
    /// write if absent.
    public let directory: URL
    public static let fileExtension = "kvs"

    public init(directory: URL) {
        self.directory = directory
    }

    public enum Failure: Error, Equatable {
        case prefixHashEmpty
    }

    // MARK: - Addressing

    /// `<hex(prefixHash)>.kvs` — content-addressed, so a byte-prefix hit maps
    /// straight to its file (the downstream client's SHA1-of-prefix scheme).
    public func fileURL(forPrefixHash prefixHash: Data) -> URL {
        directory.appendingPathComponent(
            "\(Self.hex(prefixHash)).\(Self.fileExtension)")
    }

    // MARK: - Save

    /// Seal `body` under a fresh per-blob DEK (wrapped by `kek`) and write
    /// `header ‖ sealed-body` atomically with 0600 perms. The body is bound by
    /// AAD to its `(modelID, quantTag, prefixHash)` identity so it can't be
    /// paired with a different header.
    public func save(
        prefixHash: Data,
        modelID: String,
        quantTag: String,
        scopeKey: String,
        tokenCount: UInt64,
        contextSize: UInt64,
        saveReason: KVSnapshotHeader.SaveReason,
        createdUnix: UInt64,
        lastUsedUnix: UInt64,
        body: Data,
        kek: KEKProvider
    ) throws {
        guard !prefixHash.isEmpty else { throw Failure.prefixHashEmpty }
        try ensureDirectory()
        let aad = Self.aad(modelID: modelID, quantTag: quantTag, prefixHash: prefixHash)
        let sealed = try KVSnapshotEnvelope.seal(body, aad: aad, kek: kek)
        let header = KVSnapshotHeader(
            kekType: kek.kekType,
            saveReason: saveReason,
            modelID: modelID,
            quantTag: quantTag,
            scopeKey: scopeKey,
            tokenCount: tokenCount,
            contextSize: contextSize,
            createdUnix: createdUnix,
            lastUsedUnix: lastUsedUnix,
            prefixHash: prefixHash,
            kekParams: sealed.kekParams,
            wrappedDEK: sealed.wrappedDEK)
        let file = header.encode() + sealed.body
        let url = fileURL(forPrefixHash: prefixHash)
        try file.write(to: url, options: .atomic)
        try? FileManager.default.setAttributes(
            [.posixPermissions: 0o600], ofItemAtPath: url.path)
    }

    // MARK: - Load

    /// Read + decrypt the body for `prefixHash`, or `nil` on any "go cold"
    /// signal: missing file, bad magic / unknown version / bad enum / truncation
    /// (`decode` throws), a model/quant mismatch (skip-on-skew), a `prefixHash`
    /// integrity mismatch, or a failed decrypt (wrong/rotated KEK, tamper, AAD).
    /// WP11 — `onUnrestorable` distinguishes a **present-but-unrestorable**
    /// snapshot (corrupt header, integrity/tamper, or a failed decrypt — the
    /// operator should see snapshot-dir corruption) from the silent, EXPECTED
    /// go-cold cases (no file at all; a model/quant skew that skip-on-skew is
    /// designed to ignore). AthenaCore stays dependency-free — the AthenaLLM
    /// caller wires the closure to a `notice` log.
    public func load(
        prefixHash: Data, requireModel: String, requireQuant: String,
        kek: KEKProvider,
        onUnrestorable: (@Sendable (String) -> Void)? = nil
    ) -> Data? {
        let url = fileURL(forPrefixHash: prefixHash)
        guard let file = try? Data(contentsOf: url) else { return nil }  // absent: normal
        guard let (header, bodyOffset) = try? KVSnapshotHeader.decode(file) else {
            onUnrestorable?("corrupt/unreadable header")
            return nil
        }
        // A model/quant mismatch is skip-on-skew (expected after a model change),
        // NOT corruption — go cold silently.
        guard header.isRestorable(forModel: requireModel, quant: requireQuant)
        else { return nil }
        guard header.prefixHash == prefixHash else {
            onUnrestorable?("prefix-hash integrity mismatch")
            return nil
        }
        let sealed = KVSnapshotEnvelope.Sealed(
            body: Data(file[bodyOffset...]),
            kekParams: header.kekParams,
            wrappedDEK: header.wrappedDEK)
        let aad = Self.aad(
            modelID: header.modelID, quantTag: header.quantTag, prefixHash: prefixHash)
        guard let body = KVSnapshotEnvelope.open(sealed, aad: aad, kek: kek) else {
            onUnrestorable?("decrypt/auth failed (wrong or rotated KEK, tamper, or AAD mismatch)")
            return nil
        }
        return body
    }

    // MARK: - Index + retention

    /// One on-disk snapshot's retention-relevant facts, read from its header
    /// without decrypting the body.
    public struct Indexed: Equatable {
        public let prefixHash: Data
        public let url: URL
        public let byteSize: Int
        public let lastUsedUnix: UInt64
    }

    /// Enumerate the store's `.kvs` files, parsing each header (cheap — no
    /// decrypt). Files that fail to parse are skipped (a partial/foreign file is
    /// not a valid snapshot).
    public func index() -> [Indexed] {
        let urls =
            (try? FileManager.default.contentsOfDirectory(
                at: directory, includingPropertiesForKeys: [.fileSizeKey],
                options: [.skipsHiddenFiles])) ?? []
        var out: [Indexed] = []
        for url in urls where url.pathExtension == Self.fileExtension {
            guard let file = try? Data(contentsOf: url),
                let (header, _) = try? KVSnapshotHeader.decode(file)
            else { continue }
            out.append(
                Indexed(
                    prefixHash: header.prefixHash, url: url, byteSize: file.count,
                    lastUsedUnix: header.lastUsedUnix))
        }
        return out
    }

    /// Apply the retention caps, deleting the selected files (LRU + age, see
    /// `KVSnapshotRetention`). Returns the prefix-hashes evicted.
    @discardableResult
    public func enforceRetention(
        maxEntries: Int?, maxBytes: Int?, maxAgeSecs: UInt64?, now: UInt64
    ) -> [Data] {
        let items = index().map {
            KVSnapshotRetention.Item(
                id: $0.prefixHash, bytes: $0.byteSize, lastUsedUnix: $0.lastUsedUnix)
        }
        let victims = KVSnapshotRetention.toEvict(
            items, maxEntries: maxEntries, maxBytes: maxBytes, maxAgeSecs: maxAgeSecs,
            now: now)
        for id in victims { delete(prefixHash: id) }
        return victims
    }

    public func delete(prefixHash: Data) {
        try? FileManager.default.removeItem(at: fileURL(forPrefixHash: prefixHash))
    }

    // MARK: - Helpers

    private func ensureDirectory() throws {
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true,
            attributes: [.posixPermissions: 0o700])
    }

    /// Bind a sealed body to its snapshot identity.
    static func aad(modelID: String, quantTag: String, prefixHash: Data) -> Data {
        var aad = Data(modelID.utf8)
        aad.append(0x1)
        aad.append(Data(quantTag.utf8))
        aad.append(0x1)
        aad.append(prefixHash)
        return aad
    }

    private static func hex(_ data: Data) -> String {
        data.map { String(format: "%02x", $0) }.joined()
    }
}

/// ADR 027 — the **pure** retention selection: which snapshots to evict to
/// satisfy the count / bytes / age caps. Separated from the file I/O so the
/// policy is unit-pinned (ADR 008/009) without touching disk.
public enum KVSnapshotRetention {

    public struct Item: Equatable {
        public let id: Data
        public let bytes: Int
        public let lastUsedUnix: UInt64
        public init(id: Data, bytes: Int, lastUsedUnix: UInt64) {
            self.id = id
            self.bytes = bytes
            self.lastUsedUnix = lastUsedUnix
        }
    }

    /// Returns the ids to evict. Age-expired entries go first; then, oldest-used
    /// first (LRU), until both `maxEntries` and `maxBytes` are satisfied. A nil
    /// or non-positive cap is "unbounded" for that dimension.
    public static func toEvict(
        _ items: [Item], maxEntries: Int?, maxBytes: Int?, maxAgeSecs: UInt64?,
        now: UInt64
    ) -> [Data] {
        var evicted: Set<Data> = []

        // 1) Age: anything idle longer than maxAgeSecs (when set & positive).
        if let maxAge = maxAgeSecs, maxAge > 0 {
            for item in items where now >= item.lastUsedUnix {
                if now - item.lastUsedUnix > maxAge { evicted.insert(item.id) }
            }
        }

        // Survivors, oldest-used first — the LRU eviction order.
        var survivors = items.filter { !evicted.contains($0.id) }
            .sorted { $0.lastUsedUnix < $1.lastUsedUnix }

        // 2) Count cap.
        if let maxEntries, maxEntries > 0 {
            while survivors.count > maxEntries {
                evicted.insert(survivors.removeFirst().id)
            }
        }

        // 3) Byte cap.
        if let maxBytes, maxBytes > 0 {
            var total = survivors.reduce(0) { $0 + $1.bytes }
            while total > maxBytes, !survivors.isEmpty {
                let victim = survivors.removeFirst()
                total -= victim.bytes
                evicted.insert(victim.id)
            }
        }

        // Preserve input order in the returned list for deterministic tests.
        return items.map(\.id).filter { evicted.contains($0) }
    }
}
