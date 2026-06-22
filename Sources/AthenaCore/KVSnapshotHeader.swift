import Foundation

/// ADR 027 — the self-describing leading header of a disk-backed KV snapshot.
///
/// A snapshot file is `header.encode() ‖ sealed-body`, where the body is the
/// `KVFrame` bytes of the cached KV, AES-256-GCM-sealed under a per-blob data key
/// (`KVSnapshotEnvelope`). This header carries everything the restore path needs
/// **before** touching the body: the format version (so a foreign / future binary
/// **skips-on-skew** instead of mis-reading), the model+quant identity the body
/// was produced under (so a body is never restored into a different model — ADR
/// 027 honesty boundary), the swappable-KEK material (`kekType` + `kekParams` +
/// `wrappedDEK`, per the ADR 024 amendment), and the save provenance.
///
/// MLX-free: the layout + its skip-on-skew rejection are unit-pinnable under
/// `swift test` (ADR 008/009), where MLX numerics can't run. `KVByteCodec`
/// (`AthenaLLM`, MLX-gated) produces the body bytes; this type never sees an
/// `MLXArray`.
///
/// Layout (all integers little-endian, fixed width; strings + blobs are
/// `u32` length-prefixed):
/// ```
///   u8 × 8   magic = "ATHN-KVS"
///   u16      formatVersion
///   u8       kekType
///   u8       saveReason
///   str      modelID          (u32 len ‖ utf8)
///   str      quantTag         (u32 len ‖ utf8)
///   str      scopeKey         (u32 len ‖ utf8)
///   u64      tokenCount
///   u64      contextSize
///   u64      createdUnix
///   u64      lastUsedUnix
///   blob     prefixHash       (u32 len ‖ bytes)
///   blob     kekParams        (u32 len ‖ bytes — salt / ephemeral pubkey)
///   blob     wrappedDEK       (u32 len ‖ bytes — DEK wrapped by the KEK)
/// ```
public struct KVSnapshotHeader: Equatable, Sendable {

    /// The current on-disk format version. A decode of any other version is a
    /// `skip-on-skew` (`Failure.unsupportedVersion`), never a best-effort parse.
    public static let formatVersion: UInt16 = 1

    /// Magic prefix — rejects a truncated / foreign / corrupt file up front.
    static let magic = Data("ATHN-KVS".utf8)

    /// Why this snapshot was written (the downstream client-style provenance, ADR 027 §5).
    public enum SaveReason: UInt8, Equatable, Sendable {
        case cold = 1        // long prompt reached a stable prefix
        case continued = 2   // a frontier save during a long session
        case evict = 3       // demoted to disk under governor pressure
        case shutdown = 4    // flushed on graceful drain
    }

    public var formatVersion: UInt16
    public var kekType: KEKType
    public var saveReason: SaveReason
    /// The model this body was produced under — half of the restore gate.
    public var modelID: String
    /// The packaging/quant identity (e.g. "4bit-mtp") — the other half. A body is
    /// only restorable into a byte-identical model+quant (`isRestorable`).
    public var quantTag: String
    /// Principal / `prompt_cache_key` scope the entry was cached under.
    public var scopeKey: String
    public var tokenCount: UInt64
    public var contextSize: UInt64
    public var createdUnix: UInt64
    public var lastUsedUnix: UInt64
    /// Hash of the rendered byte-prefix (also the file key); carried for integrity.
    public var prefixHash: Data
    /// Opaque KEK material the matching `KEKProvider` needs to unwrap the DEK
    /// (HKDF salt for keyfile, ephemeral pubkey for SEP). Never secret.
    public var kekParams: Data
    /// The per-blob data key, wrapped by the KEK.
    public var wrappedDEK: Data

    public init(
        formatVersion: UInt16 = KVSnapshotHeader.formatVersion,
        kekType: KEKType,
        saveReason: SaveReason,
        modelID: String,
        quantTag: String,
        scopeKey: String,
        tokenCount: UInt64,
        contextSize: UInt64,
        createdUnix: UInt64,
        lastUsedUnix: UInt64,
        prefixHash: Data,
        kekParams: Data,
        wrappedDEK: Data
    ) {
        self.formatVersion = formatVersion
        self.kekType = kekType
        self.saveReason = saveReason
        self.modelID = modelID
        self.quantTag = quantTag
        self.scopeKey = scopeKey
        self.tokenCount = tokenCount
        self.contextSize = contextSize
        self.createdUnix = createdUnix
        self.lastUsedUnix = lastUsedUnix
        self.prefixHash = prefixHash
        self.kekParams = kekParams
        self.wrappedDEK = wrappedDEK
    }

    public enum Failure: Error, Equatable {
        case truncated
        case badMagic
        /// A version this binary does not understand — the restore path treats
        /// this as a clean skip (go cold), not an error to surface.
        case unsupportedVersion(UInt16)
        case badEnum
        case lengthOverflow
    }

    // MARK: - Restore gate

    /// The model/quant half of the restore gate (ADR 027): a body may only be
    /// restored into the **exact** model+quant identity it was produced under.
    /// A mismatch ⇒ the caller goes cold (re-prefill), never wrong bytes.
    public func isRestorable(forModel model: String, quant: String) -> Bool {
        modelID == model && quantTag == quant
    }

    // MARK: - Encode

    public func encode() -> Data {
        var out = Data()
        out.append(Self.magic)
        out.appendLE(formatVersion)
        out.appendLE(kekType.rawValue)
        out.appendLE(saveReason.rawValue)
        out.appendLEString(modelID)
        out.appendLEString(quantTag)
        out.appendLEString(scopeKey)
        out.appendLE(tokenCount)
        out.appendLE(contextSize)
        out.appendLE(createdUnix)
        out.appendLE(lastUsedUnix)
        out.appendLEBlob(prefixHash)
        out.appendLEBlob(kekParams)
        out.appendLEBlob(wrappedDEK)
        return out
    }

    // MARK: - Decode

    /// Parse a header from the front of a snapshot file. Returns the header and
    /// the offset at which the sealed body begins (the caller slices the rest as
    /// the body). Throws `Failure` on a bad magic, an unknown version
    /// (skip-on-skew), a bad enum, or truncation — every throw is a "go cold"
    /// signal, never a crash.
    public static func decode(_ data: Data) throws -> (header: KVSnapshotHeader, bodyOffset: Int) {
        var cursor = Cursor(data)
        let magic = try cursor.readBytes(UInt64(Self.magic.count))
        guard magic == Self.magic else { throw Failure.badMagic }
        let version = try cursor.readU16()
        guard version == formatVersion else {
            throw Failure.unsupportedVersion(version)
        }
        guard let kekType = KEKType(rawValue: try cursor.readU8()) else {
            throw Failure.badEnum
        }
        guard let saveReason = SaveReason(rawValue: try cursor.readU8()) else {
            throw Failure.badEnum
        }
        let modelID = try cursor.readString()
        let quantTag = try cursor.readString()
        let scopeKey = try cursor.readString()
        let tokenCount = try cursor.readU64()
        let contextSize = try cursor.readU64()
        let createdUnix = try cursor.readU64()
        let lastUsedUnix = try cursor.readU64()
        let prefixHash = try cursor.readBlob()
        let kekParams = try cursor.readBlob()
        let wrappedDEK = try cursor.readBlob()
        let header = KVSnapshotHeader(
            formatVersion: version,
            kekType: kekType,
            saveReason: saveReason,
            modelID: modelID,
            quantTag: quantTag,
            scopeKey: scopeKey,
            tokenCount: tokenCount,
            contextSize: contextSize,
            createdUnix: createdUnix,
            lastUsedUnix: lastUsedUnix,
            prefixHash: prefixHash,
            kekParams: kekParams,
            wrappedDEK: wrappedDEK)
        return (header, cursor.offset)
    }

    // MARK: - Cursor

    private struct Cursor {
        private let data: Data
        private(set) var offset: Int
        init(_ data: Data) {
            self.data = data
            self.offset = data.startIndex
        }

        private mutating func take(_ n: Int) throws -> Data {
            guard n >= 0, offset + n <= data.endIndex else {
                throw Failure.truncated
            }
            let slice = data[offset ..< offset + n]
            offset += n
            return slice
        }

        mutating func readU8() throws -> UInt8 { try take(1).first! }
        mutating func readU16() throws -> UInt16 {
            UInt16(littleEndianBytes: try take(2))
        }
        mutating func readU64() throws -> UInt64 {
            UInt64(littleEndianBytes: try take(8))
        }
        mutating func readBytes(_ n: UInt64) throws -> Data {
            guard n <= UInt64(Int.max) else { throw Failure.lengthOverflow }
            return Data(try take(Int(n)))
        }
        mutating func readBlob() throws -> Data {
            try readBytes(UInt64(try readU32()))
        }
        mutating func readString() throws -> String {
            String(decoding: try readBlob(), as: UTF8.self)
        }
        private mutating func readU32() throws -> UInt32 {
            UInt32(littleEndianBytes: try take(4))
        }
    }
}

// MARK: - Little-endian helpers (file-local, mirrors KVFrame's framing)

extension Data {
    fileprivate mutating func appendLE(_ value: UInt8) { append(value) }
    fileprivate mutating func appendLE(_ value: UInt16) {
        var le = value.littleEndian
        Swift.withUnsafeBytes(of: &le) { append(contentsOf: $0) }
    }
    fileprivate mutating func appendLE(_ value: UInt32) {
        var le = value.littleEndian
        Swift.withUnsafeBytes(of: &le) { append(contentsOf: $0) }
    }
    fileprivate mutating func appendLE(_ value: UInt64) {
        var le = value.littleEndian
        Swift.withUnsafeBytes(of: &le) { append(contentsOf: $0) }
    }
    fileprivate mutating func appendLEBlob(_ blob: Data) {
        appendLE(UInt32(blob.count))
        append(blob)
    }
    fileprivate mutating func appendLEString(_ string: String) {
        appendLEBlob(Data(string.utf8))
    }
}

extension UInt16 {
    fileprivate init(littleEndianBytes data: Data) {
        var value: UInt16 = 0
        for (i, byte) in data.enumerated() { value |= UInt16(byte) << (8 * i) }
        self = value
    }
}

extension UInt32 {
    fileprivate init(littleEndianBytes data: Data) {
        var value: UInt32 = 0
        for (i, byte) in data.enumerated() { value |= UInt32(byte) << (8 * i) }
        self = value
    }
}

extension UInt64 {
    fileprivate init(littleEndianBytes data: Data) {
        var value: UInt64 = 0
        for (i, byte) in data.enumerated() { value |= UInt64(byte) << (8 * i) }
        self = value
    }
}
