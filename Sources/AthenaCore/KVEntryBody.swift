import Crypto
import Foundation

/// ADR 027 S3 — the MLX-free serialization of one prompt-cache entry's KV for
/// the disk tier, plus the content-addressing that lets a returning prompt find
/// it **without writing any prompt tokens to disk**.
///
/// A disk snapshot persists an entry at a single chosen 512-boundary `B`: the
/// recurrent (Mamba) checkpoint at `B` and each attention layer's state trimmed
/// to `B`. `KVEntryBody` packs those per-layer byte groups (each produced by
/// `KVByteCodec` in `AthenaLLM`, opaque here) into one buffer that the envelope
/// then seals. The buffer carries **no tokens** — the prompt content stays out
/// of the plaintext header and the (encrypted) body alike.
///
/// Addressing (`KVPrefixDigest`): the file key is `SHA-256(tokens[0..<B])`. On a
/// returning request we probe descending boundaries of the new prompt; a digest
/// hit means the first `B` tokens are identical (collision-resistance), so the
/// snapshot's recurrent state + trimmed attention at `B` are exactly what a cold
/// prefill of the new prompt would hold at `B` — the same bit-identical contract
/// the in-RAM path proves, now across a restart.
///
/// MLX-free: layout + digest + probe order are unit-pinned (ADR 008/009).

/// The packed KV of one entry at boundary `B`.
public struct KVEntryBody: Equatable, Sendable {
    /// Per-layer attention state bytes, `nil` at recurrent layers. Index = layer.
    public let attnSlots: [Data?]
    /// Recurrent (Mamba) checkpoint bytes keyed by layer index.
    public let recurrentLayers: [Int: Data]

    public init(attnSlots: [Data?], recurrentLayers: [Int: Data]) {
        self.attnSlots = attnSlots
        self.recurrentLayers = recurrentLayers
    }

    public static let version: UInt8 = 1

    public enum Failure: Error, Equatable {
        case truncated
        case badVersion(UInt8)
        case lengthOverflow
    }

    /// Layout (LE, fixed width):
    /// ```
    ///   u8   version
    ///   u32  attnCount
    ///   repeat attnCount:  u8 present; if present { u32 len; bytes }
    ///   u32  recurrentCount
    ///   repeat recurrentCount:  u32 layerIndex; u32 len; bytes
    /// ```
    public func encode() -> Data {
        var out = Data()
        out.appendByte(Self.version)
        out.appendU32(UInt32(attnSlots.count))
        for slot in attnSlots {
            if let slot {
                out.appendByte(1)
                out.appendU32(UInt32(slot.count))
                out.append(slot)
            } else {
                out.appendByte(0)
            }
        }
        // Sort recurrent layers by index so the encoding is deterministic.
        let recurrent = recurrentLayers.sorted { $0.key < $1.key }
        out.appendU32(UInt32(recurrent.count))
        for (layer, bytes) in recurrent {
            out.appendU32(UInt32(layer))
            out.appendU32(UInt32(bytes.count))
            out.append(bytes)
        }
        return out
    }

    public static func decode(_ data: Data) throws -> KVEntryBody {
        var cursor = Cursor(data)
        let v = try cursor.readByte()
        guard v == version else { throw Failure.badVersion(v) }
        let attnCount = try cursor.readU32()
        var attn: [Data?] = []
        attn.reserveCapacity(Int(attnCount))
        for _ in 0 ..< attnCount {
            let present = try cursor.readByte()
            attn.append(present == 1 ? try cursor.readLenPrefixed() : nil)
        }
        let recurrentCount = try cursor.readU32()
        var recurrent: [Int: Data] = [:]
        for _ in 0 ..< recurrentCount {
            let layer = Int(try cursor.readU32())
            recurrent[layer] = try cursor.readLenPrefixed()
        }
        return KVEntryBody(attnSlots: attn, recurrentLayers: recurrent)
    }

    private struct Cursor {
        private let data: Data
        private var offset: Int
        init(_ data: Data) {
            self.data = data
            self.offset = data.startIndex
        }
        private mutating func take(_ n: Int) throws -> Data {
            guard n >= 0, offset + n <= data.endIndex else { throw Failure.truncated }
            defer { offset += n }
            return data[offset ..< offset + n]
        }
        mutating func readByte() throws -> UInt8 { try take(1).first! }
        mutating func readU32() throws -> UInt32 {
            let b = try take(4)
            var v: UInt32 = 0
            for (i, byte) in b.enumerated() { v |= UInt32(byte) << (8 * i) }
            return v
        }
        mutating func readLenPrefixed() throws -> Data {
            let n = UInt64(try readU32())
            guard n <= UInt64(Int.max) else { throw Failure.lengthOverflow }
            return Data(try take(Int(n)))
        }
    }
}

/// ADR 027 S3 — content-addressing for the disk tier. The file key is
/// `SHA-256(tokens[0..<B])`; lookup probes descending chunk boundaries of the
/// returning prompt. No tokens are ever written to disk.
public enum KVPrefixDigest {

    /// `SHA-256` of the first `count` token ids serialized as little-endian
    /// `Int64`. `count` is clamped to `[0, tokens.count]`.
    public static func prefixHash(tokens: [Int], count: Int) -> Data {
        let n = max(0, min(count, tokens.count))
        var bytes = Data()
        bytes.reserveCapacity(n * 8)
        for i in 0 ..< n {
            var le = Int64(tokens[i]).littleEndian
            Swift.withUnsafeBytes(of: &le) { bytes.append(contentsOf: $0) }
        }
        return Data(SHA256.hash(data: bytes))
    }

    /// Descending candidate boundaries to probe for a prompt of `promptCount`
    /// tokens: multiples of `chunkSize` from `floor((promptCount-1)/chunk)·chunk`
    /// down to `chunk`. The `promptCount-1` cap matches the in-RAM acquire cap so
    /// a restore reproduces the final partial chunk exactly. Empty when the
    /// prompt is shorter than one chunk.
    public static func probeBoundaries(promptCount: Int, chunkSize: Int) -> [Int] {
        guard chunkSize > 0, promptCount > chunkSize else { return [] }
        let top = ((promptCount - 1) / chunkSize) * chunkSize
        guard top >= chunkSize else { return [] }
        return stride(from: top, through: chunkSize, by: -chunkSize).map { $0 }
    }
}

// MARK: - LE helpers (file-local)

extension Data {
    fileprivate mutating func appendByte(_ value: UInt8) { append(value) }
    fileprivate mutating func appendU32(_ value: UInt32) {
        var le = value.littleEndian
        Swift.withUnsafeBytes(of: &le) { append(contentsOf: $0) }
    }
}
