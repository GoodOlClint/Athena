import Foundation

/// ADR 024 Tier 3 — MLX-free wire framing for the prompt-cache KV
/// serialize/deserialize seam.
///
/// `KVByteCodec` (`AthenaLLM`, MLX-gated) turns each cached `[MLXArray]` into a
/// list of `(shape, dtypeCode, bytes)` slots and back; this type is the pure
/// byte layout those slots round-trip through — `IdleKVCipher` then seals the
/// framed `Data`. Keeping the format here (no `MLXArray` reference, opaque
/// `dtypeCode`) makes the layout + its malformed-input rejection unit-pinnable
/// under `swift test` (ADR 008/009), where the MLX numerics can't run.
///
/// This is also the seam a future disk-backed cache (M59.5) reuses: the same
/// framing serialized to a file instead of an in-RAM ciphertext buffer.
///
/// Layout (all integers little-endian, fixed width):
/// ```
///   u8   version (= 1)
///   u32  slotCount
///   repeated slotCount times:
///     u32  dtypeCode      (opaque to this type; owned by KVByteCodec)
///     u32  ndim
///     i64 × ndim  shape
///     u64  byteLength
///     u8  × byteLength  raw element bytes (native dtype, contiguous)
/// ```
public enum KVFrame {

    public static let version: UInt8 = 1

    /// One serialized tensor: its shape, an opaque dtype code, and the raw
    /// contiguous element bytes in that dtype.
    public struct Slot: Equatable {
        public let shape: [Int]
        public let dtypeCode: UInt32
        public let bytes: Data
        public init(shape: [Int], dtypeCode: UInt32, bytes: Data) {
            self.shape = shape
            self.dtypeCode = dtypeCode
            self.bytes = bytes
        }
    }

    public enum Failure: Error, Equatable {
        case truncated
        case badVersion(UInt8)
        case negativeDimension
        case lengthOverflow
    }

    // MARK: - Encode

    public static func encode(_ slots: [Slot]) -> Data {
        var out = Data()
        out.appendLE(version)
        out.appendLE(UInt32(slots.count))
        for slot in slots {
            out.appendLE(slot.dtypeCode)
            out.appendLE(UInt32(slot.shape.count))
            for dim in slot.shape { out.appendLE(Int64(dim)) }
            out.appendLE(UInt64(slot.bytes.count))
            out.append(slot.bytes)
        }
        return out
    }

    // MARK: - Decode

    /// Parse a buffer produced by `encode`. Throws `Failure` on any
    /// inconsistency rather than crashing — the input is always a successfully
    /// GCM-opened (so authentic) buffer in production, but a defensive parse
    /// keeps a codec bug from becoming a trap and lets the unit tests pin the
    /// rejection paths.
    public static func decode(_ data: Data) throws -> [Slot] {
        var cursor = Cursor(data)
        let v = try cursor.readU8()
        guard v == version else { throw Failure.badVersion(v) }
        let count = try cursor.readU32()
        var slots: [Slot] = []
        slots.reserveCapacity(Int(count))
        for _ in 0 ..< count {
            let dtypeCode = try cursor.readU32()
            let ndim = try cursor.readU32()
            var shape: [Int] = []
            shape.reserveCapacity(Int(ndim))
            for _ in 0 ..< ndim {
                let dim = try cursor.readI64()
                guard dim >= 0 else { throw Failure.negativeDimension }
                shape.append(Int(dim))
            }
            let byteLen = try cursor.readU64()
            let bytes = try cursor.readBytes(byteLen)
            slots.append(
                Slot(shape: shape, dtypeCode: dtypeCode, bytes: bytes))
        }
        return slots
    }

    // MARK: - Cursor

    private struct Cursor {
        private let data: Data
        private var offset: Int
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

        mutating func readU8() throws -> UInt8 {
            try take(1).first!
        }
        mutating func readU32() throws -> UInt32 {
            UInt32(littleEndianBytes: try take(4))
        }
        mutating func readU64() throws -> UInt64 {
            UInt64(littleEndianBytes: try take(8))
        }
        mutating func readI64() throws -> Int64 {
            Int64(bitPattern: UInt64(littleEndianBytes: try take(8)))
        }
        mutating func readBytes(_ n: UInt64) throws -> Data {
            guard n <= UInt64(Int.max) else { throw Failure.lengthOverflow }
            // Return a zero-based copy so callers get a contiguous, index-0 Data.
            return Data(try take(Int(n)))
        }
    }
}

// MARK: - Little-endian helpers

extension Data {
    fileprivate mutating func appendLE(_ value: UInt8) { append(value) }
    fileprivate mutating func appendLE(_ value: UInt32) {
        var le = value.littleEndian
        Swift.withUnsafeBytes(of: &le) { append(contentsOf: $0) }
    }
    fileprivate mutating func appendLE(_ value: UInt64) {
        var le = value.littleEndian
        Swift.withUnsafeBytes(of: &le) { append(contentsOf: $0) }
    }
    fileprivate mutating func appendLE(_ value: Int64) {
        appendLE(UInt64(bitPattern: value))
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
