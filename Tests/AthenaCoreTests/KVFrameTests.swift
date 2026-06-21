import Foundation
import XCTest

@testable import AthenaCore

/// ADR 024 Tier 3 — the KV serialize/deserialize wire format is MLX-free, so its
/// round-trip and malformed-input rejection are unit-pinned here (ADR 008/009).
/// In production the buffer decode() sees is always a GCM-authenticated one, but
/// a defensive parse keeps a codec bug from becoming a trap.
final class KVFrameTests: XCTestCase {

    func testRoundTripMultipleSlots() throws {
        let slots = [
            KVFrame.Slot(
                shape: [2, 3], dtypeCode: 9,
                bytes: Data((0 ..< 12).map { UInt8($0) })),  // 6 × f16 = 12 bytes
            KVFrame.Slot(shape: [], dtypeCode: 10, bytes: Data([1, 2, 3, 4])),  // scalar
            KVFrame.Slot(shape: [0], dtypeCode: 11, bytes: Data()),  // empty tensor
        ]
        let decoded = try KVFrame.decode(KVFrame.encode(slots))
        XCTAssertEqual(decoded, slots)
    }

    func testEmptySlotListRoundTrips() throws {
        XCTAssertEqual(try KVFrame.decode(KVFrame.encode([])), [])
    }

    func testDecodeEmptyDataThrowsTruncated() {
        XCTAssertThrowsError(try KVFrame.decode(Data())) { error in
            XCTAssertEqual(error as? KVFrame.Failure, .truncated)
        }
    }

    func testTruncatedFrameThrows() throws {
        let full = KVFrame.encode([
            KVFrame.Slot(
                shape: [4], dtypeCode: 7, bytes: Data([10, 20, 30, 40]))
        ])
        // Drop the final byte ⇒ the declared payload runs past the buffer.
        let truncated = full.dropLast()
        XCTAssertThrowsError(try KVFrame.decode(Data(truncated))) { error in
            XCTAssertEqual(error as? KVFrame.Failure, .truncated)
        }
    }

    func testBadVersionThrows() throws {
        var frame = KVFrame.encode([
            KVFrame.Slot(shape: [1], dtypeCode: 0, bytes: Data([0]))
        ])
        frame[frame.startIndex] = 0xfe  // corrupt the version byte
        XCTAssertThrowsError(try KVFrame.decode(frame)) { error in
            XCTAssertEqual(error as? KVFrame.Failure, .badVersion(0xfe))
        }
    }

    func testNegativeDimensionThrows() {
        // Hand-build a frame whose single shape dim is -1 (i64). encode() can't
        // produce this, so we assemble the little-endian bytes directly.
        var frame = Data()
        frame.append(1)  // version
        frame.append(contentsOf: [1, 0, 0, 0])  // slotCount = 1 (u32)
        frame.append(contentsOf: [9, 0, 0, 0])  // dtypeCode = 9 (u32)
        frame.append(contentsOf: [1, 0, 0, 0])  // ndim = 1 (u32)
        frame.append(contentsOf: [UInt8](repeating: 0xff, count: 8))  // dim = -1 (i64)
        frame.append(contentsOf: [UInt8](repeating: 0, count: 8))  // byteLen = 0 (u64)
        XCTAssertThrowsError(try KVFrame.decode(frame)) { error in
            XCTAssertEqual(error as? KVFrame.Failure, .negativeDimension)
        }
    }
}
