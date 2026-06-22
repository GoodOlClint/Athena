import Foundation
import XCTest

@testable import AthenaCore

/// ADR 027 S3 — the disk entry-body codec, prefix digest, and probe order are
/// MLX-free, so they are unit-pinned here (ADR 008/009).
final class KVEntryBodyTests: XCTestCase {

    // MARK: - KVEntryBody

    func testRoundTripMixedSlots() throws {
        let body = KVEntryBody(
            attnSlots: [Data("attn0".utf8), nil, Data("attn2".utf8), nil],
            recurrentLayers: [1: Data("rec1".utf8), 3: Data("rec3".utf8)])
        let decoded = try KVEntryBody.decode(body.encode())
        XCTAssertEqual(decoded, body)
    }

    func testRoundTripEmpty() throws {
        let body = KVEntryBody(attnSlots: [], recurrentLayers: [:])
        XCTAssertEqual(try KVEntryBody.decode(body.encode()), body)
    }

    func testRoundTripAllNilAttn() throws {
        let body = KVEntryBody(attnSlots: [nil, nil, nil], recurrentLayers: [0: Data([1, 2, 3])])
        XCTAssertEqual(try KVEntryBody.decode(body.encode()), body)
    }

    func testDeterministicEncoding() {
        // Recurrent dict order must not affect the bytes (sorted by layer).
        let a = KVEntryBody(attnSlots: [nil], recurrentLayers: [2: Data([2]), 0: Data([0]), 1: Data([1])])
        let b = KVEntryBody(attnSlots: [nil], recurrentLayers: [0: Data([0]), 1: Data([1]), 2: Data([2])])
        XCTAssertEqual(a.encode(), b.encode())
    }

    func testBadVersionRejected() {
        var bytes = KVEntryBody(attnSlots: [], recurrentLayers: [:]).encode()
        bytes[bytes.startIndex] = 0x99
        XCTAssertThrowsError(try KVEntryBody.decode(bytes)) {
            XCTAssertEqual($0 as? KVEntryBody.Failure, .badVersion(0x99))
        }
    }

    func testTruncatedRejected() {
        let full = KVEntryBody(
            attnSlots: [Data((0 ..< 64).map { UInt8($0) })], recurrentLayers: [0: Data([1, 2])]
        ).encode()
        XCTAssertThrowsError(try KVEntryBody.decode(Data(full.prefix(full.count / 2)))) {
            XCTAssertEqual($0 as? KVEntryBody.Failure, .truncated)
        }
    }

    // MARK: - KVPrefixDigest

    func testPrefixHashDeterministicAndStable() {
        let tokens = [10, 20, 30, 40, 50]
        XCTAssertEqual(
            KVPrefixDigest.prefixHash(tokens: tokens, count: 3),
            KVPrefixDigest.prefixHash(tokens: tokens, count: 3))
        XCTAssertEqual(KVPrefixDigest.prefixHash(tokens: tokens, count: 3).count, 32)
    }

    func testPrefixHashMatchesOnSharedPrefix() {
        // The core contract: two prompts sharing the first B tokens hash equal
        // at B, regardless of what follows.
        let a = [1, 2, 3, 4, 5]
        let b = [1, 2, 3, 99, 100]
        XCTAssertEqual(
            KVPrefixDigest.prefixHash(tokens: a, count: 3),
            KVPrefixDigest.prefixHash(tokens: b, count: 3))
        XCTAssertNotEqual(
            KVPrefixDigest.prefixHash(tokens: a, count: 4),
            KVPrefixDigest.prefixHash(tokens: b, count: 4),
            "diverging at token 4 ⇒ different hash at B=4")
    }

    func testPrefixHashCountClamped() {
        let tokens = [1, 2, 3]
        XCTAssertEqual(
            KVPrefixDigest.prefixHash(tokens: tokens, count: 99),
            KVPrefixDigest.prefixHash(tokens: tokens, count: 3))
        XCTAssertEqual(
            KVPrefixDigest.prefixHash(tokens: tokens, count: 0),
            KVPrefixDigest.prefixHash(tokens: [], count: 0))
    }

    func testProbeBoundariesDescending() {
        // promptCount 1100, chunk 512: top = floor(1099/512)*512 = 1024.
        XCTAssertEqual(
            KVPrefixDigest.probeBoundaries(promptCount: 1100, chunkSize: 512), [1024, 512])
    }

    func testProbeBoundariesShortPromptEmpty() {
        XCTAssertEqual(KVPrefixDigest.probeBoundaries(promptCount: 512, chunkSize: 512), [])
        XCTAssertEqual(KVPrefixDigest.probeBoundaries(promptCount: 100, chunkSize: 512), [])
    }

    func testProbeBoundariesExcludesFinalPartialChunk() {
        // promptCount exactly 1025: top = floor(1024/512)*512 = 1024 (≤ count-1).
        XCTAssertEqual(
            KVPrefixDigest.probeBoundaries(promptCount: 1025, chunkSize: 512), [1024, 512])
        // promptCount exactly 1024: top = floor(1023/512)*512 = 512.
        XCTAssertEqual(
            KVPrefixDigest.probeBoundaries(promptCount: 1024, chunkSize: 512), [512])
    }
}
