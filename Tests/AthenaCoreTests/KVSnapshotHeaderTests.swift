import Foundation
import XCTest

@testable import AthenaCore

/// ADR 027 — the snapshot header's layout + skip-on-skew gate are MLX-free, so
/// the round-trip, version/model/quant rejection, and truncation paths are
/// unit-pinned here (ADR 008/009).
final class KVSnapshotHeaderTests: XCTestCase {

    private func sample(
        modelID: String = "Qwen/Qwen3.5-27B",
        quantTag: String = "4bit-mtp",
        kekType: KEKType = .keyfile,
        saveReason: KVSnapshotHeader.SaveReason = .cold
    ) -> KVSnapshotHeader {
        KVSnapshotHeader(
            kekType: kekType,
            saveReason: saveReason,
            modelID: modelID,
            quantTag: quantTag,
            scopeKey: "principal=alice\u{1}key=abc",
            tokenCount: 2868,
            contextSize: 32768,
            createdUnix: 1_750_000_000,
            lastUsedUnix: 1_750_000_500,
            prefixHash: Data((0 ..< 32).map { UInt8($0) }),
            kekParams: Data((0 ..< 32).map { UInt8(0xa0 &+ $0) }),
            wrappedDEK: Data((0 ..< 60).map { UInt8(0x10 &+ $0) }))
    }

    func testRoundTrip() throws {
        let header = sample()
        let encoded = header.encode()
        let (decoded, bodyOffset) = try KVSnapshotHeader.decode(encoded)
        XCTAssertEqual(decoded, header)
        XCTAssertEqual(bodyOffset, encoded.count, "no trailing bytes after header")
    }

    func testBodyFollowsHeaderAtOffset() throws {
        let header = sample()
        let body = Data("sealed-kv-body".utf8)
        let file = header.encode() + body
        let (decoded, bodyOffset) = try KVSnapshotHeader.decode(file)
        XCTAssertEqual(decoded, header)
        XCTAssertEqual(file[bodyOffset...], body, "body recoverable from the offset")
    }

    func testEnumsSurviveRoundTrip() throws {
        for kek in [KEKType.keyfile, .passphrase, .sep] {
            for reason in [KVSnapshotHeader.SaveReason.cold, .continued, .evict, .shutdown] {
                let header = sample(kekType: kek, saveReason: reason)
                let (decoded, _) = try KVSnapshotHeader.decode(header.encode())
                XCTAssertEqual(decoded.kekType, kek)
                XCTAssertEqual(decoded.saveReason, reason)
            }
        }
    }

    func testBadMagicRejected() {
        var bytes = sample().encode()
        bytes[bytes.startIndex] ^= 0xff
        XCTAssertThrowsError(try KVSnapshotHeader.decode(bytes)) {
            XCTAssertEqual($0 as? KVSnapshotHeader.Failure, .badMagic)
        }
    }

    func testUnsupportedVersionIsSkipOnSkew() {
        var bytes = sample().encode()
        // formatVersion is the u16 right after the 8-byte magic; bump it.
        let vIndex = bytes.index(bytes.startIndex, offsetBy: KVSnapshotHeader.magic.count)
        bytes[vIndex] = 0x99
        XCTAssertThrowsError(try KVSnapshotHeader.decode(bytes)) {
            guard case .unsupportedVersion(let v)? = $0 as? KVSnapshotHeader.Failure else {
                return XCTFail("expected unsupportedVersion, got \($0)")
            }
            XCTAssertNotEqual(v, KVSnapshotHeader.formatVersion)
        }
    }

    func testTruncatedRejected() {
        let full = sample().encode()
        let half = full.prefix(full.count / 2)
        XCTAssertThrowsError(try KVSnapshotHeader.decode(Data(half))) {
            XCTAssertEqual($0 as? KVSnapshotHeader.Failure, .truncated)
        }
    }

    func testBadEnumRejected() {
        var bytes = sample().encode()
        // kekType is the byte right after magic(8) + version(2).
        let kekIndex = bytes.index(bytes.startIndex, offsetBy: KVSnapshotHeader.magic.count + 2)
        bytes[kekIndex] = 0x7f  // not a defined KEKType
        XCTAssertThrowsError(try KVSnapshotHeader.decode(bytes)) {
            XCTAssertEqual($0 as? KVSnapshotHeader.Failure, .badEnum)
        }
    }

    func testIsRestorableGate() {
        let header = sample(modelID: "M", quantTag: "4bit")
        XCTAssertTrue(header.isRestorable(forModel: "M", quant: "4bit"))
        XCTAssertFalse(header.isRestorable(forModel: "OTHER", quant: "4bit"), "model mismatch ⇒ cold")
        XCTAssertFalse(header.isRestorable(forModel: "M", quant: "8bit"), "quant mismatch ⇒ cold")
    }

    func testEmptyStringAndBlobFieldsRoundTrip() throws {
        var header = sample()
        header.scopeKey = ""
        header.prefixHash = Data()
        header.kekParams = Data()
        header.wrappedDEK = Data()
        let (decoded, _) = try KVSnapshotHeader.decode(header.encode())
        XCTAssertEqual(decoded, header)
    }
}
