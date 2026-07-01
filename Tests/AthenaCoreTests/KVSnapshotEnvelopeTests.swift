import Crypto
import Foundation
import XCTest

@testable import AthenaCore

/// ADR 027 / ADR 024 amendment — the DEK/KEK envelope is MLX-free swift-crypto,
/// so the wrap/unwrap, seal/open, AAD binding, tamper, and wrong-key rejection
/// paths are unit-pinned here (ADR 008/009).
final class KVSnapshotEnvelopeTests: XCTestCase {

    private func keyfile(_ seed: UInt8 = 7) -> Data {
        Data((0 ..< 32).map { UInt8($0) &+ seed })
    }
    private let body = Data((0 ..< 4096).map { UInt8($0 & 0xff) })

    // MARK: - KeyfileKEK

    func testKeyfileKEKWrapUnwrapRoundTrip() throws {
        let kek = try KeyfileKEK(keyfile: keyfile())
        let dek = SymmetricKey(size: .bits256)
        let (params, wrapped) = try kek.wrap(dek)
        let recovered = kek.unwrap(params: params, wrapped: wrapped)
        XCTAssertEqual(recovered?.rawBytes, dek.rawBytes, "unwrap must recover the DEK")
    }

    func testKeyfileKEKFreshSaltPerWrap() throws {
        let kek = try KeyfileKEK(keyfile: keyfile())
        let dek = SymmetricKey(size: .bits256)
        let a = try kek.wrap(dek)
        let b = try kek.wrap(dek)
        XCTAssertNotEqual(a.params, b.params, "each wrap must mint a fresh HKDF salt")
        XCTAssertNotEqual(a.wrapped, b.wrapped, "fresh salt ⇒ different wrapped bytes")
    }

    func testKeyfileTooShortRejected() {
        XCTAssertThrowsError(try KeyfileKEK(keyfile: Data(repeating: 0, count: 31))) {
            XCTAssertEqual($0 as? KVSnapshotCryptoError, .keyfileTooShort(31))
        }
    }

    func testWrongKeyfileFailsUnwrap() throws {
        let kek = try KeyfileKEK(keyfile: keyfile(1))
        let (params, wrapped) = try kek.wrap(SymmetricKey(size: .bits256))
        let other = try KeyfileKEK(keyfile: keyfile(2))
        XCTAssertNil(other.unwrap(params: params, wrapped: wrapped), "different keyfile ⇒ miss")
    }

    func testTamperedWrappedDEKFailsUnwrap() throws {
        let kek = try KeyfileKEK(keyfile: keyfile())
        var (params, wrapped) = try kek.wrap(SymmetricKey(size: .bits256))
        wrapped[wrapped.index(wrapped.startIndex, offsetBy: 14)] ^= 0xff
        XCTAssertNil(kek.unwrap(params: params, wrapped: wrapped), "GCM tag rejects tamper")
        _ = params
    }

    // MARK: - Envelope seal/open

    func testEnvelopeSealOpenRoundTrip() throws {
        let kek = try KeyfileKEK(keyfile: keyfile())
        let sealed = try KVSnapshotEnvelope.seal(body, kek: kek)
        XCTAssertNotEqual(sealed.body, body, "body must be ciphertext")
        XCTAssertEqual(KVSnapshotEnvelope.open(sealed, kek: kek), body)
    }

    func testEnvelopeAADBindsIdentity() throws {
        let kek = try KeyfileKEK(keyfile: keyfile())
        let aad = Data("Qwen/X\u{1}4bit\u{1}<prefix-hash>".utf8)
        let sealed = try KVSnapshotEnvelope.seal(body, aad: aad, kek: kek)
        XCTAssertEqual(KVSnapshotEnvelope.open(sealed, aad: aad, kek: kek), body)
        XCTAssertNil(
            KVSnapshotEnvelope.open(sealed, aad: Data("other".utf8), kek: kek),
            "a body bound to one identity must not open under another")
    }

    func testEnvelopeWrongKEKFailsOpen() throws {
        let sealed = try KVSnapshotEnvelope.seal(body, kek: try KeyfileKEK(keyfile: keyfile(1)))
        XCTAssertNil(
            KVSnapshotEnvelope.open(sealed, kek: try KeyfileKEK(keyfile: keyfile(2))),
            "wrong KEK can't unwrap the DEK ⇒ skip → cold")
    }

    func testEnvelopeTamperedBodyFailsOpen() throws {
        let kek = try KeyfileKEK(keyfile: keyfile())
        var sealed = try KVSnapshotEnvelope.seal(body, kek: kek)
        var b = sealed.body
        b[b.index(b.startIndex, offsetBy: 20)] ^= 0xff
        sealed = KVSnapshotEnvelope.Sealed(
            body: b, kekParams: sealed.kekParams, wrappedDEK: sealed.wrappedDEK)
        XCTAssertNil(KVSnapshotEnvelope.open(sealed, kek: kek), "tampered body ⇒ nil")
    }

    func testEmptyBodyRoundTrips() throws {
        let kek = try KeyfileKEK(keyfile: keyfile())
        let sealed = try KVSnapshotEnvelope.seal(Data(), kek: kek)
        XCTAssertEqual(KVSnapshotEnvelope.open(sealed, kek: kek), Data())
    }
}

extension SymmetricKey {
    /// Test helper: raw key bytes for comparison (the production `rawBytes` is
    /// fileprivate to the envelope source).
    fileprivate var rawBytes: Data { withUnsafeBytes { Data($0) } }
}
