import Foundation
import XCTest

@testable import AthenaCore

/// ADR 024 Tier 3 — the idle-cache cipher's crypto algebra is MLX-free, so the
/// seal/open round-trip, AAD binding, tamper rejection, and key-rotation
/// semantics are unit-pinned here (ADR 008/009). The MLXArray↔bytes bridge
/// (`KVByteCodec`) and the end-to-end byte-identicality are validated by the
/// host-bound bit-identical gate instead.
final class IdleKVCipherTests: XCTestCase {

    private let aad = Data("attn:7".utf8)

    func testSealOpenRoundTrip() throws {
        let cipher = IdleKVCipher()
        let plaintext = Data((0 ..< 4096).map { UInt8($0 & 0xff) })
        let sealed = try cipher.seal(plaintext, aad: aad)
        XCTAssertNotEqual(sealed, plaintext, "ciphertext must differ from input")
        XCTAssertEqual(cipher.open(sealed, aad: aad), plaintext)
    }

    func testEmptyPlaintextRoundTrips() throws {
        let cipher = IdleKVCipher()
        let sealed = try cipher.seal(Data(), aad: aad)
        XCTAssertEqual(cipher.open(sealed, aad: aad), Data())
    }

    func testTamperedCiphertextFailsOpen() throws {
        let cipher = IdleKVCipher()
        var sealed = try cipher.seal(Data("patient-record".utf8), aad: aad)
        // Flip a byte in the ciphertext body (past the 12-byte nonce).
        sealed[sealed.index(sealed.startIndex, offsetBy: 14)] ^= 0xff
        XCTAssertNil(cipher.open(sealed, aad: aad), "GCM tag must reject tamper")
    }

    func testWrongAADFailsOpen() throws {
        let cipher = IdleKVCipher()
        let sealed = try cipher.seal(Data("pan-4111".utf8), aad: aad)
        XCTAssertNil(
            cipher.open(sealed, aad: Data("attn:8".utf8)),
            "a slot sealed for one context must not open under another")
    }

    func testOpenBeforeAnySealIsNil() {
        // A fresh cipher holds no key; opening anything is a miss, not a crash.
        let cipher = IdleKVCipher()
        XCTAssertFalse(cipher.hasKey)
        XCTAssertNil(cipher.open(Data((0 ..< 60).map { UInt8($0) }), aad: aad))
    }

    func testDistinctNoncePerSeal() throws {
        // Same plaintext + AAD sealed twice yields different ciphertext (random
        // per-seal nonce) but both open to the same plaintext.
        let cipher = IdleKVCipher()
        let plaintext = Data("KV-block".utf8)
        let a = try cipher.seal(plaintext, aad: aad)
        let b = try cipher.seal(plaintext, aad: aad)
        XCTAssertNotEqual(a, b, "nonce reuse — sealing must be randomized")
        XCTAssertEqual(cipher.open(a, aad: aad), plaintext)
        XCTAssertEqual(cipher.open(b, aad: aad), plaintext)
    }

    func testRotateDropsKeyAndInvalidatesPriorBlobs() throws {
        let cipher = IdleKVCipher()
        let plaintext = Data("idle-entry".utf8)
        let sealed = try cipher.seal(plaintext, aad: aad)
        XCTAssertTrue(cipher.hasKey)

        cipher.rotate()
        XCTAssertFalse(cipher.hasKey, "rotate must drop the key")
        XCTAssertNil(
            cipher.open(sealed, aad: aad),
            "no key after rotate ⇒ open is a miss")

        // The next seal mints a fresh key; the OLD blob stays unopenable under it.
        let reSealed = try cipher.seal(plaintext, aad: aad)
        XCTAssertTrue(cipher.hasKey)
        XCTAssertNil(
            cipher.open(sealed, aad: aad),
            "a blob from the rotated-away key must never open under the new key")
        XCTAssertEqual(cipher.open(reSealed, aad: aad), plaintext)
    }
}
