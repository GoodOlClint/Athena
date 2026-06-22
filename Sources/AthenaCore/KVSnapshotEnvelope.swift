import Crypto
import Foundation

/// ADR 027 §3 / ADR 024 amendment — the **DEK/KEK envelope** for a persisted KV
/// snapshot, and the **swappable KEK** seam that keeps the persisted blob from
/// being hardware-locked-forever.
///
/// A blob is never encrypted directly under a persisted key (the
/// "locked-forever" trap). Instead a random **per-blob data key (DEK)**
/// AES-256-GCM-seals the body, and a **key-encrypting key (KEK)** wraps the DEK;
/// the wrapped DEK rides in the header (`KVSnapshotHeader.wrappedDEK`). Swapping
/// the KEK — `keyfile`/`passphrase` now, SEP-bound later, per-peer ECIES for a
/// future cluster — is a `KEKProvider` swap, not a reformat.
///
/// MLX-free + pure swift-crypto, so the wrap/unwrap/seal/open algebra is
/// unit-pinnable under `swift test` (ADR 008/009).

/// Which kind of KEK wrapped a blob's DEK — recorded in the header so the restore
/// path picks the matching `KEKProvider`, and so the at-rest posture is
/// self-describing (a `keyfile` blob never claims SEP-grade protection).
public enum KEKType: UInt8, Equatable, Sendable {
    case keyfile = 1
    case passphrase = 2   // S2 (PBKDF2 over a passphrase) — not yet provided
    case sep = 3          // S6 (Secure Enclave P-256) — stubbed below
}

public enum KVSnapshotCryptoError: Error, Equatable {
    /// `SealedBox.combined` was nil (only with a non-default nonce length, which
    /// we never use) — surfaced rather than force-unwrapped.
    case sealProducedNoCombinedBox
    /// A keyfile shorter than 32 bytes is rejected — too little entropy to be a
    /// key-encrypting key.
    case keyfileTooShort(Int)
    /// The SEP-backed KEK is the S6 follow-up (gated on the headless-SEP
    /// operability spike, ADR 027 plan); the seam exists but the body does not.
    case sepNotYetImplemented
}

/// The swappable key-encrypting-key. Wrapping yields opaque `params` (stored in
/// the header) plus the wrapped DEK; unwrapping reverses it and returns `nil`
/// (never throws) on any failure, so the restore path treats a bad key uniformly
/// as "skip → go cold".
public protocol KEKProvider: Sendable {
    var kekType: KEKType { get }
    func wrap(_ dek: SymmetricKey) throws -> (params: Data, wrapped: Data)
    func unwrap(params: Data, wrapped: Data) -> SymmetricKey?
}

/// KEK derived from a high-entropy operator **keyfile** (≥32 random bytes). Each
/// wrap mints a fresh HKDF salt (the `params`) so wraps are not deterministic;
/// the wrapping key is `HKDF<SHA256>(keyfile, salt)`, and the DEK is sealed under
/// it with AES-256-GCM. Cross-restart works immediately and the blob is
/// encrypted-at-rest — but the key lives on the host (operator-managed), which
/// the header's `kekType` states honestly (ADR 027 honesty boundary).
public struct KeyfileKEK: KEKProvider {

    public let kekType: KEKType = .keyfile

    private let keyMaterial: SymmetricKey
    private static let saltLen = 32
    private static let info = Data("athena.kv-snapshot.kek.v1".utf8)

    /// `keyfile` must be ≥32 bytes of high-entropy material.
    public init(keyfile: Data) throws {
        guard keyfile.count >= 32 else {
            throw KVSnapshotCryptoError.keyfileTooShort(keyfile.count)
        }
        self.keyMaterial = SymmetricKey(data: keyfile)
    }

    public func wrap(_ dek: SymmetricKey) throws -> (params: Data, wrapped: Data) {
        let salt = Self.randomBytes(Self.saltLen)
        let box = try AES.GCM.seal(dek.rawBytes, using: deriveWrapKey(salt: salt))
        guard let combined = box.combined else {
            throw KVSnapshotCryptoError.sealProducedNoCombinedBox
        }
        return (params: salt, wrapped: combined)
    }

    public func unwrap(params salt: Data, wrapped: Data) -> SymmetricKey? {
        guard salt.count == Self.saltLen else { return nil }
        guard let box = try? AES.GCM.SealedBox(combined: wrapped),
            let dekBytes = try? AES.GCM.open(box, using: deriveWrapKey(salt: salt))
        else { return nil }
        return SymmetricKey(data: dekBytes)
    }

    /// The wrapping key for a given salt: `HKDF<SHA256>(keyfile, salt, info)`.
    private func deriveWrapKey(salt: Data) -> SymmetricKey {
        HKDF<SHA256>.deriveKey(
            inputKeyMaterial: keyMaterial, salt: salt, info: Self.info,
            outputByteCount: 32)
    }

    private static func randomBytes(_ count: Int) -> Data {
        var rng = SystemRandomNumberGenerator()
        return Data((0 ..< count).map { _ in UInt8.random(in: 0 ... 255, using: &rng) })
    }
}

/// SEP-backed KEK — the S6 follow-up (Secure Enclave P-256, ECIES wrap),
/// **gated on the headless-launchd SEP operability spike** (ADR 027 plan). The
/// seam conforms now so `kekType == .sep` round-trips through the header and the
/// store dispatches on it; the wrap/unwrap bodies arrive in S6.
public struct SepKEK: KEKProvider {
    public let kekType: KEKType = .sep
    public init() {}
    public func wrap(_ dek: SymmetricKey) throws -> (params: Data, wrapped: Data) {
        throw KVSnapshotCryptoError.sepNotYetImplemented
    }
    public func unwrap(params: Data, wrapped: Data) -> SymmetricKey? { nil }
}

/// The envelope operation: seal `plaintext` (the `KVFrame` body bytes) under a
/// fresh per-blob DEK, wrapping that DEK with `kek`; and the inverse. `aad` binds
/// the body to its snapshot identity (the store passes the model/quant/prefix
/// tuple) so a body can't be paired with a different header.
public enum KVSnapshotEnvelope {

    /// The encrypted product: the sealed body plus the header-bound KEK fields.
    public struct Sealed: Equatable, Sendable {
        public let body: Data
        public let kekParams: Data
        public let wrappedDEK: Data
        public init(body: Data, kekParams: Data, wrappedDEK: Data) {
            self.body = body
            self.kekParams = kekParams
            self.wrappedDEK = wrappedDEK
        }
    }

    public static func seal(
        _ plaintext: Data, aad: Data = Data(), kek: KEKProvider
    ) throws -> Sealed {
        let dek = SymmetricKey(size: .bits256)
        let bodyBox = try AES.GCM.seal(plaintext, using: dek, authenticating: aad)
        guard let body = bodyBox.combined else {
            throw KVSnapshotCryptoError.sealProducedNoCombinedBox
        }
        let (params, wrapped) = try kek.wrap(dek)
        return Sealed(body: body, kekParams: params, wrappedDEK: wrapped)
    }

    /// Recover the plaintext, or `nil` on any failure (wrong/rotated KEK, tampered
    /// body or wrapped-DEK, wrong `aad`) — a uniform "skip → go cold" signal.
    public static func open(
        _ sealed: Sealed, aad: Data = Data(), kek: KEKProvider
    ) -> Data? {
        guard let dek = kek.unwrap(params: sealed.kekParams, wrapped: sealed.wrappedDEK),
            let box = try? AES.GCM.SealedBox(combined: sealed.body),
            let plaintext = try? AES.GCM.open(box, using: dek, authenticating: aad)
        else { return nil }
        return plaintext
    }
}

extension SymmetricKey {
    /// The key's raw bytes as `Data` (for wrapping the DEK under a KEK).
    fileprivate var rawBytes: Data {
        withUnsafeBytes { Data($0) }
    }
}
