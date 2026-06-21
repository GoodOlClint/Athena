import Crypto
import Foundation

/// ADR 024 Tier 3 — AES-256-GCM seal/open for **idle prompt-cache KV entries**.
///
/// The M59 prompt-prefix cache parks long-dwell KV entries (reused PHI/PAN
/// prefixes) in RAM for up to the idle-TTL. This cipher lets the pool hold those
/// entries as **ciphertext at rest**, so that:
///   1. only ciphertext is ever swappable / compressible (the strongest,
///      T1-independent reason — see `docs/confidential-kv-cache-plan.md` §1);
///   2. a momentary read (slipped core dump, a debugger that beat
///      `PT_DENY_ATTACH`, a future `task_for_pid` regression) catches only the
///      one entry currently decoding, not the whole idle pool.
///
/// MLX-free + pure swift-crypto, so the round-trip / tamper / key-rotation
/// algebra is unit-pinnable under `swift test` (ADR 008/009). The MLXArray↔bytes
/// conversion lives in `KVByteCodec` (`AthenaLLM`) and is gated.
///
/// ## Honesty boundary (binding, ADR 024)
/// The key is necessarily in our RAM during every bulk seal/open, so a
/// kernel/SIP-off-root adversary reads both the key and the plaintext working
/// set — T3 does **not** defend against that (same as the keychain's in-use
/// secrets). T1 is what blocks the non-root co-resident scraper. This shrinks the
/// plaintext window; it is not a guarantee against a privileged adversary.
///
/// ## Key model (ADR 024 T3, operator-chosen)
/// A random per-process `SymmetricKey(.bits256)`, created lazily on first seal
/// and rotated/zeroed on a full pool flush. **No Secure-Enclave wrapping** — the
/// key is process-ephemeral and never persisted, so SEP would protect nothing it
/// does not already (a RAM-only key gains nothing from SEP against the actual
/// threat). The `currentKeyLocked()` boundary is deliberately the *only* place
/// key material is sourced: a future SEP migration (meaningful only if a key is
/// ever **persisted**, e.g. a disk-backed M59.5 cache) replaces that one function
/// body — `seal`/`open` and the ciphertext format stay identical, so the
/// bit-identical gate is unaffected. See `docs/confidential-kv-cache-plan.md`.
public final class IdleKVCipher: @unchecked Sendable {

    public enum Failure: Error, Equatable {
        /// `SealedBox.combined` was nil (only happens with a non-default nonce
        /// length, which we never use) — defensively surfaced rather than
        /// force-unwrapped.
        case sealProducedNoCombinedBox
    }

    /// Guards the lazily-created key. The cache calls under its own pool lock
    /// already; this keeps the cipher self-contained (uncontended second lock,
    /// consistent ordering, no reverse path) rather than relying on a caller
    /// invariant.
    private let lock = NSLock()
    private var key: SymmetricKey?

    public init() {}

    /// Seal `plaintext` into a combined GCM box (`nonce ‖ ciphertext ‖ tag`).
    /// `aad` is authenticated-but-not-encrypted associated data binding the
    /// slot's context (layer index / slot kind / shape-dtype) so a sealed slot
    /// can't be silently substituted for another; it need not be secret.
    public func seal(_ plaintext: Data, aad: Data) throws -> Data {
        lock.lock()
        defer { lock.unlock() }
        let box = try AES.GCM.seal(
            plaintext, using: currentKeyLocked(), authenticating: aad)
        guard let combined = box.combined else {
            throw Failure.sealProducedNoCombinedBox
        }
        return combined
    }

    /// Open a combined GCM box produced by `seal` under the **current** key and
    /// matching `aad`. Returns `nil` (never throws) on any failure — no key yet,
    /// malformed box, tampered ciphertext/tag, wrong AAD, or a box sealed under a
    /// rotated-away key — so callers treat a failed open uniformly as a miss.
    public func open(_ sealed: Data, aad: Data) -> Data? {
        lock.lock()
        defer { lock.unlock() }
        guard let key else { return nil }
        guard let box = try? AES.GCM.SealedBox(combined: sealed) else {
            return nil
        }
        return try? AES.GCM.open(box, using: key, authenticating: aad)
    }

    /// Rotate (drop) the key. CryptoKit zeroes a `SymmetricKey`'s backing on
    /// dealloc, so releasing the reference wipes the material; the next `seal`
    /// lazily mints a fresh key. Any blob still sealed under the old key becomes
    /// **permanently unopenable**, so this must only be called on a FULL pool
    /// flush, when no live ciphertext outlives the key.
    public func rotate() {
        lock.lock()
        defer { lock.unlock() }
        key = nil
    }

    /// Test/diagnostic: whether a key is currently held (after a `rotate` with no
    /// subsequent `seal`, this is false).
    public var hasKey: Bool {
        lock.lock()
        defer { lock.unlock() }
        return key != nil
    }

    /// The sole key-material source (ADR 024 T3 — the SEP-migration seam). Must
    /// be called with `lock` held.
    private func currentKeyLocked() -> SymmetricKey {
        if let key { return key }
        let fresh = SymmetricKey(size: .bits256)
        key = fresh
        return fresh
    }
}
