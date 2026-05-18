import CommonCrypto
import Foundation

/// WebUI account password hashing (M12.4). Passwords are low-entropy
/// → a slow salted KDF (PBKDF2-HMAC-SHA256), unlike the 256-bit API
/// tokens (plain SHA-256 is fine there). CommonCrypto = macOS system,
/// zero new dependency. Only salt+hash+iters are persisted (SQLite);
/// the password is never stored.
enum Passwords {
    /// OWASP-floor iteration count for PBKDF2-HMAC-SHA256 (2023+).
    static let defaultIterations = 210_000
    static let saltLen = 16
    static let hashLen = 32

    static func randomSalt() -> Data {
        var b = Data(count: saltLen)
        _ = b.withUnsafeMutableBytes {
            SecRandomCopyBytes(
                kSecRandomDefault, saltLen, $0.baseAddress!)
        }
        return b
    }

    static func derive(
        password: String, salt: Data, iters: Int
    ) -> Data {
        let pw = Array(password.utf8)
        var out = Data(count: hashLen)
        let rc = out.withUnsafeMutableBytes { o in
            salt.withUnsafeBytes { s in
                CCKeyDerivationPBKDF(
                    CCPBKDFAlgorithm(kCCPBKDF2),
                    pw.map { CChar(bitPattern: $0) }, pw.count,
                    s.bindMemory(to: UInt8.self).baseAddress,
                    salt.count,
                    CCPseudoRandomAlgorithm(kCCPRFHmacAlgSHA256),
                    UInt32(iters),
                    o.bindMemory(to: UInt8.self).baseAddress,
                    hashLen)
            }
        }
        precondition(Int(rc) == kCCSuccess, "PBKDF2 failed")
        return out
    }

    /// Constant-time verify against a stored salt/hash/iters.
    static func verify(
        password: String, salt: Data, hash: Data, iters: Int
    ) -> Bool {
        let got = Array(derive(
            password: password, salt: salt, iters: iters))
        return AuthConfig.constantTimeEqual(got, Array(hash))
    }
}
