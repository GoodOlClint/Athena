import Foundation

/// ADR 017 — inbound upload-size decision algebra. The byte caps are
/// operator config (`max_audio_upload_bytes` / `max_request_body_bytes`);
/// this is the pure, MLX-free logic over a request's declared
/// `Content-Length` and a cap, kept here so it is unit-testable under
/// `swift test` (ADR 008/009) rather than buried in the MLX-linked daemon.
///
/// Two enforcement points use it: an up-front `Content-Length` check (this
/// type) that fast-fails an over-cap upload before the body is read, and a
/// streamed `collect(upTo: cap)` backstop in the handler for chunked /
/// absent / understated `Content-Length`. Both surface the same clean
/// `413` — never an internal NIO error type.
public enum UploadLimit {
    public enum Decision: Equatable, Sendable {
        /// The declared length is within cap (or absent/unparseable — the
        /// streamed backstop then enforces the cap as the body arrives).
        case proceed
        /// `Content-Length` already exceeds the cap ⇒ reject 413 up front.
        case rejectTooLarge
    }

    /// Up-front decision from a parsed `Content-Length` (nil ⇒ header
    /// absent or unparseable). Absent length never rejects here — a lying
    /// or chunked client is caught by the streamed backstop instead.
    public static func check(contentLength: Int?, cap: Int) -> Decision {
        guard let n = contentLength, n > cap else { return .proceed }
        return .rejectTooLarge
    }

    /// The `413` body message. States the cap in bytes; deliberately
    /// contains no internal type names (the old path leaked
    /// `NIOTooManyBytesError(...)`).
    public static func tooLargeMessage(cap: Int) -> String {
        "request body exceeds the \(cap)-byte upload limit"
    }
}
