import Foundation
import XCTest

@testable import AthenaLLM

/// M71.1 — the pure image content-part decoder. Passive-oracle: inline `data:`
/// URLs only; `http(s)` and unknown schemes are rejected; malformed/non-image
/// payloads are rejected. No MLX, so this runs in the fast `swift test` tier.
final class ChatImageTests: XCTestCase {
    // A 1x1 transparent PNG, base64.
    private let pngB64 =
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
        + "+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="

    func testDataURLBase64PNGDecodes() throws {
        let img = try ChatImage.fromImageURL(
            "data:image/png;base64,\(pngB64)")
        XCTAssertEqual(img.mediaType, "image/png")
        XCTAssertEqual(img.data, Data(base64Encoded: pngB64))
        XCTAssertFalse(img.data.isEmpty)
    }

    func testMediaTypeWithParamsIsTrimmedToTypeOnly() throws {
        let img = try ChatImage.fromImageURL(
            "data:image/jpeg;charset=utf-8;base64,\(pngB64)")
        XCTAssertEqual(img.mediaType, "image/jpeg")
    }

    func testHTTPURLRejectedAsRemote() {
        XCTAssertThrowsError(
            try ChatImage.fromImageURL("http://example.com/cat.png")
        ) { XCTAssertEqual($0 as? ChatImageError, .remoteURLUnsupported) }
    }

    func testHTTPSURLRejectedAsRemote() {
        XCTAssertThrowsError(
            try ChatImage.fromImageURL("https://example.com/cat.png")
        ) { XCTAssertEqual($0 as? ChatImageError, .remoteURLUnsupported) }
    }

    func testUnknownSchemeRejected() {
        XCTAssertThrowsError(
            try ChatImage.fromImageURL("ftp://host/cat.png")
        ) { XCTAssertEqual($0 as? ChatImageError, .unsupportedScheme) }
        XCTAssertThrowsError(
            try ChatImage.fromImageURL("not-a-url")
        ) { XCTAssertEqual($0 as? ChatImageError, .unsupportedScheme) }
    }

    func testMalformedDataURLNoCommaRejected() {
        XCTAssertThrowsError(
            try ChatImage.fromImageURL("data:image/png;base64")
        ) { XCTAssertEqual($0 as? ChatImageError, .malformedDataURL) }
    }

    func testNonImageMediaTypeRejected() {
        XCTAssertThrowsError(
            try ChatImage.fromImageURL(
                "data:text/plain;base64,aGVsbG8=")
        ) { XCTAssertEqual($0 as? ChatImageError, .malformedDataURL) }
    }

    func testBadBase64Rejected() {
        XCTAssertThrowsError(
            try ChatImage.fromImageURL("data:image/png;base64,!!!notb64!!!")
        ) { XCTAssertEqual($0 as? ChatImageError, .malformedDataURL) }
    }

    func testEmptyPayloadRejected() {
        XCTAssertThrowsError(
            try ChatImage.fromImageURL("data:image/png;base64,")
        ) { XCTAssertEqual($0 as? ChatImageError, .malformedDataURL) }
    }

    func testChatTurnDefaultsToNoImages() {
        // Backward compat: the text-only initializer is unchanged.
        let t = ChatTurn(role: "user", content: "hi")
        XCTAssertTrue(t.images.isEmpty)
    }
}
