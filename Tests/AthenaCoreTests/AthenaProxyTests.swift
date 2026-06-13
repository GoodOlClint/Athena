import Foundation
import XCTest

@testable import AthenaCore

/// M13.2 — proxy URL parsing + bypass list. Pure, env-independent
/// (no `*_PROXY` env touched here; that path is covered by e2e).
final class AthenaProxyTests: XCTestCase {

    func testParsesSchemeHostPort() {
        let p = AthenaProxy.parse("http://proxy.corp:3128")
        XCTAssertEqual(p?.host, "proxy.corp")
        XCTAssertEqual(p?.port, 3128)
        XCTAssertNil(p?.user)
        XCTAssertEqual(p?.isSOCKS, false)
    }

    func testParsesInlineCredentials() {
        let p = AthenaProxy.parse("https://u%40x:p%3As@gw:8443")
        XCTAssertEqual(p?.host, "gw")
        XCTAssertEqual(p?.port, 8443)
        XCTAssertEqual(p?.user, "u@x")  // percent-decoded
        XCTAssertEqual(p?.pass, "p:s")
    }

    func testSocksDefaultPortAndFlag() {
        let p = AthenaProxy.parse("socks5://s")
        XCTAssertEqual(p?.isSOCKS, true)
        XCTAssertEqual(p?.port, 1080)
    }

    func testBareHostPortAssumesHTTP() {
        let p = AthenaProxy.parse("10.0.0.9:3128")
        XCTAssertEqual(p?.host, "10.0.0.9")
        XCTAssertEqual(p?.port, 3128)
        XCTAssertEqual(p?.isSOCKS, false)
    }

    func testRejectsEmptyAndOutOfRangePort() {
        XCTAssertNil(AthenaProxy.parse(""))
        XCTAssertNil(AthenaProxy.parse("   "))
        XCTAssertNil(AthenaProxy.parse("http://h:99999"))
    }

    func testBypassAlwaysIncludesLoopback() {
        let b = AthenaProxy.bypassList()
        XCTAssertTrue(b.contains("127.0.0.1"))
        XCTAssertTrue(b.contains("::1"))
        XCTAssertTrue(b.contains("localhost"))
    }

    // MARK: - M70.3 NE8 — describe() redacts the proxy password

    /// `describe` is the `proxy status` / `doctor` one-liner; it must NEVER
    /// surface the password. Tested via the extracted pure formatter so no
    /// `*_PROXY` env is touched (the redaction logic is identical to the
    /// env-reading `describe()`).
    func testDescribeRedactsPassword() throws {
        let authed = try XCTUnwrap(
            AthenaProxy.parse("https://alice:s3cr3t@gw.corp:8443"))
        let line = AthenaProxy.describe(authed)
        XCTAssertTrue(line.contains("gw.corp"))
        XCTAssertTrue(line.contains("8443"))
        XCTAssertTrue(
            line.contains("(auth: alice:***)"), "username shown, secret masked")
        XCTAssertFalse(
            line.contains("s3cr3t"), "the password must NEVER appear")

        // A no-auth proxy carries no auth suffix at all.
        let bare = try XCTUnwrap(AthenaProxy.parse("http://proxy:3128"))
        let bareLine = AthenaProxy.describe(bare)
        XCTAssertFalse(bareLine.contains("auth:"))
        XCTAssertEqual(bareLine, "http(s)://proxy:3128")
    }
}
