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
}
