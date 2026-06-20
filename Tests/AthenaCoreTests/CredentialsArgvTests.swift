import Foundation
import XCTest

@testable import AthenaClient

/// K7 regression — the Keychain writer must never place a secret on the
/// `security` command line (argv is visible to any local `ps`). The
/// password is fed on stdin via `-w` at end of command; these pin the
/// invariant on the pure argv builders so a future edit can't reintroduce
/// the `-w <value>` footgun ADR 005 set out to kill.
#if os(macOS)
    final class CredentialsArgvTests: XCTestCase {
        private let secret =
            "sup3r-s3cret-bearer-key-must-not-appear-in-argv"

        func testStoreArgvCarriesNoSecret() {
            let args = Secrets.addPasswordArgs(account: "host:7447")
            XCTAssertFalse(
                args.contains(secret),
                "secret must never be an argv element")
            // `-w` at end (no trailing value) ⇒ prompt-on-stdin form.
            XCTAssertEqual(args.last, "-w")
            // No argv token should equal the value for ANY secret —
            // the builder doesn't take the value at all.
            for arg in args {
                XCTAssertNotEqual(arg, secret)
            }
        }

        func testSudoStoreArgvCarriesNoSecret() {
            let args = Secrets.sudoAddPasswordArgs(
                user: "operator", account: "host:7447")
            XCTAssertFalse(args.contains(secret))
            XCTAssertEqual(args.last, "-w")
            // Sanity: it still routes through security as the dropped user.
            XCTAssertEqual(args.first, "-u")
            XCTAssertTrue(args.contains("/usr/bin/security"))
        }

        func testStdinPayloadIsDoubleEntered() {
            // `security … -w` prompts twice (enter + retype); the value
            // is delivered here, off-argv.
            let payload = Secrets.stdinPasswordPayload(secret)
            XCTAssertEqual(
                payload, Data("\(secret)\n\(secret)\n".utf8))
        }
    }
#endif
