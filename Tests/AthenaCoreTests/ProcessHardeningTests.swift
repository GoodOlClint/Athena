#if canImport(Darwin)
import Darwin
#endif
import Foundation
import XCTest

@testable import AthenaCore

/// ADR 024 Tier 2 unit coverage. `secureZero` and the core-dump policy are pure
/// libsystem decisions (MLX-free, ADR 008/009). `denyDebuggerAttachNow` is NOT
/// invoked here — it has a process-global side effect (and SIGKILLs on a forced
/// attach); it is validated by the daemon's startup log line in e2e.
final class ProcessHardeningTests: XCTestCase {

    func testSecureZeroFloatBuffer() {
        var buf: [Float] = [1, -2, 3.5, 4, -5]
        ProcessHardening.secureZero(&buf)
        XCTAssertTrue(buf.allSatisfy { $0 == 0 }, "buffer not zeroed")
    }

    func testSecureZeroEmptyIsSafe() {
        var empty: [Float] = []
        ProcessHardening.secureZero(&empty)  // must not crash
        XCTAssertTrue(empty.isEmpty)
    }

    func testSecureZeroData() {
        var d = Data([0xde, 0xad, 0xbe, 0xef, 0x01, 0x02])
        ProcessHardening.secureZero(&d)
        XCTAssertEqual(Array(d), [0, 0, 0, 0, 0, 0])
    }

    func testDisableCoreDumpsSetsLimitToZero() {
        XCTAssertTrue(ProcessHardening.disableCoreDumps())
        #if canImport(Darwin)
        var lim = rlimit()
        XCTAssertEqual(getrlimit(RLIMIT_CORE, &lim), 0)
        XCTAssertEqual(lim.rlim_cur, 0, "RLIMIT_CORE soft limit should be 0")
        #endif
    }
}
