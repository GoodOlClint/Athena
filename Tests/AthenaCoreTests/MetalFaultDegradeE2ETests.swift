import MLX
import XCTest

@testable import AthenaCore

/// ADR 030 Part 2 (WP2) — heavy Metal test: trigger a GENUINE `[metal::malloc]`
/// "maximum allowed buffer size" device-cap fault and prove the daemon degrades
/// it to a 503 instead of aborting, then keeps computing.
///
/// Gated behind `ATHENA_RUN_MODEL_TESTS=1` — it needs a real Metal device (the
/// stub CI tier has no metallib). Run under a `deploy/build.sh`/`xcodebuild`
/// binary on an Apple-Silicon host:
///
///     ATHENA_RUN_MODEL_TESTS=1 ./deploy/test.sh --filter MetalFaultDegradeE2E
///
/// Motivation: the unit tests pin the latch/needle/gate algebra MLX-free; this
/// pins the one thing they can't — that returning from the global error handler
/// (the same mechanism mlx-swift's own `withError` uses) actually keeps the
/// process alive after a rejected device-cap allocation, and that a subsequent
/// eval computes correctly.
final class MetalFaultDegradeE2ETests: XCTestCase {

    func testDeviceCapFaultDegradesTo503AndSurvivesWP2() async throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["ATHENA_RUN_MODEL_TESTS"] == "1",
            "heavy: needs a real Metal device")

        InferenceGate.enabled = true
        MetalFaultDegrade.enabled = true
        MetalFaultLatch.shared.clear()

        // Install the SAME global handler the daemon installs (Load.swift):
        // record a recognized allocation fault and return, instead of aborting.
        MLX.setErrorHandler({ message, _ in
            let m = message.map { String(cString: $0) } ?? "(no message)"
            if AthenaError.isMetalOOMMessage(m) {
                MetalFaultLatch.shared.record(m)
                return
            }
            fatalError("unexpected (non-OOM) MLX fault in test: \(m)")
        })
        defer { MLX.setErrorHandler(nil) }

        // A single fp16 buffer of ~100 GiB — larger than this device's max
        // buffer length (80.64 GiB on M5 Max) — is a hard `[metal::malloc]`
        // rejection, the exact WP2 fault class. Two dims keep each < Int32 max.
        do {
            _ = try await InferenceGate.shared.withExclusiveExecution {
                () -> Int in
                let big = zeros([200_000, 250_000], dtype: .float16)  // ~93 GiB
                asyncEval(big)  // fault fires on MLX's worker thread → handler
                // Wait for the async fault to reach the handler (≤ ~5 s).
                for _ in 0 ..< 200 where !MetalFaultLatch.shared.isSet {
                    try? await Task.sleep(nanoseconds: 25_000_000)
                }
                return 0
            }
            XCTFail("expected the device-cap fault to degrade to a 503")
        } catch let e as AthenaError {
            XCTAssertEqual(e.code, "metal_oom", "must classify as 503 metal_oom")
            XCTAssertEqual(e.httpStatus, 503)
        }

        // Daemon-survivable: the allocator is intact after a REJECTED malloc, so
        // a normal small computation still evaluates correctly.
        let a = MLXArray([1, 2, 3] as [Float])
        let s = (a + a).sum()
        eval(s)
        XCTAssertEqual(s.item(Float.self), 12, accuracy: 1e-4)
    }
}
