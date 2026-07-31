import XCTest

/// Deterministic waiting for ordering-sensitive async tests (issues #3/#4).
///
/// A fixed `Task.sleep` window is not a scheduling guarantee: on a slow shared
/// runner the state a test asserts on may not have been produced yet, which is
/// exactly the flake class #3 hit in `InferenceGateTests` (CI run 30599158302).
/// Poll the observable state against a MONOTONIC deadline instead — fast when
/// the state lands quickly, generous when the runner is starved, and it fails
/// with one diagnostic instead of hanging.
///
/// Promoted here from `InferenceGateTests` so every suite shares one helper.
extension XCTestCase {

    /// Poll `condition` until true or the (monotonic) deadline passes — then
    /// XCTFail and return false so the caller can bail with one diagnostic.
    ///
    /// `condition` may throw (and the call `rethrows`), so a poll over a
    /// throwing API propagates the REAL error immediately instead of swallowing
    /// it with `try?` and reporting a generic timeout `seconds` later.
    @discardableResult
    func waitUntil(
        _ label: String, seconds: Double = 10,
        file: StaticString = #filePath, line: UInt = #line,
        _ condition: () async throws -> Bool
    ) async rethrows -> Bool {
        let deadline = ContinuousClock.now + .seconds(seconds)
        while !(try await condition()) {
            if ContinuousClock.now > deadline {
                XCTFail(
                    "timed out waiting for: \(label)", file: file, line: line)
                return false
            }
            try? await Task.sleep(for: .milliseconds(1))
        }
        return true
    }
}
