import XCTest

/// ONE thread-safe box for the whole suite (#70).
///
/// Six hand-rolled `final class X: @unchecked Sendable` types wrapping an
/// `NSLock` around a `Bool`, an `Int` or an array had accumulated across
/// `AthenaCoreTests` — each correct, but six chances for one to acquire a
/// subtly different memory-ordering property, in helpers whose entire job is
/// making concurrency tests trustworthy. Same precedent as the `GuidedMask`
/// extraction (`Sources/AthenaLLM/GuidedMask.swift`, 52e15554): two copies of
/// a bit-unpacking loop "could drift", so they became one.
///
/// Generic rather than a `Flag` + a `Counter` + a `Collector`, because one
/// type covering all three shapes is less to keep consistent than three.
///
/// `Value: Sendable` is load-bearing, not decoration: over an unconstrained
/// generic this type would claim `Sendable` while `current` handed out an
/// unprotected REFERENCE for any class `Value`, and `inout` in `mutate` would
/// be equally useless. The six specialised copies were structurally immune to
/// that; a shared box is the thing everyone reaches for next, so it has to
/// refuse the unsound case rather than merely not be used unsoundly yet.
///
/// `NSLock`, not `Synchronization.Mutex`: the package floor is macOS 14
/// (`Package.swift`) and `Mutex` requires 15. Same primitive the six copies
/// used, so this is a consolidation and not a change of locking semantics —
/// including its non-recursive-ness. See `mutate`.
final class Locked<Value: Sendable>: @unchecked Sendable {
    private let lock = NSLock()
    private var value: Value

    init(_ value: Value) { self.value = value }

    /// Read a snapshot. The lock is released before the caller sees it, so
    /// this is a point-in-time copy — never read twice expecting agreement.
    var current: Value {
        lock.lock()
        defer { lock.unlock() }
        return value
    }

    /// Read-modify-write under one acquisition. Takes `inout` so
    /// increment/append cannot be split into a racing read and write — the
    /// exact defect `MemoryGovernorTests.Counter` was introduced to fix.
    ///
    /// `body` RUNS UNDER THE LOCK, and `NSLock` is not recursive: touching
    /// this same box from inside the closure — `box.mutate { _ in box.current }`
    /// — self-deadlocks. This is the one surface the six fixed-body copies did
    /// not have, so it is the one thing consolidating made worse rather than
    /// better. Keep bodies to the mutation itself.
    @discardableResult
    func mutate<R>(_ body: (inout Value) -> R) -> R {
        lock.lock()
        defer { lock.unlock() }
        return body(&value)
    }
}

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
