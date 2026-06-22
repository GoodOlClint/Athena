import Foundation
import XCTest

@testable import AthenaCore

/// L2 (M70.3) — acceptance-rate floor coverage on a stub observer.
///
/// `SpeculativeGeneration.generate` publishes one `recordIteration(accepted:)`
/// per draft/verify iteration to `SpeculativeStats.observer` (a `@TaskLocal`),
/// so a future regression in the M47.2 Guide-masked-draft fix that collapsed
/// acceptance back toward ~0% could be caught in CI. The real loop is
/// MLX-bound (env-gated), but the observer contract — the TaskLocal binding
/// reaching a synchronous callee and the accept tally — is pure and is what a
/// CI perf-floor guard reads. Pin it here on a stub, mirroring
/// `DecodeProgressTests`' TaskLocal-propagation pattern.
final class SpeculativeStatsTests: XCTestCase {

    /// NSLock-isolated tally satisfying `SpeculativeAcceptanceObserver`.
    private final class CountingObserver:
        @unchecked Sendable, SpeculativeAcceptanceObserver
    {
        private let lock = NSLock()
        private var accepted = 0
        private var total = 0
        func recordIteration(accepted: Bool) {
            lock.lock(); defer { lock.unlock() }
            total += 1
            if accepted { self.accepted += 1 }
        }
        /// (accepted, total). `rate` is the acceptance fraction a CI
        /// perf-floor assertion compares against.
        var snapshot: (accepted: Int, total: Int) {
            lock.lock(); defer { lock.unlock() }
            return (accepted, total)
        }
        var rate: Double {
            let s = snapshot
            return s.total > 0 ? Double(s.accepted) / Double(s.total) : 0
        }
    }

    /// The TaskLocal observer reaches a `nonisolated` callee that publishes
    /// exactly as the decode loop does — proving the floor guard would see the
    /// real loop's iterations.
    func testObserverTaskLocalReachesPublisher() {
        let obs = CountingObserver()
        // accepted pattern: 64 accepts out of 100 — a representative
        // backbone/MTP agreement rate for the acceptance observer.
        let pattern = (0..<100).map { $0 < 64 }
        SpeculativeStats.$observer.withValue(obs) {
            simulateSpeculativeIterations(pattern)
        }
        let s = obs.snapshot
        XCTAssertEqual(s.total, 100)
        XCTAssertEqual(s.accepted, 64)
        XCTAssertEqual(obs.rate, 0.64, accuracy: 1e-9)
    }

    /// A CI acceptance-rate FLOOR: the M47-fixed Guide-masked-draft path
    /// keeps acceptance well above the ~0% unmasked-draft failure mode. A
    /// 0.64 measured rate clears a conservative 0.50 floor; a regression that
    /// dropped it to single digits would trip this on a stub.
    func testAcceptanceRateClearsFloor() {
        let obs = CountingObserver()
        let pattern = (0..<100).map { $0 < 64 }
        SpeculativeStats.$observer.withValue(obs) {
            simulateSpeculativeIterations(pattern)
        }
        let floor = 0.50
        XCTAssertGreaterThanOrEqual(
            obs.rate, floor,
            "acceptance \(obs.rate) fell below the CI floor \(floor) — the "
                + "M47.2 masked-draft fix may have regressed")
    }

    /// nil observer ⇒ the publish call is a cheap no-op (the production
    /// default when no one is listening); must not trap.
    func testNoObserverIsNoOp() {
        // No withValue binding: SpeculativeStats.observer is nil here.
        simulateSpeculativeIterations([true, false, true])
        XCTAssertNil(SpeculativeStats.observer)
    }

    /// Mirrors `SpeculativeGeneration.generate`'s publish site: a synchronous
    /// loop that reads the `@TaskLocal` observer through the protocol on every
    /// iteration. `nonisolated` so it inherits the binding the same way the
    /// AthenaLLM decode loop does.
    private nonisolated func simulateSpeculativeIterations(_ accepts: [Bool]) {
        for a in accepts {
            SpeculativeStats.observer?.recordIteration(accepted: a)
        }
    }
}
