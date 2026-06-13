import Foundation
import XCTest

@testable import AthenaLLM

/// L1 (M70.3) — the most load-bearing invariant: greedy MTP speculative
/// decoding emits a token sequence bit-identical to non-speculative greedy
/// decoding (MTP only changes speed). It was validated only model-on
/// (env-gated). The numeric half (the backbone's masked-argmax equalling a
/// reference) stays gated, but the accept/reject+bonus SEQUENCING that makes
/// the output greedy regardless of the draft is pure algebra: this drives
/// `SpeculativeAcceptance.accepts` — the exact decision the production loop
/// branches on — through a faithful loop model against a deterministic greedy
/// oracle, and proves the emitted sequence equals plain greedy for EVERY draft
/// strategy.
final class SpeculativeAcceptanceTests: XCTestCase {

    /// Deterministic stand-in for the backbone's greedy pick after committing
    /// `p` (a varied, cycle-free map over a small vocab).
    private func greedy(_ p: Int) -> Int { (p * 31 + 7) % 50 }

    /// The reference: plain non-speculative greedy continuation from `p0`.
    private func referenceGreedy(from p0: Int, count: Int) -> [Int] {
        var out: [Int] = []
        var p = p0
        while out.count < count {
            let n = greedy(p)
            out.append(n)
            p = n
        }
        return out
    }

    /// Faithful model of `SpeculativeGeneration`'s draft/verify/accept loop:
    /// every committed token is the backbone's greedy pick at its position;
    /// `SpeculativeAcceptance.accepts` decides accept (commit draft + bonus)
    /// vs reject (commit the verify prediction). `proposer(prev, step)` is the
    /// MTP draft — ANY strategy. Returns the emitted ids plus iteration/accept
    /// tallies.
    private func simulate(
        p0: Int, maxTokens: Int, proposer: (_ prev: Int, _ step: Int) -> Int
    ) -> (out: [Int], iters: Int, accepts: Int) {
        var out: [Int] = []
        var prev = p0
        var step = 0
        var draft = proposer(prev, step)
        var iters = 0
        var accepts = 0
        while out.count < maxTokens {
            let verifyPred = greedy(prev)  // backbone greedy after `prev`
            iters += 1
            if SpeculativeAcceptance.accepts(draft: draft, verifyPred: verifyPred) {
                accepts += 1
                out.append(draft)  // == verifyPred (the greedy token)
                if out.count >= maxTokens { break }
                let bonus = greedy(draft)  // greedy pick at the draft position
                out.append(bonus)
                if out.count >= maxTokens { break }
                prev = bonus
            } else {
                out.append(verifyPred)  // the greedy token; draft discarded
                if out.count >= maxTokens { break }
                prev = verifyPred
            }
            step += 1
            draft = proposer(prev, step)
        }
        return (out, iters, accepts)
    }

    /// A draft that always guesses the greedy token ⇒ every iteration accepts
    /// (~2 tokens/iter), and the output is exactly greedy.
    func testAlwaysCorrectDraftMatchesGreedy() {
        let maxTokens = 40
        let r = simulate(p0: 3, maxTokens: maxTokens) { prev, _ in greedy(prev) }
        XCTAssertEqual(r.out, referenceGreedy(from: 3, count: maxTokens))
        XCTAssertEqual(r.accepts, r.iters, "every draft accepted")
        XCTAssertEqual(r.iters, maxTokens / 2, "2 tokens committed per accept")
    }

    /// A draft that always guesses wrong ⇒ every iteration rejects (1
    /// token/iter), yet the output is the SAME greedy sequence — proving the
    /// draft never changes WHAT is emitted, only the speed.
    func testAlwaysWrongDraftMatchesGreedy() {
        let maxTokens = 40
        let r = simulate(p0: 3, maxTokens: maxTokens) { prev, _ in
            (greedy(prev) + 1) % 50  // guaranteed != greedy(prev)
        }
        XCTAssertEqual(r.out, referenceGreedy(from: 3, count: maxTokens))
        XCTAssertEqual(r.accepts, 0, "every draft rejected")
        XCTAssertEqual(r.iters, maxTokens, "1 token committed per reject")
    }

    /// A mixed/seeded draft strategy ⇒ still exactly greedy. Property over the
    /// thing that actually varies run-to-run in production (draft quality).
    func testMixedDraftMatchesGreedy() {
        let maxTokens = 41  // odd: also exercises truncation mid-accept-pair
        for seed in [1, 7, 42, 2026] {
            var rng = SplitMix64(seed: UInt64(seed))
            let r = simulate(p0: 11, maxTokens: maxTokens) { prev, _ in
                // Half the time propose the greedy token, half a wrong one.
                rng.next() & 1 == 0 ? greedy(prev) : (greedy(prev) + 3) % 50
            }
            XCTAssertEqual(
                r.out, referenceGreedy(from: 11, count: maxTokens),
                "seed \(seed): speculative output must equal greedy")
        }
    }

    /// The seam decision itself: accept iff draft equals the verify
    /// prediction — the pivot the whole parity property rests on.
    func testAcceptsIsEquality() {
        XCTAssertTrue(SpeculativeAcceptance.accepts(draft: 5, verifyPred: 5))
        XCTAssertFalse(SpeculativeAcceptance.accepts(draft: 5, verifyPred: 6))
    }
}
