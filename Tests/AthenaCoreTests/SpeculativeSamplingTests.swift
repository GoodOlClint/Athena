import Foundation
import XCTest

@testable import AthenaLLM

/// Pure-math coverage for the speculative-SAMPLING helpers (M40.1).
/// CI-safe: no MLX, no model, no I/O. These tests pin the algebra so
/// the wired loop (M40.2/3) only has to be reviewed for plumbing.
final class SpeculativeSamplingTests: XCTestCase {

    // MARK: - SamplingRNG / SplitMix64 determinism

    func testSeededRNGIsReproducible() {
        var a = SamplingRNG(seed: 42)
        var b = SamplingRNG(seed: 42)
        for _ in 0..<128 {
            XCTAssertEqual(a.uniform(), b.uniform(), accuracy: 0)
        }
    }

    func testDifferentSeedsDiverge() {
        var a = SamplingRNG(seed: 1)
        var b = SamplingRNG(seed: 2)
        // Some draw within the first window must differ — uniform
        // RNGs over different seeds shouldn't lockstep.
        var differed = false
        for _ in 0..<32 where a.uniform() != b.uniform() {
            differed = true
        }
        XCTAssertTrue(differed)
    }

    func testUniformDrawsInRange() {
        var rng = SamplingRNG(seed: 7)
        for _ in 0..<1024 {
            let u = rng.uniform()
            XCTAssertGreaterThanOrEqual(u, 0)
            XCTAssertLessThan(u, 1)
        }
    }

    // MARK: - distribution(): temperature, top_p, top_k, greedy degenerate

    func testTemperatureZeroIsOneHotArgmax() {
        let logits: [Float] = [0.1, 3.5, 0.2, 0.4, 3.2]
        let d = SpeculativeSampling.distribution(
            logits: logits, temperature: 0)
        XCTAssertEqual(d.count, 5)
        for (i, v) in d.enumerated() {
            XCTAssertEqual(v, i == 1 ? 1 : 0, accuracy: 0)
        }
    }

    func testSoftmaxIsNumericallyStable() {
        // Very large logits would overflow expf without the max-shift.
        let logits: [Float] = [1000, 1001, 999]
        let d = SpeculativeSampling.distribution(
            logits: logits, temperature: 1)
        XCTAssertEqual(d.reduce(0, +), 1.0, accuracy: 1e-4)
        for v in d { XCTAssertTrue(v.isFinite) }
    }

    func testDistributionSumsToOne() {
        let logits: [Float] = [0.1, -0.3, 2.0, 0.7, -1.2, 0.0]
        let d = SpeculativeSampling.distribution(
            logits: logits, temperature: 0.7)
        XCTAssertEqual(d.reduce(0, +), 1.0, accuracy: 1e-4)
    }

    func testTopPKeepsNucleusAndRenormalizes() {
        // After temperature=1 softmax of [3,2,1,0], top-1 = ~0.6439.
        // top_p = 0.5 should keep just the argmax (cumulative ≥ 0.5
        // on the first token alone), zero the rest, renormalize ⇒
        // one-hot at the argmax.
        let logits: [Float] = [3, 2, 1, 0]
        let d = SpeculativeSampling.distribution(
            logits: logits, temperature: 1, topP: 0.5)
        XCTAssertEqual(d[0], 1.0, accuracy: 1e-5)
        for i in 1..<4 { XCTAssertEqual(d[i], 0, accuracy: 1e-6) }
    }

    func testTopPOneIsInert() {
        let logits: [Float] = [1, 0.5, 0.1, -0.2]
        let a = SpeculativeSampling.distribution(
            logits: logits, temperature: 1, topP: 1.0)
        let b = SpeculativeSampling.distribution(
            logits: logits, temperature: 1, topP: nil)
        for i in 0..<4 {
            XCTAssertEqual(a[i], b[i], accuracy: 1e-6)
        }
    }

    func testTopKKeepsKLargest() {
        let logits: [Float] = [5, 4, 3, 2, 1]
        let d = SpeculativeSampling.distribution(
            logits: logits, temperature: 1, topP: nil, topK: 2)
        XCTAssertGreaterThan(d[0], 0)
        XCTAssertGreaterThan(d[1], 0)
        XCTAssertEqual(d[2], 0, accuracy: 1e-6)
        XCTAssertEqual(d[3], 0, accuracy: 1e-6)
        XCTAssertEqual(d[4], 0, accuracy: 1e-6)
        XCTAssertEqual(d.reduce(0, +), 1, accuracy: 1e-5)
    }

    // MARK: - ratio() — the speculative-sampling core

    func testRatioCappedAtOne() {
        // p(x)/q(x) > 1 ⇒ accept with prob 1 (clamped).
        XCTAssertEqual(
            SpeculativeSampling.ratio(pProb: 0.9, qProb: 0.1), 1,
            accuracy: 1e-6)
    }

    func testRatioFractional() {
        XCTAssertEqual(
            SpeculativeSampling.ratio(pProb: 0.25, qProb: 0.5),
            0.5, accuracy: 1e-6)
    }

    func testRatioQZeroEdgeCase() {
        // Draft sampled a token the target zeroed in truncation.
        XCTAssertEqual(
            SpeculativeSampling.ratio(pProb: 0.3, qProb: 0), 1,
            accuracy: 0)
        XCTAssertEqual(
            SpeculativeSampling.ratio(pProb: 0, qProb: 0), 0,
            accuracy: 0)
    }

    // MARK: - acceptOrReject()

    func testAcceptOrRejectAcceptsWhenRatioIsOne() {
        // Identical p == q ⇒ ratio is exactly 1 for every token ⇒
        // every accept-test passes regardless of u.
        let p: [Float] = [0.2, 0.3, 0.5]
        var rng = SamplingRNG(seed: 99)
        for tok in 0..<3 {
            XCTAssertTrue(
                SpeculativeSampling.acceptOrReject(
                    token: tok, pDistribution: p, qDistribution: p,
                    rng: &rng))
        }
    }

    func testAcceptOrRejectFollowsRatio() {
        // q gives the chosen token 0.8; p gives it 0.2 ⇒ ratio = 0.25.
        // Over a deterministic stream, accept rate must be close to 25%.
        let p: [Float] = [0.2, 0.8]
        let q: [Float] = [0.8, 0.2]
        var rng = SamplingRNG(seed: 12345)
        let trials = 4000
        var accepts = 0
        for _ in 0..<trials {
            if SpeculativeSampling.acceptOrReject(
                token: 0, pDistribution: p, qDistribution: q, rng: &rng)
            {
                accepts += 1
            }
        }
        let rate = Double(accepts) / Double(trials)
        XCTAssertEqual(rate, 0.25, accuracy: 0.03)
    }

    // MARK: - residual()

    func testResidualSubtractsAndRenormalizes() {
        let p: [Float] = [0.5, 0.5]
        let q: [Float] = [0.9, 0.1]
        // max(0, p - q) = [0, 0.4]; normalized ⇒ [0, 1].
        let r = SpeculativeSampling.residual(p: p, q: q)
        XCTAssertEqual(r[0], 0, accuracy: 1e-6)
        XCTAssertEqual(r[1], 1, accuracy: 1e-6)
    }

    func testResidualSumsToOne() {
        let p: [Float] = [0.3, 0.4, 0.3]
        let q: [Float] = [0.4, 0.1, 0.5]
        let r = SpeculativeSampling.residual(p: p, q: q)
        XCTAssertEqual(r.reduce(0, +), 1, accuracy: 1e-5)
        for v in r { XCTAssertGreaterThanOrEqual(v, 0) }
    }

    func testResidualFallsBackToTargetWhenPEqualsQ() {
        // p == q ⇒ residual is identically zero; defensive fallback
        // returns p so the loop still has a valid sample distribution.
        let p: [Float] = [0.25, 0.25, 0.25, 0.25]
        let r = SpeculativeSampling.residual(p: p, q: p)
        for i in 0..<4 { XCTAssertEqual(r[i], 0.25, accuracy: 1e-6) }
    }

    // MARK: - sampleFromDistribution()

    func testSampleHitsExpectedMassFromSeed() {
        // p = [0.1, 0.2, 0.7]: large-N seeded run should land within
        // a few percentage points of the marginal — the stream is
        // deterministic so this assertion is reproducible.
        let p: [Float] = [0.1, 0.2, 0.7]
        var rng = SamplingRNG(seed: 2026_05_26)
        let trials = 8000
        var counts = [0, 0, 0]
        for _ in 0..<trials {
            counts[SpeculativeSampling.sampleFromDistribution(p, rng: &rng)]
                += 1
        }
        XCTAssertEqual(
            Double(counts[0]) / Double(trials), 0.1, accuracy: 0.02)
        XCTAssertEqual(
            Double(counts[1]) / Double(trials), 0.2, accuracy: 0.02)
        XCTAssertEqual(
            Double(counts[2]) / Double(trials), 0.7, accuracy: 0.02)
    }

    func testSampleFromOneHotIsDeterministic() {
        var d = [Float](repeating: 0, count: 16)
        d[7] = 1
        var rng = SamplingRNG(seed: 0)
        for _ in 0..<32 {
            XCTAssertEqual(
                SpeculativeSampling.sampleFromDistribution(d, rng: &rng), 7)
        }
    }

    // MARK: - Greedy reduction (the HARD constraint)

    /// At temperature == 0 the whole loop must reduce to greedy:
    /// distributions are 1-hot at argmax, q ?= p ⇒ ratio is 1 ⇒
    /// accept; q ≠ p ⇒ ratio is 0 ⇒ reject, residual is p's argmax.
    /// This pins the bit-identical greedy contract at the math level.
    func testGreedyReductionAcceptWhenDraftMatchesArgmax() {
        let pLogits: [Float] = [0.1, 4.0, 0.2]
        let qLogits: [Float] = [0.2, 5.0, 0.1]
        let p = SpeculativeSampling.distribution(
            logits: pLogits, temperature: 0)
        let q = SpeculativeSampling.distribution(
            logits: qLogits, temperature: 0)
        // Both argmax at index 1.
        var rng = SamplingRNG(seed: 0)
        let draftToken =
            SpeculativeSampling.sampleFromDistribution(q, rng: &rng)
        XCTAssertEqual(draftToken, 1)
        let accepted = SpeculativeSampling.acceptOrReject(
            token: draftToken, pDistribution: p, qDistribution: q,
            rng: &rng)
        XCTAssertTrue(accepted)
    }

    func testGreedyReductionRejectWhenDraftMismatch() {
        // Different argmaxes — draft picks 0, target picks 2.
        let p = SpeculativeSampling.distribution(
            logits: [0.1, 0.2, 4.0], temperature: 0)
        let q = SpeculativeSampling.distribution(
            logits: [4.0, 0.1, 0.2], temperature: 0)
        var rng = SamplingRNG(seed: 0)
        let draft = SpeculativeSampling.sampleFromDistribution(q, rng: &rng)
        XCTAssertEqual(draft, 0)
        let accepted = SpeculativeSampling.acceptOrReject(
            token: draft, pDistribution: p, qDistribution: q, rng: &rng)
        XCTAssertFalse(accepted)
        // Residual should collapse to target's argmax (1-hot at 2).
        let r = SpeculativeSampling.residual(p: p, q: q)
        XCTAssertEqual(r[2], 1, accuracy: 1e-6)
        for i in [0, 1] {
            XCTAssertEqual(r[i], 0, accuracy: 1e-6)
        }
    }

    // MARK: - C1: stable truncation tie-break (seed reproducibility)

    /// All-equal logits ⇒ every prob ties. top_k must deterministically keep
    /// the LOWEST indices via the index tie-break; Swift's sort is unstable,
    /// so without it the surviving set would vary run-to-run and break
    /// same-seed reproducibility.
    func testTopKTieBreakKeepsLowestIndices() {
        let logits = [Float](repeating: 1.0, count: 6)
        let d = SpeculativeSampling.distribution(
            logits: logits, temperature: 1.0, topP: nil, topK: 2)
        XCTAssertEqual(d[0], 0.5, accuracy: 1e-6)
        XCTAssertEqual(d[1], 0.5, accuracy: 1e-6)
        for i in 2..<6 { XCTAssertEqual(d[i], 0, "index \(i) truncated") }
    }

    /// Same total-order tie-break for the nucleus boundary: p=0.5 over 6
    /// equal probs keeps exactly the three lowest indices.
    func testTopPTieBreakKeepsLowestIndices() {
        let logits = [Float](repeating: 1.0, count: 6)
        let d = SpeculativeSampling.distribution(
            logits: logits, temperature: 1.0, topP: 0.5, topK: nil)
        for i in 0..<3 { XCTAssertEqual(d[i], 1.0 / 3, accuracy: 1e-6) }
        for i in 3..<6 { XCTAssertEqual(d[i], 0, "index \(i) truncated") }
    }
}
