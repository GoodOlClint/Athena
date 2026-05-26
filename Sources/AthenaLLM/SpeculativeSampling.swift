import Foundation

// Pure-math helpers for speculative SAMPLING (Leviathan et al. 2023 /
// Chen et al. 2023, "Accelerating Large Language Model Decoding with
// Speculative Sampling"). The wiring into the MTP draft/verify loop
// lives in `SpeculativeGeneration` (greedy) and the sampling-mode
// loop wired in `MLXLLMModule.runSpeculative` — this file is the
// algebra those callers compose, kept MLX-free so the unit tests stay
// CI-safe.
//
// Correctness invariant (the load-bearing assertion): same seed +
// same prompt + same temperature/top_p produces IDENTICAL output to
// non-speculative sampling. Speculation is a speedup, not a quality
// knob. At `temperature == 0` the math degenerates to greedy: the
// per-position distributions are 1-hot at argmax, the acceptance
// ratio is 1 when q and p agree (else 0), and the residual is the
// target's argmax — bit-identical to the non-speculative greedy path.

/// SplitMix64 — a tiny seedable PRNG suitable for driving the
/// speculative-sampling stream deterministically. Same seed ⇒ same
/// sequence on every host, so a seeded request reproduces. The
/// algorithm is from Vigna 2014 (the seeder used by xoshiro/xoroshiro):
/// every operation is `UInt64`-wide and `&+` / `&*` keep it modular —
/// no platform-dependent floating-point in the state itself.
public struct SplitMix64: Sendable {
    public var state: UInt64
    public init(seed: UInt64) { self.state = seed }
    public mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}

/// One stream of uniform draws. With a `seed`, every draw is
/// deterministic (drives draft sampling, accept/reject, and residual
/// resampling from ONE stream so the whole loop reproduces). Without
/// a seed, falls back to system entropy — matching the non-speculative
/// sampling path's seed-less behavior.
public struct SamplingRNG {
    private var splitmix: SplitMix64?
    private var system: SystemRandomNumberGenerator

    public init(seed: Int?) {
        if let seed {
            self.splitmix = SplitMix64(seed: UInt64(bitPattern: Int64(seed)))
        }
        self.system = SystemRandomNumberGenerator()
    }

    /// Next 64 raw bits — drives `uniform()`. Internal; callers want
    /// `uniform()`.
    public mutating func next64() -> UInt64 {
        if splitmix != nil {
            let v = splitmix!.next()
            return v
        }
        return system.next()
    }

    /// Uniform draw in [0, 1) at double precision. 53 mantissa bits
    /// (the most a `Double` can faithfully represent) so distinct draws
    /// never collide on a quantized grid.
    public mutating func uniform() -> Double {
        let bits = next64() >> 11
        return Double(bits) / Double(1 << 53 as UInt64)
    }
}

public enum SpeculativeSampling {

    /// Convert raw logits to a sampling distribution: temperature ⇒
    /// stable softmax, then optional top_k, then optional top_p, then
    /// renormalize. Truncation MUST be applied identically to draft
    /// AND target (asymmetric truncation breaks the no-quality-loss
    /// guarantee). At `temperature == 0` returns the one-hot at
    /// argmax — that's the bit-identical-greedy degenerate case.
    public static func distribution(
        logits: [Float], temperature: Float,
        topP: Float? = nil, topK: Int? = nil
    ) -> [Float] {
        precondition(!logits.isEmpty)

        // Greedy degenerate — one-hot at argmax. NB: ignores top_p/top_k
        // (they have no meaning when the distribution is a delta).
        if temperature <= 0 {
            var d = [Float](repeating: 0, count: logits.count)
            var best: Float = -.infinity
            var idx = 0
            for i in 0..<logits.count where logits[i] > best {
                best = logits[i]; idx = i
            }
            d[idx] = 1
            return d
        }

        // Temperature-scaled softmax, shifted by the max for numeric
        // stability (the standard trick — without it, large logits
        // overflow expf at fp32).
        let invT = 1.0 / temperature
        var probs = [Float](repeating: 0, count: logits.count)
        var m: Float = -.infinity
        for i in 0..<logits.count {
            let s = logits[i] * invT
            probs[i] = s
            if s > m { m = s }
        }
        if !m.isFinite { m = 0 }
        var sum: Float = 0
        for i in 0..<probs.count {
            let e = expf(probs[i] - m)
            probs[i] = e
            sum += e
        }
        if sum > 0 {
            for i in 0..<probs.count { probs[i] /= sum }
        } else {
            // Pathological — every logit was -inf. Fall back to
            // uniform so callers still get a valid distribution.
            let u = Float(1.0 / Double(probs.count))
            for i in 0..<probs.count { probs[i] = u }
        }

        // top_k: keep the K largest, zero the rest, renormalize.
        if let k = topK, k > 0, k < probs.count {
            let kept = Set(probs.indices.sorted { probs[$0] > probs[$1] }.prefix(k))
            var newSum: Float = 0
            for i in 0..<probs.count {
                if !kept.contains(i) { probs[i] = 0 }
                newSum += probs[i]
            }
            if newSum > 0 {
                for i in 0..<probs.count { probs[i] /= newSum }
            }
        }

        // top_p (nucleus): keep the smallest set whose cumulative
        // probability ≥ p, zero the rest, renormalize. At p == 1 (or
        // ≥ 1) the truncation is inert — every token survives.
        if let p = topP, p > 0, p < 1 {
            let sortedIdx = probs.indices.sorted { probs[$0] > probs[$1] }
            var keep = Set<Int>()
            var cum: Float = 0
            for i in sortedIdx {
                keep.insert(i)
                cum += probs[i]
                if cum >= p { break }
            }
            var newSum: Float = 0
            for i in 0..<probs.count {
                if !keep.contains(i) { probs[i] = 0 }
                newSum += probs[i]
            }
            if newSum > 0 {
                for i in 0..<probs.count { probs[i] /= newSum }
            }
        }

        return probs
    }

    /// Acceptance ratio `min(1, p(x) / q(x))` from the speculative
    /// sampling derivation. `q == 0` (draft sampled a token the target
    /// considers impossible after truncation) is an edge case: when
    /// `p > 0` the ratio collapses to 1 (accept — target agrees);
    /// when `p == 0` the ratio is 0 (reject — neither side allows it).
    public static func ratio(pProb: Float, qProb: Float) -> Float {
        if qProb <= 0 {
            return pProb > 0 ? 1 : 0
        }
        return min(1, pProb / qProb)
    }

    /// One acceptance test: draws `u ~ Uniform[0,1)` from `rng` and
    /// accepts iff `u < min(1, p/q)`. The caller is responsible for
    /// drawing draft samples and residual samples from the same `rng`
    /// — that single stream is the determinism contract.
    public static func acceptOrReject(
        token: Int, pDistribution: [Float], qDistribution: [Float],
        rng: inout SamplingRNG
    ) -> Bool {
        precondition(pDistribution.count == qDistribution.count)
        precondition(token >= 0 && token < pDistribution.count)
        let r = ratio(
            pProb: pDistribution[token], qProb: qDistribution[token])
        if r >= 1 { _ = rng.uniform(); return true }  // still consume the draw
        let u = Float(rng.uniform())
        return u < r
    }

    /// Residual distribution `normalize(max(0, p - q))`. Used after a
    /// reject: the next token is drawn from this corrected distribution
    /// so the overall output marginal matches `p` exactly. If `p == q`
    /// (residual is all-zero) falls back to `p` — a defensive choice;
    /// in practice the loop would have accepted, so this is unreachable.
    public static func residual(p: [Float], q: [Float]) -> [Float] {
        precondition(p.count == q.count)
        var diff = [Float](repeating: 0, count: p.count)
        var sum: Float = 0
        for i in 0..<p.count {
            let d = max(0, p[i] - q[i])
            diff[i] = d
            sum += d
        }
        if sum > 0 {
            for i in 0..<diff.count { diff[i] /= sum }
            return diff
        }
        return p
    }

    /// Inverse-CDF sample from a categorical distribution. Robust to
    /// floating-point slop in the tail: a `u` of 0.999999… that creeps
    /// past the cumulative sum returns the last non-zero index rather
    /// than overshooting.
    public static func sampleFromDistribution(
        _ distribution: [Float], rng: inout SamplingRNG
    ) -> Int {
        let u = Float(rng.uniform())
        var cum: Float = 0
        for i in 0..<distribution.count {
            cum += distribution[i]
            if u < cum { return i }
        }
        for i in stride(from: distribution.count - 1, through: 0, by: -1)
        where distribution[i] > 0 {
            return i
        }
        return distribution.count - 1
    }
}
