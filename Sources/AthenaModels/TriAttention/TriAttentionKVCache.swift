import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Self-evicting attention KV cache implementing norm-only TriAttention
/// (arXiv:2604.04921). Conforms to the public `KVCache` protocol
/// directly (NOT a `BaseKVCache` subclass: that class's initializer is
/// `internal`, so external subclassing is impossible). Protocol
/// conformance is the entire integration seam, so the substrate clone
/// stays pristine — contrast M20/TurboQuant, which had to modify shared
/// `MLXLMCommon` cache routing.
///
/// Storage is exactly-sized (concat on update, slice on read):
/// correctness-first, mirroring the M20.1 host-round-trip precedent. A
/// step-buffered allocation like `KVCacheSimple` is a later perf concern.
///
/// Eviction is per-layer-independent: norm-only scoring needs no token
/// positions and no cross-layer aggregation, and each attention layer
/// attends only to its own KV, so dropping a layer's low-`‖k‖` positions
/// is self-consistent. (Cross-layer-global + trigonometric scoring is the
/// deferred calibrated follow-up.) This cache is for the standard
/// attention path only; it is never substituted for `MambaCache`/GDN
/// layers nor on the MTP/speculative path.
public final class TriAttentionKVCache: KVCache {
    public let config: TriAttentionConfig

    public var offset: Int = 0
    public var maxSize: Int? { nil }

    internal var keys: MLXArray?
    internal var values: MLXArray?

    /// Tokens pinned as prefill (the first `update` is the prompt pass).
    private var prefixLength: Int = 0
    private var sawPrefill = false
    /// Decode steps since the prompt pass (drives the `divideLength` gate).
    private var stepCount: Int = 0

    public init(config: TriAttentionConfig = .init()) {
        self.config = config
    }

    public func innerState() -> [MLXArray] {
        [keys, values].compactMap { $0 }
    }

    public func update(keys newKeys: MLXArray, values newValues: MLXArray)
        -> (MLXArray, MLXArray)
    {
        if let k = keys, let v = values {
            self.keys = concatenated([k, newKeys], axis: 2)
            self.values = concatenated([v, newValues], axis: 2)
        } else {
            self.keys = newKeys
            self.values = newValues
        }
        offset = self.keys!.dim(2)

        if !sawPrefill {
            // First pass = the prompt; pin it (the substrate
            // `TokenIterator` prefills the whole prompt in one forward).
            sawPrefill = true
            prefixLength = offset
            return current()
        }

        stepCount += 1
        if shouldCompress() {
            compress()
        }
        return current()
    }

    private func current() -> (MLXArray, MLXArray) {
        guard let k = keys, let v = values else {
            fatalError("TriAttentionKVCache.current() before any update")
        }
        if offset == k.dim(2) { return (k, v) }
        return (k[.ellipsis, ..<offset, 0...], v[.ellipsis, ..<offset, 0...])
    }

    private func shouldCompress() -> Bool {
        let effective =
            config.prefillPin ? max(0, offset - prefixLength) : offset
        return effective >= config.kvBudget
            && stepCount > 0
            && stepCount % config.divideLength == 0
    }

    private func compress() {
        guard let k = keys, let v = values else { return }
        let prefix = config.prefillPin ? min(prefixLength, offset) : 0
        let decodeLen = offset - prefix
        let decodeBudget = max(0, config.kvBudget - prefix)
        if decodeLen <= decodeBudget { return }
        let keepK = min(decodeBudget, decodeLen)

        let scores = TriAttentionScorer.scoreKeys(
            k[.ellipsis, ..<offset, 0...],
            aggregation: config.scoreAggregation)

        // Top-`keepK` decode positions by score; deterministic tiebreak
        // keeps the earlier token. Re-sorted ascending so retained tokens
        // stay in temporal order; prefill (pinned) prepended.
        let decodeOrder = (prefix..<offset).sorted {
            scores[$0] != scores[$1]
                ? scores[$0] > scores[$1] : $0 < $1
        }
        let keptDecode = decodeOrder.prefix(keepK).sorted()
        let keepIndices = Array(0..<prefix) + keptDecode
        let idx = MLXArray(keepIndices.map { Int32($0) })

        keys = take(k[.ellipsis, ..<offset, 0...], idx, axis: 2)
        values = take(v[.ellipsis, ..<offset, 0...], idx, axis: 2)
        offset = keepIndices.count
    }

    public var isTrimmable: Bool { true }

    @discardableResult
    public func trim(_ n: Int) -> Int {
        let trimmed = min(offset, n)
        offset -= trimmed
        return trimmed
    }

    public var state: [MLXArray] {
        get {
            guard let k = keys, let v = values else { return [] }
            if offset == k.dim(2) { return [k, v] }
            return [
                k[.ellipsis, ..<offset, 0...],
                v[.ellipsis, ..<offset, 0...],
            ]
        }
        set {
            guard newValue.count == 2 else {
                fatalError(
                    "TriAttentionKVCache state must have exactly 2 arrays")
            }
            keys = newValue[0]
            values = newValue[1]
            offset = keys!.dim(2)
        }
    }

    /// Minimal placeholder; full prompt-cache round-trip
    /// (prefix/step/config metadata + `fromState`) is M21.3.
    public var metaState: [String] {
        get { [""] }
        set {}
    }

    /// Mirrors `BaseKVCache`'s default mask logic (single token ⇒ none;
    /// otherwise causal over the current offset). Valid after eviction:
    /// every retained key is in the past relative to new tokens.
    public func makeMask(
        n: Int, windowSize: Int?, returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        if n == 1 { return .none }
        if returnArray || (windowSize != nil && n > windowSize!) {
            return .array(
                createCausalMask(n: n, offset: offset, windowSize: windowSize))
        }
        return .causal
    }

    public func copy() -> any KVCache {
        let new = TriAttentionKVCache(config: config)
        new.offset = offset
        new.prefixLength = prefixLength
        new.sawPrefill = sawPrefill
        new.stepCount = stepCount
        new.keys = keys.map { $0[.ellipsis] }
        new.values = values.map { $0[.ellipsis] }
        return new
    }
}
