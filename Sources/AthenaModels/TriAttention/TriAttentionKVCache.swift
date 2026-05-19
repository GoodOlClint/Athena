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
    /// `private(set)`: fixed for normal use, but `fromState` rebuilds it
    /// from `metaState` during prompt-cache restore.
    public private(set) var config: TriAttentionConfig

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

    /// Keys/values sliced to the live offset (2 arrays, or empty when
    /// the cache has never been updated). Reconstruction goes through
    /// `fromState` (or set `metaState` *then* `state`): `metaState` owns
    /// `offset`/config, so the `state` setter only binds the arrays.
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
            switch newValue.count {
            case 0:
                keys = nil
                values = nil
            case 2:
                keys = newValue[0]
                values = newValue[1]
            default:
                fatalError(
                    "TriAttentionKVCache state must have 0 or 2 arrays")
            }
        }
    }

    /// `[kvBudget, divideLength, scoreAggregation, prefillPin, offset,
    /// prefixLength, sawPrefill, stepCount]` — fully reconstructs the
    /// eviction policy and bookkeeping. The substrate's prompt-cache
    /// registry (`cacheClassName`/`restoreCacheFromMetaState`) is
    /// `private` + hardcoded and Athena's standard path does not invoke
    /// the substrate persistence, so this cache is deliberately
    /// self-describing rather than substrate-registered (keeping the
    /// substrate clone pristine — the M21 fork decision). `fromState`
    /// is the canonical round-trip.
    public var metaState: [String] {
        get {
            [
                String(config.kvBudget),
                String(config.divideLength),
                config.scoreAggregation.rawValue,
                String(config.prefillPin),
                String(offset),
                String(prefixLength),
                String(sawPrefill),
                String(stepCount),
            ]
        }
        set { applyMetaState(newValue) }
    }

    struct CacheError: Error, CustomStringConvertible {
        let message: String
        var description: String { message }
    }

    private func applyMetaState(_ m: [String]) {
        guard m.count >= 8,
            let budget = Int(m[0]), let divide = Int(m[1]),
            let agg = TriAttentionConfig.ScoreAggregation(rawValue: m[2]),
            let off = Int(m[4]), let prefix = Int(m[5]),
            let step = Int(m[7])
        else { return }
        config = TriAttentionConfig(
            kvBudget: budget, divideLength: divide,
            scoreAggregation: agg, prefillPin: m[3] == "true")
        offset = off
        prefixLength = prefix
        sawPrefill = m[6] == "true"
        stepCount = step
    }

    /// Canonical round-trip: parse `metaState` first (it owns offset and
    /// the eviction policy), then bind the saved arrays.
    public static func fromState(
        state: [MLXArray], metaState: [String]
    ) throws -> TriAttentionKVCache {
        guard metaState.count >= 8 else {
            throw CacheError(
                message: "TriAttentionKVCache: metaState needs 8 fields")
        }
        guard state.count == 0 || state.count == 2 else {
            throw CacheError(
                message: "TriAttentionKVCache: state must be 0 or 2 arrays")
        }
        let cache = TriAttentionKVCache()
        cache.metaState = metaState
        cache.state = state
        return cache
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
