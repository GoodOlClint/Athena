import AthenaModels
import Foundation
import Logging
import MLX
import MLXLMCommon

/// M59.1 — cross-request prompt-prefix KV reuse for the MTP/speculative
/// backbone.
///
/// Today every `/v1/chat/completions` request prefills the whole prompt from
/// a fresh cache. When N back-to-back requests share a bit-identical leading
/// token run (a static system prompt + verbatim document, varying only in the
/// trailing instruction), passes 2..N pay a full cold prefill they shouldn't.
/// This store keeps a small, bounded set of post-prefill backbone cache
/// snapshots and, on a request whose prompt shares a long common prefix,
/// resumes prefill from a chunk boundary inside that shared run.
///
/// ## Bit-identical contract (the hard gate, matches M20/M21)
/// Prefix reuse + suffix prefill MUST produce byte-identical greedy output to
/// a cold full prefill. Two code facts make that provable:
///   1. Prefill chunks the prompt on a fixed 512-token absolute grid
///      (`SpeculativeGeneration.generate`). Re-chunking on the SAME grid is
///      bit-identical because the GatedDeltaNet forward carries recurrent
///      state across chunks identically whether or not we resumed
///      (the cached-decode invariant the MTP `nConfirmed` split already
///      relies on — see `AthenaQwen35GatedDeltaNet.callAsFunction`).
///   2. Every substrate `KVCache` conformer exposes an exact deep `copy()`,
///      and `MambaCache.state` is the same `[conv, ssm]` pair `GDNRollback`
///      snapshots. So clone-on-hit and recurrent checkpoints are first-class.
/// Therefore we snapshot recurrent state ONLY at 512-multiples, restore at
/// `B = floor(L/512)·512` for a common-prefix length `L`, trim attention to
/// `B`, and re-run one continuous prefill from `B`. Tokens `[0:B]` are shared
/// (`B ≤ L`), so the restored state equals what a cold prefill of the NEW
/// prompt would hold at `B`.
///
/// The MTP draft cache is intentionally NOT snapshotted: prefill never calls
/// `mtpForward`, so `makeMTPCache()` is at offset 0 at the prefill→decode seam
/// in both the cold and warm paths. The first draft consumes the freshly
/// recomputed last-token hidden + an empty MTP cache identically either way.
///
/// ## Concurrency / immutability
/// Implemented as a lock-guarded `@unchecked Sendable` class (not an `actor`)
/// so every `MLXArray`/`KVCache` operation stays inside the model's
/// serial-access domain (`ModelContainer.perform`) — no non-`Sendable`
/// transfer across isolation. Mirrors the `GDNRollback` / M29 rate-limit
/// idiom. A cached entry is NEVER handed to a running generation: on a hit we
/// `copy()` the attention caches and `[.ellipsis]`-copy the recurrent
/// checkpoint into a fresh working cache, and we refcount the entry so the
/// LRU evictor can't reclaim it while a request is mid-flight.
///
/// M59.1 scope: keyed by resident model id only, bounded by entry COUNT.
/// Per-principal / `prompt_cache_key` scoping and byte/idle-TTL governor
/// eviction are M59.2/M59.3.
public final class PrefixKVCache: @unchecked Sendable {

    nonisolated static let log = Logger(label: "athena.prefix-cache")

    /// Prefill chunk granularity — MUST match the grid
    /// `SpeculativeGeneration.generate` prefills on, or reuse is not
    /// bit-identical.
    public static let chunkSize = 512

    // MARK: - Stored entry

    private final class Entry {
        let id: Int
        let scope: String
        let tokens: [Int]
        /// Full-length (prompt-length) attention-cache clones, indexed by
        /// backbone layer; `nil` at recurrent (Mamba) layers.
        let attn: [KVCache?]
        /// boundary offset (512-multiple) → layerIdx → cloned `[conv, ssm]`.
        let checkpoints: [Int: [Int: [MLXArray]]]
        let byteEstimate: Int
        /// Wall-clock of last store/hit — drives both LRU ordering (oldest
        /// evicted first) and idle-TTL expiry (M59.2).
        var lastUsed: Date
        var refcount: Int

        init(
            id: Int, scope: String, tokens: [Int], attn: [KVCache?],
            checkpoints: [Int: [Int: [MLXArray]]], byteEstimate: Int,
            lastUsed: Date
        ) {
            self.id = id
            self.scope = scope
            self.tokens = tokens
            self.attn = attn
            self.checkpoints = checkpoints
            self.byteEstimate = byteEstimate
            self.lastUsed = lastUsed
            self.refcount = 0
        }
    }

    /// A working set returned on a hit. `caches` is caller-owned (cloned) and
    /// safe to mutate/decode against; `startOffset` is the 512-boundary `B` to
    /// resume prefill from; `commonPrefix` is `L` (for logging).
    public struct Hit {
        public let caches: [KVCache]
        public let startOffset: Int
        public let commonPrefix: Int
        fileprivate let entryId: Int
    }

    /// Collects recurrent checkpoints during a cold prefill so the entry can
    /// be stored at the prefill→decode seam. Created only when caching is
    /// enabled and the request missed.
    public final class Recorder {
        fileprivate var checkpoints: [Int: [Int: [MLXArray]]] = [:]
        fileprivate init() {}

        /// Snapshot recurrent (Mamba) layer state at an absolute 512-boundary.
        /// Capturing a reference is a valid snapshot: the GDN forward
        /// REASSIGNS `cache[0]`/`cache[1]` to new arrays each step, so the
        /// arrays we hold are never mutated in place. `eval` decouples them
        /// from the lazy graph and bounds memory.
        public func snapshot(offset: Int, backbone: [KVCache]) {
            guard offset > 0, offset % PrefixKVCache.chunkSize == 0 else { return }
            var layers: [Int: [MLXArray]] = [:]
            for (i, c) in backbone.enumerated() {
                guard let mc = c as? MambaCache else { continue }
                let s = mc.state.map { $0[.ellipsis] }
                if !s.isEmpty {
                    eval(s)
                    layers[i] = s
                }
            }
            if !layers.isEmpty { checkpoints[offset] = layers }
        }
    }

    // MARK: - State

    private let lock = NSLock()
    private let maxEntries: Int
    /// Pool byte cap (M59.2). Default governor-derived (= promptCacheCapBytes)
    /// so the persistent pool and the per-request admission guard share one
    /// cap and can't starve each other. 0 ⇒ no byte cap (count-only, M59.1).
    private let maxBytes: Int
    /// Evict entries idle longer than this many seconds (M59.2, default 600,
    /// mirrors OpenAI's 5–10 min inactivity eviction). 0 ⇒ no idle expiry.
    private let idleTTL: TimeInterval
    private var entries: [Entry] = []
    private var nextId = 0
    private var hits = 0
    private var misses = 0
    /// Cumulative count of entries evicted by the count/byte/idle policy,
    /// for operator legibility (M59.2 snapshot / M59.4 stats).
    private var evictions = 0

    public init(maxEntries: Int, maxBytes: Int = 0, idleTTLSecs: Int = 600) {
        self.maxEntries = max(1, maxEntries)
        self.maxBytes = max(0, maxBytes)
        self.idleTTL = TimeInterval(max(0, idleTTLSecs))
    }

    // MARK: - Lookup / acquire

    /// Look up the longest shared prefix within `scope`. On a usable hit
    /// (shared prefix spans at least one full 512 chunk), returns a fresh
    /// working cache restored to boundary `B` and increments the entry
    /// refcount — the caller MUST `release(_:)` it when generation ends.
    public func acquire(
        scope: String, promptTokens: [Int], model: AthenaQwen35Model
    ) -> Hit? {
        lock.lock()
        defer { lock.unlock() }
        let now = Date()
        sweepIdle(now: now)

        var best: (entry: Entry, L: Int, B: Int)?
        // B must be a stored boundary, ≤ L (shared tokens), and
        // ≤ promptTokens.count - 1 so the warm loop reproduces cold's final
        // partial chunk exactly.
        let cap0 = promptTokens.count - 1
        for e in entries where e.scope == scope {
            let l = Self.commonPrefixLength(e.tokens, promptTokens)
            let cap = min(l, cap0)
            guard cap >= Self.chunkSize else { continue }
            guard let b = e.checkpoints.keys.filter({ $0 <= cap }).max() else {
                continue
            }
            if best == nil || b > best!.B { best = (e, l, b) }
        }
        guard let pick = best else {
            misses += 1
            return nil
        }
        let e = pick.entry
        let b = pick.B

        // Build a fresh working cache (correct per-layer Mamba/attention
        // structure), then inject the cloned attention + restored recurrent
        // checkpoint. Nothing from the entry is aliased into the result.
        var working = model.newCache(parameters: nil)
        let snap = e.checkpoints[b]
        for i in working.indices {
            if let mc = working[i] as? MambaCache {
                if let layer = snap?[i] {
                    mc.state = layer.map { $0[.ellipsis] }
                    mc.offset = b
                }
            } else if let cached = e.attn[i] {
                let c = cached.copy()
                c.trim(e.tokens.count - b)
                working[i] = c
            }
        }
        e.refcount += 1
        e.lastUsed = now
        hits += 1
        return Hit(
            caches: working, startOffset: b, commonPrefix: pick.L, entryId: e.id)
    }

    public func release(_ hit: Hit) {
        lock.lock()
        defer { lock.unlock() }
        if let e = entries.first(where: { $0.id == hit.entryId }), e.refcount > 0 {
            e.refcount -= 1
        }
    }

    public func makeRecorder() -> Recorder { Recorder() }

    // MARK: - Store (cold-prefill seam)

    /// Store a new entry from the post-prefill backbone (offset ==
    /// `promptTokens.count`, BEFORE decode mutates it). Attention caches are
    /// cloned here; recurrent checkpoints come from the `Recorder`. Skips when
    /// no usable checkpoint was captured (prompt shorter than one chunk).
    public func store(
        scope: String, promptTokens: [Int], backbone: [KVCache],
        recorder: Recorder
    ) {
        guard !recorder.checkpoints.isEmpty else { return }
        lock.lock()
        defer { lock.unlock() }
        let now = Date()
        sweepIdle(now: now)

        var attn = [KVCache?](repeating: nil, count: backbone.count)
        var bytes = 0
        for i in backbone.indices where !(backbone[i] is MambaCache) {
            let c = backbone[i].copy()
            eval(c.state)
            attn[i] = c
            bytes += c.state.reduce(0) { $0 + $1.nbytes }
        }
        for layers in recorder.checkpoints.values {
            for arrays in layers.values {
                bytes += arrays.reduce(0) { $0 + $1.nbytes }
            }
        }
        nextId += 1
        let entry = Entry(
            id: nextId, scope: scope, tokens: promptTokens, attn: attn,
            checkpoints: recorder.checkpoints, byteEstimate: bytes,
            lastUsed: now)
        entries.append(entry)
        evictIfNeeded()
    }

    // MARK: - Eviction (count + bytes + idle-TTL, refcount-protected)

    /// Drop entries idle longer than `idleTTL` (skipping in-use ones). Lazy:
    /// driven from `acquire`/`store`/`stats`, so an idle pool drains without a
    /// background timer. No-op when `idleTTL == 0`.
    private func sweepIdle(now: Date) {
        guard idleTTL > 0 else { return }
        let before = entries.count
        entries.removeAll {
            $0.refcount == 0 && now.timeIntervalSince($0.lastUsed) > idleTTL
        }
        evictions += before - entries.count
    }

    /// Evict LRU entries until under BOTH the count cap and the byte cap,
    /// skipping any a live generation still holds (refcount > 0). Bytes cap is
    /// inert when `maxBytes == 0`.
    private func evictIfNeeded() {
        func overCap() -> Bool {
            if entries.count > maxEntries { return true }
            if maxBytes > 0 {
                return entries.reduce(0) { $0 + $1.byteEstimate } > maxBytes
            }
            return false
        }
        while overCap() {
            guard
                let victimIdx = entries.enumerated()
                    .filter({ $0.element.refcount == 0 })
                    .min(by: { $0.element.lastUsed < $1.element.lastUsed })?
                    .offset
            else { break }  // all remaining entries are in use
            entries.remove(at: victimIdx)
            evictions += 1
        }
    }

    /// Drop every entry not currently in use (refcount == 0). Returns the
    /// count freed. Used by the governor's pressure relief (M59.2) and the
    /// operator flush (M59.4). Entries held by an in-flight generation
    /// survive — they're freed when that request releases them.
    @discardableResult
    public func flushIdle() -> Int {
        lock.lock()
        defer { lock.unlock() }
        let before = entries.count
        entries.removeAll { $0.refcount == 0 }
        let freed = before - entries.count
        evictions += freed
        return freed
    }

    // MARK: - Stats (M59.2/M59.4 surfaces)

    public struct Stats: Sendable {
        public let entries: Int
        public let bytes: Int
        public let hits: Int
        public let misses: Int
        public let evictions: Int
        public let maxEntries: Int
        public let maxBytes: Int
    }

    public func stats() -> Stats {
        lock.lock()
        defer { lock.unlock() }
        // Idle-sweep here too so a frequently-scraped /healthz keeps the
        // reported pool honest without waiting for the next store/hit.
        sweepIdle(now: Date())
        return Stats(
            entries: entries.count,
            bytes: entries.reduce(0) { $0 + $1.byteEstimate },
            hits: hits, misses: misses, evictions: evictions,
            maxEntries: maxEntries, maxBytes: maxBytes)
    }

    /// Sync (bytes, entries) probe for the governor snapshot — cheap, takes
    /// the lock, no idle-sweep (snapshot must not mutate timing semantics).
    public func poolBytesAndEntries() -> (bytes: Int, entries: Int) {
        lock.lock()
        defer { lock.unlock() }
        return (entries.reduce(0) { $0 + $1.byteEstimate }, entries.count)
    }

    // MARK: - Helpers

    static func commonPrefixLength(_ a: [Int], _ b: [Int]) -> Int {
        let n = min(a.count, b.count)
        var i = 0
        while i < n && a[i] == b[i] { i += 1 }
        return i
    }
}
