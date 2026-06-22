import AthenaCore
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

    /// M59.3 — how an entry's scope is keyed. The resident model id is
    /// ALWAYS part of the key (a rebind must never serve one model's KV to
    /// another); this selects what else joins it.
    ///   - `.principal`  (default): isolate by authenticated principal —
    ///     one caller's prefix is never reused for another (the security
    ///     default).
    ///   - `.cacheKey`: key by the OpenAI `prompt_cache_key` hint (falling
    ///     back to the principal when the request omits it).
    ///   - `.both`: principal AND cache_key — the narrowest scope.
    public enum ScopeMode: String, Sendable {
        case principal
        case cacheKey = "cache_key"
        case both
    }

    /// Build the scope key for a request. The model id is always included;
    /// `\u{1}` separates fields (it can't appear in a model id, principal,
    /// or JSON-string cache key, so fields can't collide).
    public func scopeKey(
        model: String, principal: String?, cacheKey: String?
    ) -> String {
        let p = principal ?? "anon"
        switch scopeMode {
        case .principal:
            return "\(model)\u{1}p:\(p)"
        case .cacheKey:
            return "\(model)\u{1}k:\(cacheKey ?? p)"
        case .both:
            return "\(model)\u{1}p:\(p)\u{1}k:\(cacheKey ?? "")"
        }
    }

    // MARK: - Stored entry

    /// What an entry holds for its KV. With `[prompt_cache]` idle encryption OFF
    /// (default) the entry parks the plaintext substrate caches exactly as M59
    /// always has; with it ON (ADR 024 T3) the entry holds only AES-256-GCM
    /// ciphertext, so the idle pool is never plaintext-at-rest. Every entry in a
    /// given pool is the same variant (the mode is a cache-wide constant), so the
    /// two never mix.
    private enum Payload {
        /// Full-length attention-cache clones (nil at recurrent layers) +
        /// 512-boundary → layerIdx → `[conv, ssm]` recurrent checkpoints.
        case plain(attn: [KVCache?], checkpoints: [Int: [Int: [MLXArray]]])
        /// The same two structures, each tensor group serialized + sealed:
        /// per-layer sealed attention state (nil at recurrent layers) +
        /// boundary → layerIdx → sealed `[conv, ssm]`.
        case sealed(attn: [Data?], checkpoints: [Int: [Int: Data]])
    }

    private final class Entry {
        let id: Int
        let scope: String
        let tokens: [Int]
        let payload: Payload
        let byteEstimate: Int
        /// Wall-clock of last store/hit — drives both LRU ordering (oldest
        /// evicted first) and idle-TTL expiry (M59.2).
        var lastUsed: Date
        var refcount: Int

        /// Stored recurrent-checkpoint boundary offsets (512-multiples),
        /// independent of whether the payload is plain or sealed — the hit scan
        /// picks the largest `≤ cap`.
        var checkpointOffsets: [Int] {
            switch payload {
            case .plain(_, let checkpoints): return Array(checkpoints.keys)
            case .sealed(_, let checkpoints): return Array(checkpoints.keys)
            }
        }

        init(
            id: Int, scope: String, tokens: [Int], payload: Payload,
            byteEstimate: Int, lastUsed: Date
        ) {
            self.id = id
            self.scope = scope
            self.tokens = tokens
            self.payload = payload
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

    // MARK: - Disk tier (ADR 027)

    /// Optional disk L2 under the in-RAM L1 pool. When present, idle entries
    /// being dropped (evict / idle-sweep / flush) are **demoted to disk** instead
    /// of discarded, and an in-RAM miss falls through to a disk restore. Off by
    /// default; constructed by the daemon only when `prompt_cache_persist_to_disk`
    /// is on and the daemon is not in a no-write loopback posture (ADR 025).
    public struct DiskTier {
        public let store: KVSnapshotStore
        public let kek: KEKProvider
        /// Skip-on-skew identity written into each blob header. The served model
        /// id (which encodes quant for store models); `quantTag` reserved.
        public let modelID: String
        public let quantTag: String
        public let maxEntries: Int?
        public let maxBytes: Int?
        public let maxAgeSecs: UInt64?
        /// Frontier/eager spill (ADR 027 S4): when true, a new entry is written
        /// to disk at the store seam (`save_reason=continued`), not only on
        /// idle-drop/shutdown — so a hard CRASH (SIGKILL/panic) doesn't lose it.
        /// Off by default (the spill is synchronous I/O on the post-prefill path,
        /// a TTFT cost the operator opts into for crash survival).
        public let eager: Bool
        public init(
            store: KVSnapshotStore, kek: KEKProvider, modelID: String,
            quantTag: String = "", maxEntries: Int?, maxBytes: Int?, maxAgeSecs: UInt64?,
            eager: Bool = false
        ) {
            self.store = store
            self.kek = kek
            self.modelID = modelID
            self.quantTag = quantTag
            self.maxEntries = maxEntries
            self.maxBytes = maxBytes
            self.maxAgeSecs = maxAgeSecs
            self.eager = eager
        }
    }

    /// Sentinel `Hit.entryId` for a disk-restored working set: there is no RAM
    /// `Entry` to refcount, so `release` is a no-op for it.
    private static let diskEntryId = -1

    // MARK: - State

    private let lock = NSLock()
    private let disk: DiskTier?
    private let maxEntries: Int
    /// Pool byte cap (M59.2). Default governor-derived (= promptCacheCapBytes)
    /// so the persistent pool and the per-request admission guard share one
    /// cap and can't starve each other. 0 ⇒ no byte cap (count-only, M59.1).
    private let maxBytes: Int
    /// Evict entries idle longer than this many seconds (M59.2, default 600,
    /// mirrors OpenAI's 5–10 min inactivity eviction). 0 ⇒ no idle expiry.
    private let idleTTL: TimeInterval
    private let scopeMode: ScopeMode
    /// ADR 024 T3 — non-nil iff `prompt_cache_encrypt_idle` is on. Holds the
    /// process-ephemeral AES-256-GCM key and seals/opens idle entries. Rotated
    /// (key dropped + zeroed) whenever the pool drains to empty, so the key
    /// never outlives the data it protected.
    private let cipher: IdleKVCipher?
    private var entries: [Entry] = []
    private var nextId = 0
    private var hits = 0
    private var misses = 0
    /// Cumulative count of entries evicted by the count/byte/idle policy,
    /// for operator legibility (M59.2 snapshot / M59.4 stats).
    private var evictions = 0

    public init(
        maxEntries: Int, maxBytes: Int = 0, idleTTLSecs: Int = 600,
        scope: ScopeMode = .principal, encryptIdle: Bool = false,
        disk: DiskTier? = nil
    ) {
        self.maxEntries = max(1, maxEntries)
        self.maxBytes = max(0, maxBytes)
        self.idleTTL = TimeInterval(max(0, idleTTLSecs))
        self.scopeMode = scope
        self.cipher = encryptIdle ? IdleKVCipher() : nil
        self.disk = disk
    }

    /// Whether a disk L2 tier is attached (ADR 027).
    public var persistsToDisk: Bool { disk != nil }

    /// Whether idle entries are held as AES-256-GCM ciphertext (ADR 024 T3).
    public var encryptsIdleEntries: Bool { cipher != nil }

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
            guard let b = e.checkpointOffsets.filter({ $0 <= cap }).max() else {
                continue
            }
            if best == nil || b > best!.B { best = (e, l, b) }
        }
        guard let pick = best else {
            // No in-RAM (L1) entry — fall through to the disk (L2) tier.
            if let hit = restoreFromDisk(
                scope: scope, promptTokens: promptTokens, model: model)
            {
                hits += 1
                return hit
            }
            misses += 1
            return nil
        }
        let e = pick.entry
        let b = pick.B

        // Build a fresh working cache (correct per-layer Mamba/attention
        // structure), then inject the restored attention + recurrent checkpoint.
        // Nothing from the entry is aliased into the result. A sealed entry is
        // decrypted just-in-time here (the only place its plaintext exists, and
        // only under this lock); on any decrypt/decode failure we fall back to a
        // miss so the caller cold-prefills rather than serving a corrupt cache.
        guard let working = rebuildWorking(entry: e, b: b, model: model) else {
            misses += 1
            return nil
        }
        e.refcount += 1
        e.lastUsed = now
        hits += 1
        return Hit(
            caches: working, startOffset: b, commonPrefix: pick.L, entryId: e.id)
    }

    /// Reconstruct the per-request working cache from an entry at boundary `b`.
    /// `nil` only when a sealed entry fails to decrypt/decode (treated as a
    /// miss). Plain and sealed produce the same logical cache — full prefix KV
    /// restored then trimmed to `b` — so the resumed prefill is bit-identical.
    private func rebuildWorking(
        entry e: Entry, b: Int, model: AthenaQwen35Model
    ) -> [KVCache]? {
        var working = model.newCache(parameters: nil)
        switch e.payload {
        case .plain(let attn, let checkpoints):
            let snap = checkpoints[b]
            for i in working.indices {
                if let mc = working[i] as? MambaCache {
                    if let layer = snap?[i] {
                        mc.state = layer.map { $0[.ellipsis] }
                        mc.offset = b
                    }
                } else if let cached = attn[i] {
                    let c = cached.copy()
                    c.trim(e.tokens.count - b)
                    working[i] = c
                }
            }
        case .sealed(let sealedAttn, let sealedCheckpoints):
            // Invariant: a sealed payload only exists when the cipher does.
            guard let cipher else { return nil }
            let snap = sealedCheckpoints[b]
            for i in working.indices {
                if let mc = working[i] as? MambaCache {
                    if let blob = snap?[i] {
                        guard
                            let arrays = openArrays(
                                blob, aad: checkpointAAD(b, i), cipher: cipher)
                        else { return nil }
                        mc.state = arrays
                        mc.offset = b
                    }
                } else if let blob = sealedAttn[i] {
                    guard
                        let arrays = openArrays(
                            blob, aad: attnAAD(i), cipher: cipher)
                    else { return nil }
                    // Restore the full-prefix attention state onto the fresh
                    // KVCacheSimple (the setter takes offset to the stored
                    // length), then trim to the resume boundary `b` — matching
                    // the plain path's `copy().trim()`.
                    working[i].state = arrays
                    _ = working[i].trim(e.tokens.count - b)
                }
            }
        }
        return working
    }

    public func release(_ hit: Hit) {
        lock.lock()
        defer { lock.unlock() }
        // A disk-restored hit has no RAM entry to refcount — no-op.
        if let e = entries.first(where: { $0.id == hit.entryId }), e.refcount > 0 {
            e.refcount -= 1
        }
    }

    // MARK: - Disk tier restore/spill (ADR 027)

    /// On an in-RAM miss, probe the disk store at descending chunk boundaries of
    /// the prompt (`KVPrefixDigest`). The largest boundary with a digest hit whose
    /// blob decrypts + decodes + rebuilds is the restore point. `nil` when no disk
    /// tier, no hit, or every candidate fails to materialize (caller cold-prefills).
    /// Called with `lock` held.
    private func restoreFromDisk(
        scope: String, promptTokens: [Int], model: AthenaQwen35Model
    ) -> Hit? {
        guard let disk else { return nil }
        for b in KVPrefixDigest.probeBoundaries(
            promptCount: promptTokens.count, chunkSize: Self.chunkSize)
        {
            let key = KVPrefixDigest.prefixHash(
                scope: scope, tokens: promptTokens, count: b)
            guard
                let bodyData = disk.store.load(
                    prefixHash: key, requireModel: disk.modelID,
                    requireQuant: disk.quantTag, kek: disk.kek),
                let body = try? KVEntryBody.decode(bodyData),
                let working = rebuildWorkingFromDisk(body: body, b: b, model: model)
            else { continue }
            Self.log.notice(
                """
                prefix-cache DISK HIT prompt=\(promptTokens.count) B=\(b) \
                suffix=\(promptTokens.count - b)
                """,
                metadata: ["function": "PrefixKVCache.restoreFromDisk"])
            return Hit(
                caches: working, startOffset: b, commonPrefix: b,
                entryId: Self.diskEntryId)
        }
        return nil
    }

    /// Rebuild a fresh working cache from a disk body whose attention slots were
    /// stored already trimmed to `b` (so setting `.state` lands the cache at
    /// offset `b`) and whose recurrent layers are the checkpoint at `b`. `nil` on
    /// any decode failure (treated as a miss).
    private func rebuildWorkingFromDisk(
        body: KVEntryBody, b: Int, model: AthenaQwen35Model
    ) -> [KVCache]? {
        var working = model.newCache(parameters: nil)
        for i in working.indices {
            if let mc = working[i] as? MambaCache {
                if let blob = body.recurrentLayers[i] {
                    guard let arrays = try? KVByteCodec.decode(blob) else { return nil }
                    mc.state = arrays
                    mc.offset = b
                }
            } else if i < body.attnSlots.count, let blob = body.attnSlots[i] {
                guard let arrays = try? KVByteCodec.decode(blob) else { return nil }
                working[i].state = arrays  // arrays are length-b ⇒ offset == b
            }
        }
        return working
    }

    /// Demote a `.plain` entry to disk before it is dropped — one blob per stored
    /// 512-boundary (attention trimmed to that boundary + the recurrent
    /// checkpoint), so a returning prompt finds its largest shared boundary,
    /// matching the in-RAM divergent-reuse semantics. Best-effort: a write failure
    /// is swallowed (the entry is dropped from RAM regardless). Sealed entries
    /// (`prompt_cache_encrypt_idle` on) are not spilled in this slice — RAM-cipher
    /// and disk-cipher are alternative at-rest strategies (see the construction
    /// warning). Called with `lock` held.
    private func demoteToDisk(_ e: Entry, reason: KVSnapshotHeader.SaveReason) {
        guard let disk else { return }
        guard case .plain(let attn, let checkpoints) = e.payload else { return }
        let now = UInt64(max(0, Date().timeIntervalSince1970))
        for (b, recurLayers) in checkpoints {
            var slots = [Data?](repeating: nil, count: attn.count)
            for i in attn.indices {
                guard let cache = attn[i] else { continue }
                let c = cache.copy()
                _ = c.trim(e.tokens.count - b)
                eval(c.state)
                slots[i] = KVByteCodec.encode(c.state)
            }
            var recurrent: [Int: Data] = [:]
            for (layer, arrays) in recurLayers {
                recurrent[layer] = KVByteCodec.encode(arrays)
            }
            let body = KVEntryBody(attnSlots: slots, recurrentLayers: recurrent)
            let key = KVPrefixDigest.prefixHash(scope: e.scope, tokens: e.tokens, count: b)
            try? disk.store.save(
                prefixHash: key, modelID: disk.modelID, quantTag: disk.quantTag,
                scopeKey: e.scope, tokenCount: UInt64(b),
                contextSize: UInt64(e.tokens.count), saveReason: reason,
                createdUnix: now, lastUsedUnix: now, body: body.encode(), kek: disk.kek)
        }
        disk.store.enforceRetention(
            maxEntries: disk.maxEntries, maxBytes: disk.maxBytes,
            maxAgeSecs: disk.maxAgeSecs, now: now)
    }

    /// Demote each soon-to-be-dropped victim to disk (if a disk tier exists).
    /// Called with `lock` held, before the victims leave `entries`.
    private func demoteVictims(_ victims: Set<Int>, reason: KVSnapshotHeader.SaveReason) {
        guard disk != nil, !victims.isEmpty else { return }
        for (i, e) in entries.enumerated() where victims.contains(i) {
            demoteToDisk(e, reason: reason)
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

        let payload: Payload
        let bytes: Int
        if let cipher {
            // ADR 024 T3 — seal each tensor group; the entry holds only
            // ciphertext, the transient plaintext byte buffers are zeroed in
            // `sealArrays`, and the source MLXArrays are dropped (released to
            // MLX) so no plaintext copy is parked for the idle TTL.
            guard let s = sealEntry(backbone: backbone, recorder: recorder, cipher: cipher)
            else { return }  // a seal failure (not expected) just skips caching
            payload = s.payload
            bytes = s.bytes
        } else {
            var attn = [KVCache?](repeating: nil, count: backbone.count)
            var plainBytes = 0
            for i in backbone.indices where !(backbone[i] is MambaCache) {
                let c = backbone[i].copy()
                eval(c.state)
                attn[i] = c
                plainBytes += c.state.reduce(0) { $0 + $1.nbytes }
            }
            for layers in recorder.checkpoints.values {
                for arrays in layers.values {
                    plainBytes += arrays.reduce(0) { $0 + $1.nbytes }
                }
            }
            payload = .plain(attn: attn, checkpoints: recorder.checkpoints)
            bytes = plainBytes
        }

        nextId += 1
        let entry = Entry(
            id: nextId, scope: scope, tokens: promptTokens, payload: payload,
            byteEstimate: bytes, lastUsed: now)
        entries.append(entry)
        // ADR 027 S4 — eager/frontier spill: persist the new entry to disk now
        // (continued save) so a crash before the next idle-drop/shutdown doesn't
        // lose it. Only when disk persistence is on AND eager is enabled.
        if disk?.eager == true { demoteToDisk(entry, reason: .continued) }
        evictIfNeeded()
    }

    // MARK: - Idle-entry encryption (ADR 024 T3)

    private func attnAAD(_ layer: Int) -> String { "attn:\(layer)" }
    private func checkpointAAD(_ offset: Int, _ layer: Int) -> String {
        "ckpt:\(offset):\(layer)"
    }

    /// Serialize + seal one tensor group. The intermediate plaintext byte buffer
    /// is `secureZero`'d on the way out (best-effort, ADR 023/T2 boundary);
    /// `nil` only on a seal failure (never expected — surfaced rather than
    /// trapped).
    private func sealArrays(
        _ arrays: [MLXArray], aad: String, cipher: IdleKVCipher
    ) -> Data? {
        var plain = KVByteCodec.encode(arrays)
        defer { ProcessHardening.secureZero(&plain) }
        return try? cipher.seal(plain, aad: Data(aad.utf8))
    }

    /// Open + deserialize one sealed tensor group. The decrypted plaintext is
    /// zeroed after the MLXArrays are rebuilt (MLX copies the bytes into its own
    /// buffers). `nil` on a tampered/AAD-mismatched/rotated-key box.
    private func openArrays(
        _ blob: Data, aad: String, cipher: IdleKVCipher
    ) -> [MLXArray]? {
        guard var plain = cipher.open(blob, aad: Data(aad.utf8)) else { return nil }
        defer { ProcessHardening.secureZero(&plain) }
        return try? KVByteCodec.decode(plain)
    }

    /// Build a sealed payload (+ its ciphertext byte total for governor
    /// accounting) from the post-prefill backbone. `nil` on any seal failure.
    private func sealEntry(
        backbone: [KVCache], recorder: Recorder, cipher: IdleKVCipher
    ) -> (payload: Payload, bytes: Int)? {
        var sealedAttn = [Data?](repeating: nil, count: backbone.count)
        var bytes = 0
        for i in backbone.indices where !(backbone[i] is MambaCache) {
            // `state` slices to the live offset (== prompt length); the seal
            // snapshots a contiguous byte copy, so the live backbone may keep
            // decoding afterwards without affecting the ciphertext.
            guard
                let blob = sealArrays(
                    backbone[i].state, aad: attnAAD(i), cipher: cipher)
            else { return nil }
            sealedAttn[i] = blob
            bytes += blob.count
        }
        var sealedCheckpoints: [Int: [Int: Data]] = [:]
        for (offset, layers) in recorder.checkpoints {
            var sealedLayers: [Int: Data] = [:]
            for (layer, arrays) in layers {
                guard
                    let blob = sealArrays(
                        arrays, aad: checkpointAAD(offset, layer), cipher: cipher)
                else { return nil }
                sealedLayers[layer] = blob
                bytes += blob.count
            }
            sealedCheckpoints[offset] = sealedLayers
        }
        return (.sealed(attn: sealedAttn, checkpoints: sealedCheckpoints), bytes)
    }

    /// ADR 024 T3 — once the encrypted pool drains to empty, rotate (drop +
    /// zero) the key so it never outlives the data it protected. Safe only when
    /// `entries` is empty: a still-present entry (even in-flight) may be
    /// re-acquired and needs its blobs decryptable under the current key. Call
    /// with `lock` held, after any path that can empty the pool.
    private func rotateKeyIfPoolEmptyLocked() {
        if let cipher, entries.isEmpty { cipher.rotate() }
    }

    // MARK: - Eviction (count + bytes + idle-TTL, refcount-protected)

    /// Metadata-only view of `entries` (same order) for the pure eviction
    /// policy — no `MLXArray`/`KVCache` crosses into `PrefixCachePolicy`.
    private func entryMetas() -> [PrefixCachePolicy.Meta] {
        entries.map {
            PrefixCachePolicy.Meta(
                byteEstimate: $0.byteEstimate, lastUsed: $0.lastUsed,
                refcount: $0.refcount)
        }
    }

    /// Drop entries idle longer than `idleTTL` (skipping in-use ones). Lazy:
    /// driven from `acquire`/`store`/`stats`, so an idle pool drains without a
    /// background timer. No-op when `idleTTL == 0`. Decision is
    /// `PrefixCachePolicy.idleVictimIndices` (NC4); this applies it.
    private func sweepIdle(now: Date) {
        guard idleTTL > 0 else { return }
        let victims = Set(
            PrefixCachePolicy.idleVictimIndices(
                entryMetas(), now: now, idleTTL: idleTTL))
        guard !victims.isEmpty else { return }
        demoteVictims(victims, reason: .evict)  // ADR 027 — demote-to-disk, not drop
        let before = entries.count
        entries = entries.enumerated()
            .filter { !victims.contains($0.offset) }.map { $0.element }
        evictions += before - entries.count
        rotateKeyIfPoolEmptyLocked()
    }

    /// Evict LRU entries until under BOTH the count cap and the byte cap,
    /// skipping any a live generation still holds (refcount > 0). Bytes cap is
    /// inert when `maxBytes == 0`. Victim selection is
    /// `PrefixCachePolicy.lruVictimIndex` (NC4), looped one-at-a-time exactly
    /// as before.
    private func evictIfNeeded() {
        while let victimIdx = PrefixCachePolicy.lruVictimIndex(
            entryMetas(), maxEntries: maxEntries, maxBytes: maxBytes)
        {
            entries.remove(at: victimIdx)
            evictions += 1
        }
        rotateKeyIfPoolEmptyLocked()
    }

    /// Drop every entry not currently in use (refcount == 0). Returns the
    /// count freed. Used by the governor's pressure relief (M59.2) and the
    /// operator flush (M59.4). Entries held by an in-flight generation
    /// survive — they're freed when that request releases them.
    @discardableResult
    public func flushIdle(reason: KVSnapshotHeader.SaveReason = .evict) -> Int {
        lock.lock()
        defer { lock.unlock() }
        let victims = Set(PrefixCachePolicy.flushIdleIndices(entryMetas()))
        demoteVictims(victims, reason: reason)  // ADR 027 — spill to disk before dropping
        let before = entries.count
        entries = entries.enumerated()
            .filter { !victims.contains($0.offset) }.map { $0.element }
        let freed = before - entries.count
        evictions += freed
        rotateKeyIfPoolEmptyLocked()
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
