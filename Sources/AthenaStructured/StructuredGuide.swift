import CAthenaStructured
import Foundation

public struct StructuredError: Error, CustomStringConvertible {
    public let message: String
    init(_ stage: String) {
        let detail = StructuredShim.lastError()
        message = detail.isEmpty ? stage : "\(stage): \(detail)"
    }
    public var description: String { message }
}

/// A token's id and the decoded bytes the model emits for it. The caller
/// builds these from the *model's own* tokenizer so the byte mapping
/// matches exactly (mismatch ⇒ outlines-core `IncompatibleVocabulary`).
public struct VocabToken: Sendable {
    public let id: UInt32
    public let bytes: [UInt8]
    public init(id: UInt32, bytes: [UInt8]) {
        self.id = id
        self.bytes = bytes
    }
}

/// Owned outlines-core vocabulary. Reference type: frees the Rust handle
/// on deinit. Not Sendable — confine to one isolation domain.
public final class StructuredVocabulary {
    let ptr: OpaquePointer

    public init(tokens: [VocabToken], eosTokenId: UInt32) throws {
        let ids = tokens.map(\.id)
        let lens = tokens.map { $0.bytes.count }
        var blob: [UInt8] = []
        var offsets: [Int] = []
        offsets.reserveCapacity(tokens.count)
        for t in tokens {
            offsets.append(blob.count)
            blob.append(contentsOf: t.bytes)
        }
        if blob.isEmpty { blob = [0] }  // ensure a valid base address

        let handle: OpaquePointer? = ids.withUnsafeBufferPointer { idp in
            lens.withUnsafeBufferPointer { lp in
                blob.withUnsafeBufferPointer { bp in
                    let base = bp.baseAddress!
                    var ptrs: [UnsafePointer<UInt8>?] =
                        offsets.map { base + $0 }
                    return ptrs.withUnsafeMutableBufferPointer { pp in
                        oc_vocab_new_from_tokens(
                            idp.baseAddress, pp.baseAddress, lp.baseAddress,
                            tokens.count, eosTokenId)
                    }
                }
            }
        }
        guard let handle else {
            throw StructuredError("vocab build")
        }
        ptr = handle
    }

    deinit { oc_vocab_free(ptr) }

    /// Maps a whitespace-prefixed JSON-opener token id → the bare opener
    /// token id with the same bytes minus leading ASCII whitespace
    /// (` {` → `{`, ` {"` → `{"`). Schema-independent — depends only on
    /// the model vocab, so build once per model. Lets deferred
    /// enforcement honor the model's opener when it is emitted as a
    /// single space-prefixed token: outlines-core anchors the JSON root
    /// at `{`/`[` with no leading whitespace, so the raw token has no
    /// start transition and the real opener would otherwise be dropped
    /// (issue #2). Restricted to `{`/`[` openers so the result stays
    /// tiny and the intent is unambiguous.
    public static func openerAliases(
        tokens: [VocabToken]
    ) -> [UInt32: UInt32] {
        var byBytes: [[UInt8]: UInt32] = [:]
        byBytes.reserveCapacity(tokens.count)
        for t in tokens { byBytes[t.bytes] = t.id }
        func isWS(_ b: UInt8) -> Bool {
            b == 0x20 || b == 0x09 || b == 0x0A || b == 0x0D
        }
        var alias: [UInt32: UInt32] = [:]
        for t in tokens {
            let b = t.bytes
            var i = 0
            while i < b.count, isWS(b[i]) { i += 1 }
            guard i > 0, i < b.count, b[i] == 0x7B || b[i] == 0x5B
            else { continue }
            if let inner = byBytes[Array(b[i...])], inner != t.id {
                alias[t.id] = inner
            }
        }
        return alias
    }
}

/// Owned compiled DFA index (from a regex or a JSON schema).
///
/// `@unchecked Sendable` because the compiled DFA is immutable
/// post-construction — `oc_index_from_schema` / `oc_index_from_regex`
/// produce a read-only structure that the per-request `StructuredGuide`
/// walks with its own state. No method on `StructuredIndex` mutates
/// the underlying outlines-core data; the only operation is
/// `oc_guide_new(index.ptr)` which copies the relevant DFA edges into
/// a fresh walker. Safe to share across isolation domains — and M49.1
/// requires it so the compiled DFA can be cached on `MLXLLMModule`
/// and reused across requests without recompiling on every call.
public final class StructuredIndex: @unchecked Sendable {
    let ptr: OpaquePointer
    let vocab: StructuredVocabulary  // keep alive

    public init(regex: String, vocabulary: StructuredVocabulary) throws {
        guard let p = regex.withCString({ oc_index_from_regex($0, vocabulary.ptr) })
        else { throw StructuredError("index from regex") }
        ptr = p
        vocab = vocabulary
    }

    public init(
        jsonSchema: String, whitespace: String? = nil,
        vocabulary: StructuredVocabulary
    ) throws {
        let p: OpaquePointer? = jsonSchema.withCString { js in
            if let ws = whitespace {
                return ws.withCString {
                    oc_index_from_schema(js, $0, vocabulary.ptr)
                }
            }
            return oc_index_from_schema(js, nil, vocabulary.ptr)
        }
        guard let p else { throw StructuredError("index from schema") }
        ptr = p
        vocab = vocabulary
    }

    deinit { oc_index_free(ptr) }
}

/// Stateful walker over a `StructuredIndex`: per-step allowed-token
/// bitmask, `advance`, and bounded `rollback` (the 32-entry ring in the
/// Rust shim) — the structured-decoding counterpart to the MTP
/// speculative loop's KV/Mamba rollback.
public final class StructuredGuide {
    let ptr: OpaquePointer
    let index: StructuredIndex  // keep alive

    /// Whitespace-prefixed-opener tolerance for the IDLE→ENFORCING
    /// probe (issue #2). Set by the caller from the model vocab via
    /// `StructuredVocabulary.openerAliases`; empty ⇒ plain advance.
    public var openerAlias: [UInt32: UInt32] = [:]

    public init(index: StructuredIndex) throws {
        guard let p = oc_guide_new(index.ptr) else {
            throw StructuredError("guide new")
        }
        ptr = p
        self.index = index
    }

    deinit { oc_guide_free(ptr) }

    /// Bitmask length in bytes (`ceil(vocabSize / 8)`).
    public var maskLength: Int { oc_guide_mask_len(ptr) }

    /// Fill `buffer` (≥ `maskLength`) with allowed tokens: bit `i` of
    /// byte `i>>3` set ⇒ token `i` permitted from the current state.
    public func allowedMask(into buffer: inout [UInt8]) -> Bool {
        if buffer.count < maskLength { buffer = [UInt8](repeating: 0, count: maskLength) }
        return buffer.withUnsafeMutableBufferPointer {
            oc_guide_allowed_mask(ptr, $0.baseAddress, $0.count) == 0
        }
    }

    /// Advance on a committed token; false ⇒ no FSM transition (caller
    /// should not record it — mirrors mlx-lm `_try_advance`).
    @discardableResult
    public func advance(_ token: UInt32) -> Bool {
        oc_guide_advance(ptr, token)
    }

    /// Like `advance`, but if `token` has no transition AND it is a
    /// whitespace-prefixed JSON opener, retry with the bare opener id so
    /// the model's space-prefixed `{` is honored at the IDLE→ENFORCING
    /// boundary instead of dropped (issue #2). A failed `advance` does
    /// not mutate FSM state, so the fallback is safe. Only the
    /// opener-boundary caller should use this.
    @discardableResult
    public func advanceOpenerTolerant(_ token: UInt32) -> Bool {
        if advance(token) { return true }
        if let bare = openerAlias[token] { return advance(bare) }
        return false
    }

    public var isFinal: Bool { oc_guide_is_final(ptr) }
    public var allowedRollback: Int { oc_guide_allowed_rollback(ptr) }

    /// Roll back `n` advances; false ⇒ `n` exceeds the 32-entry ring
    /// (caller must rebuild a fresh guide and replay — see the M3.3
    /// `_sync` port).
    @discardableResult
    public func rollback(_ n: Int) -> Bool {
        oc_guide_rollback(ptr, n)
    }
}
