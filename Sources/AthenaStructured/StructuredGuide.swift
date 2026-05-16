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
}

/// Owned compiled DFA index (from a regex or a JSON schema).
public final class StructuredIndex {
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
