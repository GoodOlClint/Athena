import AthenaCore
import Foundation
import MLX

/// ADR 024 Tier 3 — the MLX bridge of the prompt-cache KV serialize/deserialize
/// seam: `[MLXArray]` ↔ `Data`, via the MLX-free `KVFrame` wire format.
///
/// This is the only MLX-touching part of T3, so it stays in `AthenaLLM` and is
/// validated by the **bit-identical gate** (cold-vs-warm on a real model,
/// `deploy/integration/e2e-m59-prefix-cache.sh`) rather than `swift test` — no
/// `MLXArray` can be evaluated under `swift test` (ADR 009). The framing layout
/// and its malformed-input rejection are unit-pinned over in `KVFrame`.
///
/// Round-trip is **bit-preserving**: `asData(access: .copy)` returns the
/// contiguous native-dtype bytes, and `MLXArray(_:_:dtype:)` rebuilds the same
/// shape+dtype contiguously — so a sealed→opened→decoded entry reproduces the
/// exact KV bytes, which is what keeps prefix reuse byte-identical to a cold
/// prefill once AES-GCM (lossless) is in the path.
enum KVByteCodec {

    enum Failure: Error, Equatable {
        case unknownDtypeCode(UInt32)
        case unsupportedDtype(String)
    }

    /// Serialize a cached `[MLXArray]` (a `KVCacheSimple.state` pair, or a
    /// recurrent checkpoint's `[conv, ssm]`) to a single `Data` for sealing.
    static func encode(_ arrays: [MLXArray]) -> Data {
        let slots = arrays.map { array -> KVFrame.Slot in
            let backing = array.asData(access: .copy)
            return KVFrame.Slot(
                shape: backing.shape,
                dtypeCode: code(for: backing.dType),
                bytes: backing.data)
        }
        return KVFrame.encode(slots)
    }

    /// Reconstruct the `[MLXArray]` a prior `encode` produced (after the cipher
    /// has opened the buffer). Throws on an unknown dtype code; the buffer is
    /// otherwise authentic (GCM-verified) so shapes/lengths are consistent.
    static func decode(_ data: Data) throws -> [MLXArray] {
        let slots = try KVFrame.decode(data)
        return try slots.map { slot in
            MLXArray(slot.bytes, slot.shape, dtype: try dtype(for: slot.dtypeCode))
        }
    }

    // MARK: - Stable dtype ↔ code mapping

    /// Fixed codes pinned HERE (not `DType`'s internal representation, which
    /// could drift across substrate bumps) so a frame written by one build
    /// always decodes the same in another.
    static func code(for dtype: DType) -> UInt32 {
        switch dtype {
        case .bool: return 0
        case .uint8: return 1
        case .uint16: return 2
        case .uint32: return 3
        case .uint64: return 4
        case .int8: return 5
        case .int16: return 6
        case .int32: return 7
        case .int64: return 8
        case .float16: return 9
        case .float32: return 10
        case .bfloat16: return 11
        case .complex64: return 12
        case .float64: return 13
        }
    }

    static func dtype(for code: UInt32) throws -> DType {
        switch code {
        case 0: return .bool
        case 1: return .uint8
        case 2: return .uint16
        case 3: return .uint32
        case 4: return .uint64
        case 5: return .int8
        case 6: return .int16
        case 7: return .int32
        case 8: return .int64
        case 9: return .float16
        case 10: return .float32
        case 11: return .bfloat16
        case 12: return .complex64
        case 13: return .float64
        default: throw Failure.unknownDtypeCode(code)
        }
    }
}
