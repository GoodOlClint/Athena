import AthenaCore
import Foundation
import MLXLMCommon

/// ADR 039 S2 — continuous batching over the substrate `BatchGenerator`, minimal
/// core: **fixed-batch** (drain queue → admit a batch → drive to completion →
/// next batch), **default-off**, plain-text chat only. The ADR-039 "join/leave"
/// mid-flight scheduler is the deferred polish.
///
/// Composes with ADR 011/029/023: the worker holds **one** `InferenceGate` span
/// for the batch (one batched `next()` = one eval graph), the resident model is
/// the only model batched (a different-model request takes the serial path,
/// which rebinds under the same gate — so the gate *is* the rebind barrier), and
/// every admitted row reserves worst-case KV against the ADR-023 truthful budget
/// (`SequenceKVLedger`, S1). The worker itself lives on `MLXLLMModule` (it needs
/// the resident container); this enum is just the boot-set knob + provider.
public enum BatchScheduler {
    /// Default-off revert knob (boot-set from TOML `batching_enabled` + env,
    /// mirroring `InferenceGate.enabled`). Write-once at boot before any request.
    public nonisolated(unsafe) static var enabled = false

    /// Boot-set hook returning the governor's live admission inputs. Set in
    /// `Load.run()` to `{ await governor.admissionInputs() }`; nil ⇒ admit freely
    /// (only in a governor-less test/dev context — never the real daemon).
    public nonisolated(unsafe)
        static var admissionInputsProvider:
            (@Sendable () async -> (denominator: Int, budget: Int))?

    static func admissionInputs() async -> (denominator: Int, budget: Int) {
        if let p = admissionInputsProvider { return await p() }
        return (0, .max)
    }
}

/// One queued batchable request: its prompt tokens, decode params, worst-case KV
/// reservation, and the output stream continuation the worker fans tokens into.
public struct BatchPending: Sendable {
    let uid: Int
    let promptTokens: [Int]
    let promptCount: Int
    let maxTokens: Int
    let sampler: RowSampler
    let kvBytes: Int
    let continuation: AsyncStream<GenChunk>.Continuation
}
