import AthenaStructured
import Foundation
import MLX

/// Deferred structured enforcement with an idle budget (operator
/// decisions: "phase flag in our loop" + "idle budget then force").
///
/// IDLE: the model generates unconstrained (it may emit a
/// `<think>…</think>` / tool preamble); each committed token probes the
/// Guide via the shim's `advance` (which only mutates on a valid
/// transition). The first token the Guide accepts — the JSON value's
/// opener — flips to ENFORCING. To guarantee non-empty schema-valid
/// output even when the model never voluntarily emits JSON (terse
/// prompt / small model / EOS in IDLE), enforcement is FORCED once
/// `idleBudget` non-JSON tokens have passed (or on `forceEnforce()`,
/// called by the loop when the model tries to stop in IDLE).
///
/// ENFORCING: picks are Guide-masked and every commit advances the
/// Guide (monotonic — still no rollback). The structured response is the
/// JSON span only (`.jsonStart` onward); the IDLE prefix is dropped, so
/// `response_format` stays spec-compliant even with a thinking prefix.
enum CommitResult {
    case unconstrained  // no guide — keep everything
    case idlePrefix  // pre-JSON, dropped from the structured response
    case jsonStart  // first enforced token (JSON span begins here)
    case jsonBody  // subsequent enforced token
}

struct GuidedDecoder {
    let guide: StructuredGuide?
    let vocab: Int
    let idleBudget: Int
    private(set) var enforcing: Bool
    private(set) var jsonStarted = false
    private var firstJSONRecorded = false
    private var idleCount = 0
    private var maskBuf: [UInt8] = []

    init(guide: StructuredGuide?, vocab: Int, idleBudget: Int) {
        self.guide = guide
        self.vocab = vocab
        self.idleBudget = max(1, idleBudget)
        self.enforcing = false
    }

    /// Pick a token id from a `(1, vocab)` logits slice.
    mutating func pick(_ slice: MLXArray) -> Int {
        if let guide, enforcing {
            return SpeculativeGeneration.guidedArgmax(
                slice, vocab: vocab, guide: guide, maskBuf: &maskBuf)
        }
        return argMax(slice, axis: -1).item(Int.self)
    }

    /// Force ENFORCING without consuming a token — called by the loop
    /// when the model emits EOS while still in IDLE (so we constrain
    /// rather than return empty).
    mutating func forceEnforce() {
        if guide != nil, !enforcing {
            enforcing = true
            jsonStarted = true
        }
    }

    /// Record a committed token; advance the Guide / drive the phase.
    mutating func commit(_ token: Int) -> CommitResult {
        guard let guide else { return .unconstrained }
        if enforcing {
            _ = guide.advance(UInt32(token))
            if !firstJSONRecorded {
                firstJSONRecorded = true
                return .jsonStart
            }
            return .jsonBody
        }
        // IDLE: the JSON opener is the first token the fresh Guide
        // takes — tolerant of a single space-prefixed opener token
        // (` {`), which outlines-core's `{`-anchored root would
        // otherwise reject, dropping the real opener (issue #2).
        if guide.advanceOpenerTolerant(UInt32(token)) {
            enforcing = true
            jsonStarted = true
            firstJSONRecorded = true
            return .jsonStart
        }
        idleCount += 1
        if idleCount >= idleBudget {
            enforcing = true  // force: next pick is Guide-masked
            jsonStarted = true
        }
        return .idlePrefix
    }
}
