/// Which decode path a chat request takes out of `MLXLLMModule.runSpeculative`.
///
/// MLX-free and unit-pinned (ADR 008/009): the routing decision is pure
/// predicate logic over two request properties, so it is testable without a
/// Metal device even though the paths it names are not.
///
/// `runSpeculative` exists to serve the two cases the substrate's own stream
/// cannot: masking to a schema, and capturing logits. Everything else — including
/// every speculative request — belongs on the substrate stream in
/// `beginGeneration`, which is where the ADR 032 MTP separate-drafter overload
/// lives.
///
/// This used to be inferred from `greedyEligible`/`samplingEligible`, which
/// consulted `speculative` and `temperature`. Those named in-closure decode
/// branches (M40.2/M40.3 sampling-mode, the Qwen3.5 vendored fork) that
/// publication S0 deleted, so a speculative unstructured request was admitted
/// past `container.prepare` only to be turned away inside the closure — paying a
/// full chat-template render and tokenize, plus the substrate's
/// `SerialAccessContainer` mutex, for work that was thrown away — and the
/// `dispatch path=` line told the operator it had taken a path that no longer
/// existed.
public enum DecodeDispatch: String, Sendable, CaseIterable {
    /// Return nil from `runSpeculative`; `beginGeneration` serves it (plain
    /// stream, or the MTP drafter overload when one is resident).
    case substrateStream = "substrate-stream"
    /// Schema present: decode under the structured Guide's mask.
    case guidedSubstrate = "guided-substrate"
    /// No schema but logprobs requested: plain decode with logit capture, which
    /// `beginGeneration` has no seam for.
    case logprobCapture = "logprob-capture"

    /// Route on what the request actually needs, NOT on `speculative`/
    /// `temperature` — neither changes which code serves it.
    public static func route(
        hasSchema: Bool, hasLogprobSink: Bool
    ) -> DecodeDispatch {
        if hasSchema { return .guidedSubstrate }
        if hasLogprobSink { return .logprobCapture }
        return .substrateStream
    }

    /// True when `runSpeculative` should hand the request straight back so the
    /// substrate stream serves it — checked *before* any prompt preparation.
    public var defersToSubstrateStream: Bool { self == .substrateStream }

    /// The temperature a request decodes at: a non-negative per-request
    /// override wins (zero = explicit greedy), a negative override is
    /// ignored, nil falls back to the loaded default. The ONE home for this
    /// rule — `runSpeculative` logs it and `beginGeneration`/the batch path
    /// decode at it, so on the substrate-stream path the `dispatch path=`
    /// `temp=` field equals the decoded value by construction, not by
    /// copy-paste agreement. Under a Guide the decode is masked-argmax and
    /// temperature is inert (M48.3) — the field then reports what the request
    /// *asked for*, not a knob the guided path consults.
    public static func effectiveTemperature(
        _ override: Double?, _ fallback: Float
    ) -> Float {
        override.map { Float($0) }.flatMap { $0 >= 0 ? $0 : nil } ?? fallback
    }
}
