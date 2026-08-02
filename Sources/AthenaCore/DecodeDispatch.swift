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
    /// ignored (one narrowing caveat below), nil falls back to the loaded
    /// default. The ONE home for this rule — `runSpeculative` logs it and
    /// `beginGeneration`/the batch path decode at it, so on the
    /// substrate-stream path the `dispatch path=` `temp=` field equals the
    /// decoded value by construction, not by copy-paste agreement. Under a
    /// Guide the decode is masked-argmax and
    /// temperature is inert (M48.3) — the field then reports what the request
    /// *asked for*, not a knob the guided path consults.
    ///
    /// The caveat: the sign is tested after narrowing to `Float` (the decode's
    /// own parameter type), so a negative that *underflows* — `-1e-60`, though
    /// not `-1e-45`, which stays subnormal and is duly ignored — becomes
    /// `-0.0`, passes `>= 0`, and is returned.
    ///
    /// **Scope of that equivalence (#72): it holds for DECODE, not for
    /// ADMISSION.** For decode it is equivalent to an explicit `0` override,
    /// NOT to the fallback the ignore branch would have applied — `-0.0`
    /// compares equal to zero, so sampler selection takes the greedy arm and
    /// only the returned value's sign differs. Admission is a *different*
    /// predicate on the *un-narrowed* `Double`: see
    /// `decodesDeterministically(rawTemperature:hasSchema:)` below, for which
    /// `-1e-60` is NOT zero. So `{"temperature": -1e-60, "logprobs": true}` is
    /// rejected 400 `logprobs_requires_deterministic` where the identical body
    /// with `temperature: 0` is accepted — the two differ by a status code, not
    /// merely a sign.
    ///
    /// The class is wider than the sign suggests: it is **any override that
    /// narrows to `±0` without being literally `0`**, so the positive
    /// underflows behave identically — `1e-60` and `5e-324` also decode greedy
    /// (`Float` gives `+0.0`, and every decode consumer takes the greedy arm at
    /// `temperature <= 0`) while being admission-nonzero. Negatives are merely
    /// the case the caveat above was written about.
    ///
    /// `testEffectiveTemperature` pins the narrowing edges (`.nan` falls back,
    /// `1e60` saturates to `+inf`, `-1e-45` stays subnormal and is ignored);
    /// `testUnderflowingNegativeIsNotAdmissionZero` pins the divergence itself,
    /// on both signs.
    public static func effectiveTemperature(
        _ override: Double?, _ fallback: Float
    ) -> Float {
        override.map { Float($0) }.flatMap { $0 >= 0 ? $0 : nil } ?? fallback
    }

    /// Whether a request decodes deterministically, for ADMISSION purposes —
    /// the C2 gate (ADR 013 §4) that decides whether `logprobs` is honored or
    /// 400s, since only a deterministic path has a logit-capture seam.
    ///
    /// **`rawTemperature` is the un-narrowed `Double` on purpose**, and that is
    /// the whole reason this is its own function rather than a reuse of
    /// `effectiveTemperature`: the two rules genuinely disagree, and the
    /// disagreement is observable as a status code (#72). Narrowing here would
    /// turn a request that is currently a 400 into a 200 — a deliberate API
    /// change, not a tidy-up, and explicitly not what was chosen. Structured
    /// requests are deterministic regardless, because temperature is inert
    /// under a Guide (M48.3).
    ///
    /// Extracted from `AthenaServer+ChatOpenAI` verbatim so the rule is
    /// unit-testable, and so it sits beside the decode rule it disagrees with
    /// instead of a file away. ADR 008 chose extraction over
    /// `@testable import athena` — SE-0294 permits the latter, so this is a
    /// cost decision, not an impossibility.
    public static func decodesDeterministically(
        rawTemperature: Double?, hasSchema: Bool
    ) -> Bool {
        (rawTemperature == 0) || hasSchema
    }
}
