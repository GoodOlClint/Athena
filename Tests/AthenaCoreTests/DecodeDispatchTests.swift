import XCTest

@testable import AthenaCore

/// Issue #11 — `runSpeculative`'s routing decision, MLX-free and pinned
/// (ADR 008/009). The paths it names need a Metal device; the decision does not.
final class DecodeDispatchTests: XCTestCase {

    /// The defect this replaced: an unstructured, no-logprobs request was
    /// admitted past `container.prepare` whenever `speculative` was set, then
    /// turned away inside the closure — a wasted chat-template render and
    /// tokenize, plus the substrate's serial-container mutex, per request.
    func testUnstructuredNoLogprobsAlwaysDefers() {
        let route = DecodeDispatch.route(hasSchema: false, hasLogprobSink: false)
        XCTAssertEqual(route, .substrateStream)
        XCTAssertTrue(route.defersToSubstrateStream)
    }

    /// A schema must be masked, which the substrate stream cannot do.
    func testSchemaStaysForGuidedDecode() {
        let route = DecodeDispatch.route(hasSchema: true, hasLogprobSink: false)
        XCTAssertEqual(route, .guidedSubstrate)
        XCTAssertFalse(route.defersToSubstrateStream)
    }

    /// C2: `beginGeneration` has no logit-capture seam, so a logprobs request
    /// must not defer even with no schema.
    func testLogprobsStayEvenWithoutSchema() {
        let route = DecodeDispatch.route(hasSchema: false, hasLogprobSink: true)
        XCTAssertEqual(route, .logprobCapture)
        XCTAssertFalse(route.defersToSubstrateStream)
    }

    /// Schema wins when both are set — the Guide has to mask, and capture rides
    /// along on the same path.
    func testSchemaWinsOverLogprobs() {
        XCTAssertEqual(
            DecodeDispatch.route(hasSchema: true, hasLogprobSink: true),
            .guidedSubstrate)
    }

    /// The routing ignores `speculative`/`temperature` by construction: they are
    /// not parameters. Pinned as a signature-level fact, because reintroducing
    /// them is exactly how the stale `speculative-greedy`/`speculative-sampling`
    /// labels and the redundant prepare came about. Speculative is served by
    /// `beginGeneration`'s MTP drafter overload, which is where deferring sends
    /// it — so routing on it would be routing on the wrong thing.
    func testOnlyTwoInputsDecideTheRoute() {
        // Every combination of the two real inputs, exhaustively.
        let routes = [
            DecodeDispatch.route(hasSchema: false, hasLogprobSink: false),
            DecodeDispatch.route(hasSchema: false, hasLogprobSink: true),
            DecodeDispatch.route(hasSchema: true, hasLogprobSink: false),
            DecodeDispatch.route(hasSchema: true, hasLogprobSink: true),
        ]
        XCTAssertEqual(Set(routes).count, 3, "all three cases reachable")
        XCTAssertEqual(Set(DecodeDispatch.allCases), Set(routes))
    }

    /// Issue #42 — the effective-temperature rule has ONE home; these pin it.
    /// That home is `DecodeDispatch.effectiveTemperature`'s own doc comment,
    /// including the caveats documented there. Read the rule there — this
    /// header deliberately does not restate them, so it cannot drift from them.
    ///
    /// (#74: the signpost says only that caveats exist, not what they say. The
    /// earlier wording — "the caveat under which a negative override is NOT
    /// ignored" — carried a normative fragment that a fix to the narrowing
    /// would have falsified, which is the drift this header exists to avoid.
    /// #74 prescribed "a narrowing caveat", but that phrasing is still
    /// falsified by a fix that removes the caveat entirely, so this goes one
    /// step further and is caveat-count-neutral.) Each message below describes
    /// only the one input its own assertion passes, which is why the `-1`
    /// case can say "ignored" flatly: for that input it is, no caveat.
    func testEffectiveTemperature() {
        XCTAssertEqual(DecodeDispatch.effectiveTemperature(nil, 0.7), 0.7)
        XCTAssertEqual(DecodeDispatch.effectiveTemperature(0.3, 0.7), 0.3)
        XCTAssertEqual(
            DecodeDispatch.effectiveTemperature(0, 0.7), 0,
            "zero is an explicit greedy request, not an absent override")
        XCTAssertEqual(
            DecodeDispatch.effectiveTemperature(-1, 0.7), 0.7,
            "negative override is ignored — the loaded default applies")
        // Float-narrowing edges: the sign test runs AFTER Double→Float
        // (matching the decode sites), so these pin the exact boundary.
        XCTAssertEqual(
            DecodeDispatch.effectiveTemperature(.nan, 0.7), 0.7,
            "NaN fails >= 0 and falls back")
        XCTAssertEqual(
            DecodeDispatch.effectiveTemperature(-1e-45, 0.7), 0.7,
            "the OTHER side of the documented boundary: -1e-45 narrows to a "
                + "subnormal Float that is still negative, so it fails >= 0 and "
                + "is duly ignored. Without this, a shift in the underflow "
                + "threshold would only be caught on the -1e-60 side (#74)")
        XCTAssertEqual(
            DecodeDispatch.effectiveTemperature(-1e-60, 0.7).sign, .minus,
            "tiny negative underflows to -0.0, which passes >= 0 (still greedy "
                + "downstream: -0.0 compares equal to zero, and both samplers "
                + "select on == 0 / <= 0)"
        )
        XCTAssertEqual(
            DecodeDispatch.effectiveTemperature(1e60, 0.7), .infinity,
            "overflow saturates to +inf, matching what the decode already used")
    }

    /// Issue #72 — the decode rule and the admission rule DISAGREE, and the
    /// disagreement is observable as a status code, not a sign.
    ///
    /// `effectiveTemperature` narrows to `Float` before testing the sign, so
    /// `-1e-60` becomes `-0.0`, passes `>= 0`, and decodes greedily — the same
    /// arm an explicit `0` takes. `decodesDeterministically` tests the raw
    /// `Double`, where `-1e-60 != 0`. So a `logprobs` request at `-1e-60` is
    /// refused 400 `logprobs_requires_deterministic` while the identical body
    /// at `0` is accepted, even though both would have decoded identically.
    ///
    /// This is the pin for the doc correction: the ADR-013 §4 C2 gate is what
    /// the disagreement flows into, and the equivalence `effectiveTemperature`
    /// documents is scoped to decode alone.
    ///
    /// **What this test does and does not guard.** It fails if the narrowing
    /// moves INTO `decodesDeterministically`. It does NOT catch #72 direction 2
    /// implemented where the issue actually locates it — at the call site,
    /// narrowing `body.temperature` before handing it over — because the
    /// predicate would then still be given a Double that is literally 0. That
    /// wiring is unpinnable from here: `ChatCompletionRequest` is internal to
    /// the executable module. `deploy/e2e-rbac.sh` carries the status-code pin
    /// for it (a `-1e-60` + logprobs 400 beside a `temperature:0` 200), which
    /// is #72's acceptance criterion at the HTTP layer. That gate is
    /// operator-run, not CI, which is why this unit pin exists as well rather
    /// than instead.
    func testUnderflowingNegativeIsNotAdmissionZero() {
        // Decode: indistinguishable from an explicit 0 — both greedy.
        XCTAssertEqual(DecodeDispatch.effectiveTemperature(-1e-60, 0.7), 0)
        XCTAssertEqual(DecodeDispatch.effectiveTemperature(0, 0.7), 0)

        // Admission: NOT indistinguishable. This is the whole finding.
        XCTAssertFalse(
            DecodeDispatch.decodesDeterministically(
                rawTemperature: -1e-60, hasSchema: false),
            "raw Double: -1e-60 != 0, so a logprobs request here is a 400")

        // The class is "narrows to ±0 without being literally 0", so the
        // POSITIVE underflows diverge identically. Pinned because the doc now
        // says so, and because a reader given only the negative case would
        // reasonably assume the sign is what matters.
        XCTAssertEqual(DecodeDispatch.effectiveTemperature(1e-60, 0.7), 0)
        XCTAssertFalse(
            DecodeDispatch.decodesDeterministically(
                rawTemperature: 1e-60, hasSchema: false),
            "positive underflow decodes greedy but is admission-nonzero too")
        XCTAssertFalse(
            DecodeDispatch.decodesDeterministically(
                rawTemperature: 5e-324, hasSchema: false),
            "the smallest subnormal Double narrows to +0.0, same divergence")
        XCTAssertTrue(
            DecodeDispatch.decodesDeterministically(
                rawTemperature: 0, hasSchema: false),
            "the identical body at temperature 0 is admitted")

        // A schema admits regardless of temperature — inert under a Guide.
        XCTAssertTrue(
            DecodeDispatch.decodesDeterministically(
                rawTemperature: 0.9, hasSchema: true))
        XCTAssertFalse(
            DecodeDispatch.decodesDeterministically(
                rawTemperature: 0.9, hasSchema: false))
        // Absent override is not zero: it means "use the loaded default",
        // which may well be a sampling temperature.
        XCTAssertFalse(
            DecodeDispatch.decodesDeterministically(
                rawTemperature: nil, hasSchema: false),
            "nil is an absent override, not an explicit greedy request")
    }

    /// The `path=` values are an operator-facing contract: they are what a
    /// `dispatch path=` log line reports, so an operator greps for them.
    func testRawValuesAreTheLoggedNames() {
        XCTAssertEqual(DecodeDispatch.substrateStream.rawValue, "substrate-stream")
        XCTAssertEqual(DecodeDispatch.guidedSubstrate.rawValue, "guided-substrate")
        XCTAssertEqual(DecodeDispatch.logprobCapture.rawValue, "logprob-capture")
        // The removed labels named in-closure branches publication S0 deleted.
        for stale in ["speculative-greedy", "speculative-sampling"] {
            XCTAssertFalse(
                DecodeDispatch.allCases.map(\.rawValue).contains(stale),
                "\(stale) names a decode path that no longer exists")
        }
    }
}
