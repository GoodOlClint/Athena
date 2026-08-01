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
    /// including the narrowing caveat under which a negative override is NOT
    /// ignored. Read the rule there — this header deliberately does not
    /// restate it, so it cannot drift from it. Each message below describes
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
            DecodeDispatch.effectiveTemperature(-1e-60, 0.7).sign, .minus,
            "tiny negative underflows to -0.0, which passes >= 0 (still greedy "
                + "downstream: -0.0 compares equal to zero, and both samplers "
                + "select on == 0 / <= 0)"
        )
        XCTAssertEqual(
            DecodeDispatch.effectiveTemperature(1e60, 0.7), .infinity,
            "overflow saturates to +inf, matching what the decode already used")
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
