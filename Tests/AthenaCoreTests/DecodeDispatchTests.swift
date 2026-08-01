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

    /// The `path=` values are an operator-facing contract — they appear in
    /// `dispatch path=` log lines and in `docs/logging.md` grep recipes.
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
