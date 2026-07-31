import Foundation
import XCTest

@testable import AthenaCore

/// Pins the float-faithful `config.json` emitter (ADR 016). MLX-free (ADR
/// 008/009): the whole point is that an `athena convert` rewrite preserves the
/// int/float distinction so the output loads in upstream mlx-lm/transformers,
/// whose strict-config validation rejects an int where a `float` is declared
/// (the gemma4 `final_logit_softcapping: 30.0` field incident, 2026-06-29).
final class ConfigJSONEmitTests: XCTestCase {

    /// Round-trip a JSON string through parse → emit and return the emitted text.
    private func roundTrip(_ json: String) throws -> String {
        let obj =
            try JSONSerialization.jsonObject(with: Data(json.utf8))
            as! [String: Any]
        return String(decoding: try ConfigJSONEmit.data(from: obj), as: UTF8.self)
    }

    func testWholeFloatStaysFloat() throws {
        // The actual failure: 30.0 must NOT become 30.
        let out = try roundTrip(#"{"final_logit_softcapping": 30.0}"#)
        XCTAssertTrue(
            out.contains("\"final_logit_softcapping\": 30.0"),
            "whole float collapsed to int:\n\(out)")
    }

    func testIntStaysInt() throws {
        let out = try roundTrip(#"{"num_hidden_layers": 48}"#)
        XCTAssertTrue(out.contains("\"num_hidden_layers\": 48"))
        XCTAssertFalse(out.contains("48.0"))
    }

    func testLargeWholeFloatAndFractionalFloat() throws {
        let out = try roundTrip(#"{"rope_theta": 1000000.0, "rms_norm_eps": 0.000001}"#)
        XCTAssertTrue(out.contains("\"rope_theta\": 1000000.0"), out)
        // Fractional floats survive (exact textual repr is round-trippable).
        XCTAssertTrue(out.contains("\"rms_norm_eps\": 1e-06"), out)
    }

    func testBoolNotCollapsedToNumber() throws {
        let out = try roundTrip(#"{"tie_word_embeddings": true, "use_cache": false}"#)
        XCTAssertTrue(out.contains("\"tie_word_embeddings\": true"), out)
        XCTAssertTrue(out.contains("\"use_cache\": false"), out)
    }

    func testNullAndNestedAndArrays() throws {
        let out = try roundTrip(
            #"{"text_config": {"head_dim": 256, "attn_logit_softcapping": 50.0}, "x": null, "layer_types": ["full", "sliding"]}"#
        )
        XCTAssertTrue(out.contains("\"attn_logit_softcapping\": 50.0"), out)
        XCTAssertTrue(out.contains("\"head_dim\": 256"), out)
        XCTAssertTrue(out.contains("\"x\": null"), out)
        XCTAssertTrue(out.contains("\"full\""), out)
        // Re-parse to prove it's still valid JSON.
        XCTAssertNoThrow(
            try JSONSerialization.jsonObject(with: Data(out.utf8)))
    }

    func testStringEscaping() throws {
        let out = try roundTrip(#"{"k": "a\"b\\c\n"}"#)
        let reparsed =
            try JSONSerialization.jsonObject(with: Data(out.utf8))
            as! [String: Any]
        XCTAssertEqual(reparsed["k"] as? String, "a\"b\\c\n")
    }

    func testSortedKeysStableOrder() throws {
        let out = try roundTrip(#"{"b": 1, "a": 2}"#)
        XCTAssertLessThan(
            out.range(of: "\"a\"")!.lowerBound,
            out.range(of: "\"b\"")!.lowerBound)
    }
}
