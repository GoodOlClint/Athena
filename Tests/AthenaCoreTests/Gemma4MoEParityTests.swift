import Foundation
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import Tokenizers
import XCTest

@testable import AthenaModels

/// M64.2 — Gemma4 MoE forward parity.
///
/// Loads a Gemma4 MoE target (gemma-4-26b-a4b-it-4bit: 128 experts, top-8,
/// mixed 4/8-bit quant) and asserts the substrate forward's softcapped logits
/// match the upstream mlx-lm Python reference (the
/// `Fixtures/gemma4_moe_parity.safetensors`, written by
/// `deploy/gemma4-moe/gen_parity_fixture.py`) over a fixed short token
/// sequence. The logits are the end-to-end product of the whole MoE forward
/// (router top-k + expert gather-matmul + the three extra norms + the hybrid
/// sum + the per-tensor quant), so a high-cosine / matching-argmax assertion is
/// the MoE-numerics correctness gate. Confirming the checkpoint loads at all
/// also empirically validates the mixed per-tensor 4/8-bit quantization path.
final class Gemma4MoEParityTests: XCTestCase {

    private func requireModelTests() throws {
        guard ProcessInfo.processInfo.environment["ATHENA_RUN_MODEL_TESTS"] == "1"
        else { throw XCTSkip("set ATHENA_RUN_MODEL_TESTS=1 (needs the metallib)") }
    }

    private func modelDir() -> URL? {
        let env = ProcessInfo.processInfo.environment
        if let p = env["ATHENA_TEST_GEMMA4_MOE_DIR"], !p.isEmpty {
            return URL(fileURLWithPath: p)
        }
        let home = FileManager.default.homeDirectoryForCurrentUser
        let snaps = home.appending(
            path:
                ".cache/huggingface/hub/models--mlx-community--gemma-4-26b-a4b-it-4bit/snapshots")
        if let entries = try? FileManager.default.contentsOfDirectory(
            at: snaps, includingPropertiesForKeys: nil),
            let first = entries.first(where: {
                FileManager.default.fileExists(
                    atPath: $0.appending(component: "config.json").path)
            })
        { return first }
        let store = home.appending(path: ".athena/models/gemma-4-26b-a4b-it-4bit")
        if FileManager.default.fileExists(
            atPath: store.appending(component: "config.json").path)
        { return store }
        return nil
    }

    private func fixtureURL() -> URL {
        if let p = ProcessInfo.processInfo.environment["ATHENA_TEST_GEMMA4_MOE_FIXTURE"],
            !p.isEmpty
        {
            return URL(fileURLWithPath: p)
        }
        return URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .appending(path: "Fixtures/gemma4_moe_parity.safetensors")
    }

    func testForwardMatchesPythonReference() async throws {
        try requireModelTests()
        guard let dir = modelDir() else {
            throw XCTSkip("no gemma-4-26b-a4b-it-4bit checkpoint present")
        }
        let fixture = fixtureURL()
        guard FileManager.default.fileExists(atPath: fixture.path) else {
            throw XCTSkip(
                "no parity fixture — run deploy/gemma4-moe/gen_parity_fixture.py")
        }

        let arrays = try MLX.loadArrays(url: fixture)
        // Extract to plain Sendable values — the `perform` closure is
        // @Sendable and cannot capture non-Sendable MLXArrays (M23 gotcha);
        // rebuild the arrays inside the closure.
        let tokenInts = try XCTUnwrap(arrays["token_ids"]).asType(.int32)
            .asArray(Int32.self)
        let expectedFloats = try XCTUnwrap(arrays["logits_last"]).asType(.float32)
            .reshaped([-1]).asArray(Float.self)
        let expectedArgmax = try XCTUnwrap(arrays["argmax_last"]).reshaped([-1])
            .item(Int.self)
        // Per-position reference greedy argmax (the decisive signal).
        let expectedArgmaxAll = try XCTUnwrap(arrays["argmax_all"]).asType(.int32)
            .reshaped([-1]).asArray(Int32.self)

        let container = try await loadModelContainer(
            from: dir.resolvingSymlinksInPath(),
            using: #huggingFaceTokenizerLoader())

        try await container.perform { (ctx: ModelContext) in
            let L = tokenInts.count
            let inputs = MLXArray(tokenInts, [1, L])
            let expectedLast = MLXArray(expectedFloats)

            let logits: MLXArray
            if let g = ctx.model as? Gemma4Model {
                logits = g(inputs, cache: g.newCache(parameters: nil))
            } else if let g = ctx.model as? Gemma4TextModel {
                logits = g(inputs, cache: g.newCache(parameters: nil))
            } else {
                throw XCTSkip(
                    "loaded model is \(type(of: ctx.model)), not a Gemma4 target")
            }

            let last = logits[0, L - 1].asType(.float32)  // (vocab,)
            // Per-position greedy argmax over the whole sequence.
            let argmaxAll = argMax(logits[0], axis: -1).asType(.int32)  // (L,)
            eval(last, argmaxAll)

            XCTAssertEqual(
                last.shape, expectedLast.shape, "logits shape mismatch")

            // PRIMARY GATE — per-position greedy argmax agreement, with each
            // mismatch verified to be a genuine bf16 NEAR-TIE rather than a
            // real divergence (mirrors ADR 001's kernel-tie verification). Two
            // correct independent ports of the same arch disagree only where
            // the top-2 logits are within bf16 rounding: at such a position the
            // Swift margin (swift-token logit − reference-token logit) is tiny.
            // A real routing/expert/norm bug would shift the argmax by a wide
            // margin. The validated dense Gemma4 path sets the baseline (it too
            // has rare near-tie mismatches vs this same oracle).
            let swiftArgmaxAll = argmaxAll.asArray(Int32.self)
            XCTAssertEqual(
                swiftArgmaxAll.count, expectedArgmaxAll.count,
                "argmax sequence length mismatch")
            // Per-position logit margin between the two ports' chosen tokens.
            let logits2D = logits[0].asType(.float32)  // (L, vocab)
            var mismatches = [Int]()
            var maxTieMargin: Float = 0
            for i in 0 ..< L where swiftArgmaxAll[i] != expectedArgmaxAll[i] {
                mismatches.append(i)
                let swiftTok = Int(swiftArgmaxAll[i])
                let refTok = Int(expectedArgmaxAll[i])
                let margin =
                    (logits2D[i, swiftTok] - logits2D[i, refTok]).item(Float.self)
                maxTieMargin = max(maxTieMargin, margin)
            }
            // Each mismatch must be a near-tie: the validated dense control's
            // mismatch sits well under this bound; a genuine port bug would not.
            XCTAssertLessThan(
                maxTieMargin, 0.5,
                "argmax mismatch at \(mismatches) with margin \(maxTieMargin) "
                    + "is NOT a near-tie — MoE forward genuinely diverges")

            let argmax = argMax(last, axis: -1).item(Int.self)
            XCTAssertEqual(
                argmax, expectedArgmax,
                "next-token argmax \(argmax) != reference \(expectedArgmax)")

            // SECONDARY — distributional parity. Cosine ≈ 1 is the tight
            // signal; relMaxAbs is a softcap-saturation-sensitive diagnostic,
            // bounded against the empirical floor the validated dense 31B path
            // sets through this same oracle (cosine 0.99983 / relMaxAbs 0.027).
            let diff = last - expectedLast
            let maxAbs = diff.abs().max().item(Float.self)
            let refMaxAbs = expectedLast.abs().max().item(Float.self)
            let dot = (last * expectedLast).sum().item(Float.self)
            let nOut = sqrt((last * last).sum().item(Float.self))
            let nExp = sqrt((expectedLast * expectedLast).sum().item(Float.self))
            let cosine = dot / (nOut * nExp + 1e-12)
            let relMaxAbs = maxAbs / (refMaxAbs + 1e-6)

            XCTAssertGreaterThan(
                cosine, 0.999,
                "MoE forward/reference cosine \(cosine) — port diverges")
            XCTAssertLessThan(
                relMaxAbs, 0.12,
                "MoE forward/reference relMaxAbs \(relMaxAbs) "
                    + "(maxAbs=\(maxAbs), refMax=\(refMaxAbs))")
            print(
                "Gemma4 MoE parity: argmax=\(argmax) cosine=\(cosine) "
                    + "relMaxAbs=\(relMaxAbs) "
                    + "argmax-agree=\(L - mismatches.count)/\(L) "
                    + "maxTieMargin=\(maxTieMargin)")
        }
    }
}
