import Foundation
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import Tokenizers
import XCTest

@testable import AthenaModels

/// M63.2 — the Gemma4 hidden-capture seam + KV-trim rollback.
///
/// Two gates:
///  1. `testRotatingTrimRecentRollback` — proves `RotatingKVCache.trimRecent`
///     (the sliding-window speculative rollback) leaves the cache behaving
///     exactly as if the rejected tokens were never appended, including past
///     the window-wrap boundary. Pure-MLX, no model weights (only needs the
///     metallib, so it is gated like the other MLX tests).
///  2. `testCaptureForwardMatchesPlain` — loads a real Gemma4 target and
///     asserts `callReturningHidden(...).logits` is bit-identical to the
///     untouched `callAsFunction(...)`, so the additive capture forward did
///     not drift from the validated decode path, and that the captured
///     per-layer hiddens + concatenated context feature have the right
///     shapes.
final class DFlashGemma4CaptureTests: XCTestCase {

    private func requireModelTests() throws {
        guard ProcessInfo.processInfo.environment["ATHENA_RUN_MODEL_TESTS"] == "1"
        else { throw XCTSkip("set ATHENA_RUN_MODEL_TESTS=1 (needs the metallib)") }
    }

    // MARK: Rollback

    /// Build a key/value block where token `i` is the constant `i`, so a
    /// later comparison can tell exactly which tokens a cache holds.
    private func tokenKV(
        ids: [Int], heads: Int, headDim: Int
    ) -> (MLXArray, MLXArray) {
        let L = ids.count
        var flat = [Float]()
        flat.reserveCapacity(heads * L * headDim)
        // shape (1, heads, L, headDim), value = id
        for _ in 0 ..< heads {
            for id in ids {
                for _ in 0 ..< headDim { flat.append(Float(id)) }
            }
        }
        let k = MLXArray(flat, [1, heads, L, headDim])
        return (k, k + Float(1000))  // values offset so k≠v
    }

    func testRotatingTrimRecentRollback() throws {
        try requireModelTests()
        let window = 8, heads = 2, headDim = 4, block = 4
        // Feed 12 tokens (0..11) in blocks of 4, then roll back the last 3.
        // Reference: a fresh cache fed only tokens 0..8 (= 12 − 3).
        func feed(_ cache: RotatingKVCache, _ ids: [Int]) {
            for start in stride(from: 0, to: ids.count, by: block) {
                let chunk = Array(ids[start ..< min(start + block, ids.count)])
                let (k, v) = tokenKV(ids: chunk, heads: heads, headDim: headDim)
                _ = cache.update(keys: k, values: v)
            }
        }
        let a = RotatingKVCache(maxSize: window, keep: 0)
        feed(a, Array(0 ..< 12))
        a.trimRecent(3)

        let b = RotatingKVCache(maxSize: window, keep: 0)
        feed(b, Array(0 ..< 9))

        XCTAssertEqual(a.offset, b.offset, "rolled-back offset diverged")

        // The raw `update` return is the ring buffer (rotation-dependent
        // order), so compare via a MULTI-token probe: `updateConcat` calls
        // `temporalOrder` first, normalizing both caches to linear order
        // before append. Identical fetched keys/values then prove the
        // rolled-back cache is indistinguishable from one that never saw the
        // rejected tokens — the speculative-rollback invariant.
        let (pk, pv) = tokenKV(ids: [999, 998], heads: heads, headDim: headDim)
        let (ak, av) = a.update(keys: pk, values: pv)
        let (bk, bv) = b.update(keys: pk, values: pv)
        eval(ak, av, bk, bv)
        XCTAssertEqual(ak.shape, bk.shape, "rolled-back cache shape diverged")
        let dk = (ak - bk).abs().max().item(Float.self)
        let dv = (av - bv).abs().max().item(Float.self)
        XCTAssertEqual(dk, 0, "keys diverge after trimRecent rollback")
        XCTAssertEqual(dv, 0, "values diverge after trimRecent rollback")
    }

    // MARK: Capture forward

    private func gemma4Dir() -> URL? {
        let env = ProcessInfo.processInfo.environment
        if let p = env["ATHENA_TEST_GEMMA4_DIR"], !p.isEmpty {
            return URL(fileURLWithPath: p)
        }
        // Prefer the lighter 12B already on disk; else the 31B DFlash target.
        let home = FileManager.default.homeDirectoryForCurrentUser
        for name in [
            "models--mlx-community--gemma-4-12B-it-8bit",
            "models--mlx-community--gemma-4-31b-it-4bit",
        ] {
            let snaps = home.appending(
                path: ".cache/huggingface/hub/\(name)/snapshots")
            if let entries = try? FileManager.default.contentsOfDirectory(
                at: snaps, includingPropertiesForKeys: nil),
                let first = entries.first(where: {
                    FileManager.default.fileExists(
                        atPath: $0.appending(component: "config.json").path)
                })
            { return first }
        }
        let store = home.appending(path: ".athena/models/gemma-4-12B-it-8bit")
        if FileManager.default.fileExists(
            atPath: store.appending(component: "config.json").path)
        { return store }
        return nil
    }

    func testCaptureForwardMatchesPlain() async throws {
        try requireModelTests()
        guard let dir = gemma4Dir() else {
            throw XCTSkip("no Gemma4 checkpoint present")
        }
        let container = try await loadModelContainer(
            from: dir.resolvingSymlinksInPath(),
            using: #huggingFaceTokenizerLoader())

        try await container.perform { (ctx: ModelContext) in
            let backbone: DFlashGemma4Backbone
            if let g = ctx.model as? Gemma4Model {
                backbone = g
            } else if let g = ctx.model as? Gemma4TextModel {
                backbone = g
            } else {
                throw XCTSkip(
                    "loaded model is \(type(of: ctx.model)), not a Gemma4 target")
            }

            let tokens = MLXArray(
                [Int32(2), 651, 6403, 576, 6081, 603, 11173, 12], [1, 8])
            let layers = [1, 5, 10, 15, 20, 25]

            // Plain forward (the untouched validated path) on a fresh cache.
            let plain: MLXArray
            if let g = ctx.model as? Gemma4Model {
                plain = g(tokens, cache: g.newCache(parameters: nil))
            } else {
                let g = ctx.model as! Gemma4TextModel
                plain = g(tokens, cache: g.newCache(parameters: nil))
            }

            // Capture forward on its own fresh cache.
            let cache = backbone.newCache(parameters: nil)
            let (logits, captured) = backbone.callReturningHidden(
                tokens, cache: cache, captureLayers: Set(layers))
            eval(plain, logits)

            XCTAssertEqual(
                logits.shape, plain.shape, "capture logits shape mismatch")
            let maxDiff = (logits.asType(.float32) - plain.asType(.float32))
                .abs().max().item(Float.self)
            XCTAssertEqual(
                maxDiff, 0,
                "capture forward diverged from callAsFunction (maxDiff=\(maxDiff))")

            let H = try XCTUnwrap(captured[layers[0]]).dim(2)
            for l in layers {
                let h = try XCTUnwrap(captured[l], "missing captured layer \(l)")
                XCTAssertEqual(h.dim(0), 1)
                XCTAssertEqual(h.dim(1), tokens.dim(1))
                XCTAssertEqual(h.dim(2), H)
            }
            let ctxFeat = DFlashGemma4Target.contextFeature(
                from: captured, layerOrder: layers)
            eval(ctxFeat)
            XCTAssertEqual(ctxFeat.dim(0), 1)
            XCTAssertEqual(ctxFeat.dim(1), tokens.dim(1))
            XCTAssertEqual(ctxFeat.dim(2), layers.count * H)
            print(
                "DFlash Gemma4 capture: logits bit-identical; "
                    + "context feature \(ctxFeat.shape)")
        }
    }
}
