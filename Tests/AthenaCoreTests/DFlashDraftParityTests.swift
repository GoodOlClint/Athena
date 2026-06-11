import Foundation
import MLX
import XCTest

@testable import AthenaModels

/// M63.1 — DFlash draft-model parity. Loads the real z-lab drafter
/// checkpoint into the vendored Swift `DFlashDraftModel`, runs the no-cache
/// block forward on the inputs captured from the bstnxbt/dflash-mlx Python
/// reference, and asserts the output matches the reference output. This is
/// the cross-implementation oracle for the port: a wrong reshape / RoPE
/// offset / mask / projection diverges here.
///
/// Gated on `ATHENA_RUN_MODEL_TESTS=1` AND the drafter weights being
/// present (env `ATHENA_DFLASH_DRAFT_DIR`, else the default HF cache
/// snapshot). The fixture (`Fixtures/dflash_draft_parity.safetensors`) is
/// produced by `deploy/dflash/gen_parity_fixture.py` and is tied to the
/// 31B drafter; regenerate if the drafter changes.
final class DFlashDraftParityTests: XCTestCase {

    private func draftDir() -> URL? {
        let env = ProcessInfo.processInfo.environment
        if let p = env["ATHENA_DFLASH_DRAFT_DIR"], !p.isEmpty {
            return URL(fileURLWithPath: p)
        }
        let home = FileManager.default.homeDirectoryForCurrentUser
        let snaps = home.appending(
            path:
                ".cache/huggingface/hub/models--z-lab--gemma-4-31B-it-DFlash/snapshots"
        )
        guard
            let entries = try? FileManager.default.contentsOfDirectory(
                at: snaps, includingPropertiesForKeys: nil),
            let first = entries.first(where: {
                FileManager.default.fileExists(
                    atPath: $0.appending(component: "config.json").path)
            })
        else { return nil }
        return first
    }

    private func fixtureURL() -> URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .appending(path: "Fixtures/dflash_draft_parity.safetensors")
    }

    func testDraftForwardMatchesPythonReference() throws {
        guard
            ProcessInfo.processInfo.environment["ATHENA_RUN_MODEL_TESTS"] == "1"
        else { throw XCTSkip("set ATHENA_RUN_MODEL_TESTS=1 to run model tests") }
        guard let dir = draftDir() else {
            throw XCTSkip("z-lab DFlash drafter checkpoint not present")
        }
        let fixture = fixtureURL()
        guard FileManager.default.fileExists(atPath: fixture.path) else {
            throw XCTSkip("parity fixture missing: \(fixture.path)")
        }

        let model = try DFlashDraftLoader.load(directory: dir)
        model.bindTargetModel(embedScale: 1.0)  // fixture pins embed_scale=1.0

        let arrays = try MLX.loadArrays(url: fixture)
        let noise = try XCTUnwrap(arrays["noise_embedding"])
        let targetHidden = try XCTUnwrap(arrays["target_hidden"])
        let expected = try XCTUnwrap(arrays["expected_out"]).asType(.float32)

        let out = model(noiseEmbedding: noise, targetHidden: targetHidden)
            .asType(.float32)
        eval(out)

        XCTAssertEqual(out.shape, expected.shape, "draft output shape mismatch")

        // Parity: cosine ≈ 1 and small relative max-abs error. Both sides
        // run the same MLX core in bf16, so agreement is tight; tolerances
        // leave headroom for reduction-order differences.
        let diff = out - expected
        let maxAbs = diff.abs().max().item(Float.self)
        let refMaxAbs = expected.abs().max().item(Float.self)
        let dot = (out * expected).sum().item(Float.self)
        let nOut = sqrt((out * out).sum().item(Float.self))
        let nExp = sqrt((expected * expected).sum().item(Float.self))
        let cosine = dot / (nOut * nExp + 1e-12)
        let relMaxAbs = maxAbs / (refMaxAbs + 1e-6)

        XCTAssertGreaterThan(
            cosine, 0.999,
            "draft/reference cosine \(cosine) — port diverges from reference")
        XCTAssertLessThan(
            relMaxAbs, 0.05,
            "draft/reference relative max-abs \(relMaxAbs) (maxAbs=\(maxAbs), refMax=\(refMaxAbs))")
        print(
            "DFlash draft parity: cosine=\(cosine) relMaxAbs=\(relMaxAbs) "
                + "maxAbs=\(maxAbs) refMax=\(refMaxAbs)")
    }
}
