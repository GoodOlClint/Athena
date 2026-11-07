import AthenaCore
import Foundation
import XCTest

@testable import AthenaLLM

/// #247 + #250 — one definition of which weight files a Whisper checkpoint
/// may carry, shared by the loader, `verify` and the `pull` preflight.
final class WeightFilesTests: XCTestCase {
    private func writeSafetensors(to url: URL) throws {
        let header =
            #"{"__metadata__":{"format":"mlx"},"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#
        var data = Data()
        var len = UInt64(header.utf8.count).littleEndian
        withUnsafeBytes(of: &len) { data.append(contentsOf: $0) }
        data.append(contentsOf: Array(header.utf8))
        data.append(contentsOf: [0, 0, 0, 0])
        try data.write(to: url)
    }

    private func whisperDir(weights: [String]) throws -> URL {
        let d = FileManager.default.temporaryDirectory
            .appendingPathComponent("wf-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(
            at: d, withIntermediateDirectories: true)
        try Data(#"{"model_type":"whisper","n_vocab":51866}"#.utf8)
            .write(to: d.appendingPathComponent("config.json"))
        for w in weights { try writeSafetensors(to: d.appendingPathComponent(w)) }
        return d
    }

    // (a)
    func testModelSafetensorsResolvesForWhisper() throws {
        let d = try whisperDir(weights: ["model.safetensors"])
        defer { try? FileManager.default.removeItem(at: d) }
        XCTAssertEqual(
            WeightFiles.whisper(in: d)?.lastPathComponent, "model.safetensors")
    }

    func testWeightsSafetensorsStillResolvesAndWins() throws {
        let d = try whisperDir(weights: ["model.safetensors", "weights.safetensors"])
        defer { try? FileManager.default.removeItem(at: d) }
        XCTAssertEqual(
            WeightFiles.whisper(in: d)?.lastPathComponent, "weights.safetensors")
    }

    // (b)
    func testVerifyAndLoaderAgreeOnUnrelatedSafetensors() throws {
        let d = try whisperDir(weights: ["foo.safetensors"])
        defer { try? FileManager.default.removeItem(at: d) }
        XCTAssertNil(WeightFiles.whisper(in: d), "loader must reject foo.safetensors")
        let problems = ModelHealth.check(d)
        XCTAssertTrue(
            problems.contains { $0.contains("weights.safetensors") },
            "verify must reject what the loader rejects, got \(problems)")
    }

    func testVerifyAcceptsModelSafetensorsWhisper() throws {
        let d = try whisperDir(weights: ["model.safetensors"])
        defer { try? FileManager.default.removeItem(at: d) }
        XCTAssertFalse(
            ModelHealth.check(d).contains { $0.contains("safetensors") })
    }

    // (c)
    func testNpzOnlyListingIsUnloadableAndNamesTheFormat() {
        let files = ["config.json", "weights.npz", "README.md"]
        let problem = WeightFiles.listingProblem(
            files: files, modality: .transcription(.whisper))
        XCTAssertNotNil(problem)
        XCTAssertTrue(problem?.contains("weights.npz") == true)

        let support = ModelPreflight.applyListing(
            ModelSupport(
                modality: .transcription(.whisper), loadability: .loadable),
            files: files)
        guard case let .unsupported(reason, _) = support.loadability else {
            return XCTFail("expected unsupported, got \(support.loadability)")
        }
        XCTAssertTrue(reason.contains("weights.npz"))
    }

    func testPullRefusesNpzOnlyRepoAndLeavesNoStoreEntry() async throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("wfstore-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        do {
            _ = try await ModelPull.pull(
                id: "org/whisper-npz", into: root,
                listFiles: { _, _ in ["config.json", "weights.npz"] },
                modality: { _, _ in nil })
            XCTFail("pull must refuse")
        } catch let e as ModelPull.PullError {
            XCTAssertTrue("\(e)".contains("weights.npz"))
        }
        let left =
            (try? FileManager.default.contentsOfDirectory(atPath: root.path)) ?? []
        XCTAssertTrue(left.isEmpty, "no store entry may remain, found \(left)")
    }

    func testGenericListingWithoutSafetensorsIsRejected() {
        XCTAssertNotNil(
            WeightFiles.listingProblem(
                files: ["config.json", "weights.npz"], modality: nil))
    }

    func testPullRefusesWhisperRepoWithOnlyUnrelatedSafetensors() async throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("wfstore-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        do {
            _ = try await ModelPull.pull(
                id: "org/whisper-foo", into: root,
                listFiles: { _, _ in ["config.json", "foo.safetensors"] },
                modality: { _, _ in .transcription(.whisper) })
            XCTFail("pull must refuse")
        } catch let e as ModelPull.PullError {
            XCTAssertTrue("\(e)".contains("weights.safetensors"))
        }
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.path))
    }

    // (d)
    func testSafetensorsListingsAreAccepted() {
        XCTAssertNil(
            WeightFiles.listingProblem(
                files: ["config.json", "model.safetensors"],
                modality: .transcription(.whisper)))
        XCTAssertNil(
            WeightFiles.listingProblem(
                files: ["config.json", "weights.safetensors"],
                modality: .transcription(.whisper)))
        XCTAssertNil(
            WeightFiles.listingProblem(
                files: [
                    "config.json", "model-00001-of-00002.safetensors",
                    "model-00002-of-00002.safetensors",
                    "model.safetensors.index.json",
                ], modality: nil))
        XCTAssertNotNil(
            WeightFiles.listingProblem(
                files: ["config.json", "foo.safetensors"],
                modality: .transcription(.whisper)))
    }

    // (e)
    func testModuleLoadFailedBodyCarriesCauseInsideEnvelope() {
        let e = AthenaError.moduleLoadFailed(
            .transcription, reason: "whisper m: /secret/path",
            cause: "whisper: missing weights.safetensors")
        XCTAssertTrue(e.message.contains("missing weights.safetensors"))
        XCTAssertFalse(e.message.contains("/secret/path"))
        XCTAssertEqual(e.httpStatus, 500)
        XCTAssertEqual(e.code, "module_load_failed")
        XCTAssertEqual(e.type, "server_error")
    }

    func testClientMessageKeepsCauseWhenModelNamed() {
        let e = AthenaError.moduleLoadFailed(
            .transcription, reason: "/secret/path", cause: "whisper: missing weights.safetensors")
        let m = e.clientMessage(model: "whisper-large-v3-fp16")
        XCTAssertTrue(m.contains("missing weights.safetensors"))
        XCTAssertFalse(m.contains("/secret/path"))
        XCTAssertEqual(
            AthenaError.moduleLoadFailed(.llm, reason: "x").clientMessage(model: "m"),
            "Module llm failed to load model 'm'.")
    }

    func testModuleLoadFailedWithoutCauseIsUnchanged() {
        XCTAssertEqual(
            AthenaError.moduleLoadFailed(.llm, reason: "x").message,
            "Module llm failed to load.")
    }
}
