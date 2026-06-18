import AthenaCore
import Foundation
import MLX
import XCTest

@testable import AthenaTranscription

/// Heavy, real-model validation of the pyannote pipeline (ADR 018 / S2–S3):
/// PyanNet segmentation → WeSpeaker embed → same-window cannot-link cluster.
/// Mirrors the `/v1/audio/diarizations?method=pyannote` route without the
/// daemon. Gated on `ATHENA_RUN_MODEL_TESTS=1` (needs the MLX metallib +
/// network) and a real multi-speaker fixture at `ATHENA_DIAR_FIXTURE`
/// (default `/tmp/audio/diar60.wav`). Synthetic `say` audio is useless here
/// (collapses to one speaker) — use REAL audio (M4.3/M25 lesson).
final class PyanNetSegmentationModelTests: XCTestCase {
    private func gate() throws -> URL {
        guard ProcessInfo.processInfo.environment["ATHENA_RUN_MODEL_TESTS"] == "1"
        else { throw XCTSkip("set ATHENA_RUN_MODEL_TESTS=1 (heavy)") }
        let path =
            ProcessInfo.processInfo.environment["ATHENA_DIAR_FIXTURE"]
            ?? "/tmp/audio/diar60.wav"
        let url = URL(fileURLWithPath: path)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw XCTSkip("no fixture at \(path)")
        }
        return url
    }

    func testSegmentationProducesSaneRegions() async throws {
        let clip = try gate()
        let pcm = try AudioDecode.pcm16kMono(from: clip)
        let durationSec = Double(pcm.count) / 16000.0
        let model = try await PyanNetSegmentationModel.fromPretrained(
            "aufklarer/Pyannote-Segmentation-MLX")
        let regions = model.segment(pcm)

        // Structural sanity: speech found, in-bounds, ordered, multiple local
        // tracks somewhere (a multi-speaker conversation), not all-speech.
        XCTAssertFalse(regions.isEmpty, "no speech regions on real audio")
        for r in regions {
            XCTAssertGreaterThanOrEqual(r.start, 0)
            XCTAssertLessThanOrEqual(r.end, durationSec + 0.05)
            XCTAssertGreaterThan(r.end, r.start)
        }
        for i in 1..<regions.count {
            XCTAssertLessThanOrEqual(regions[i - 1].start, regions[i].start)
        }
        let speech = regions.reduce(0.0) { $0 + ($1.end - $1.start) }
        let coverage = speech / durationSec
        let distinctLocal = Set(regions.map { $0.localSpeaker })
        print(
            """
            [pyannote] dur=\(String(format: "%.1f", durationSec))s \
            regions=\(regions.count) speech=\(String(format: "%.1f", speech))s \
            coverage~\(String(format: "%.2f", coverage)) \
            localSpeakers=\(distinctLocal.sorted()) \
            windows=\(Set(regions.map { $0.window }).count)
            """)
        // A real conversation: some speech, not literally everything.
        XCTAssertGreaterThan(coverage, 0.05)
        XCTAssertLessThan(coverage, 1.5)
    }

    func testFullPipelineRecoversMultipleSpeakers() async throws {
        let clip = try gate()
        let pcm = try AudioDecode.pcm16kMono(from: clip)
        let seg = try await PyanNetSegmentationModel.fromPretrained(
            "aufklarer/Pyannote-Segmentation-MLX")
        let we = try await WeSpeakerModel.fromPretrained(
            "aufklarer/WeSpeaker-ResNet34-LM-MLX")
        let regions = seg.segment(pcm)
        try XCTSkipIf(regions.isEmpty, "no regions")

        // Embed each region (slice the PCM by region time).
        var embeddings: [[Float]] = []
        for r in regions {
            let a = max(0, Int(r.start * 16000))
            let b = min(pcm.count, Int(r.end * 16000))
            guard b > a else {
                embeddings.append([Float](repeating: 0, count: 256))
                continue
            }
            embeddings.append(we.embed(Array(pcm[a..<b])))
        }
        let rawLabels = AgglomerativeClustering.cluster(
            embeddings, threshold: 0.75,
            cannotLink: PyannoteSegmentationDecode.sameWindowCannotLink(regions))
        // Auto-mode refinement (matches the route): dissolve tiny clusters.
        let labels = PyannoteSegmentationDecode.reassignSmallClusters(
            embeddings: embeddings, labels: rawLabels,
            durations: regions.map { $0.end - $0.start }, minDuration: 6.0)
        let speakers = Set(labels).count
        let turns = PyannoteSegmentationDecode.mergeSameSpeakerTurns(
            zip(regions, labels).map {
                DiarizationTurn(start: $0.start, end: $0.end, speaker: $1)
            })
        print(
            "[pyannote] full-pipeline speakers=\(speakers) "
                + "(raw=\(Set(rawLabels).count)) "
                + "regions=\(regions.count) turns=\(turns.count)")
        // A multi-speaker clip should recover ≥2 speakers (no 4-cap), and the
        // refinement must keep the count sane (not the ~90 raw fragmentation).
        XCTAssertGreaterThanOrEqual(speakers, 2)
        XCTAssertLessThanOrEqual(speakers, 20)
    }
}
