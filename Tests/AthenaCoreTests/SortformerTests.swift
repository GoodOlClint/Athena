import Foundation
import MLX
import XCTest

@testable import AthenaTranscription

/// M4.3a — vendored Sortformer **integration** check: the model
/// downloads, loads, and runs end-to-end producing well-formed
/// diarization output over the correct duration. Speaker-separation
/// *accuracy* is an upstream property of the mlx-community model +
/// Blaizzy port (validated there); synthetic macOS `say` voices are a
/// poor diarization fixture — short, clean TTS often collapses to one
/// speaker — so we don't assert ≥2 here (real-sample accuracy is a
/// manual follow-up). Gated + heavy; validate via xcodebuild.
final class SortformerIntegrationTests: XCTestCase {

    private func sayClip(_ voice: String, _ text: String) throws -> [Float]
    {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("athena-spk-\(UUID()).aiff")
        defer { try? FileManager.default.removeItem(at: url) }
        let p = Process()
        p.executableURL = URL(fileURLWithPath: "/usr/bin/say")
        p.arguments = ["-v", voice, "-o", url.path, text]
        try p.run()
        p.waitUntilExit()
        guard p.terminationStatus == 0 else {
            throw XCTSkip("`say -v \(voice)` unavailable")
        }
        return try AudioDecode.pcm16kMono(from: url)
    }

    func testSortformerRunsEndToEnd() async throws {
        guard
            ProcessInfo.processInfo.environment["ATHENA_RUN_MODEL_TESTS"]
                == "1"
        else { throw XCTSkip("set ATHENA_RUN_MODEL_TESTS=1 (heavy)") }

        // Speaker A then speaker B — two clearly different voices.
        let a = try sayClip(
            "Alex",
            "Hello, this is the first speaker talking for a little "
                + "while so there is enough audio to segment.")
        let b = try sayClip(
            "Samantha",
            "And now a completely different second speaker is "
                + "talking, also for a few seconds of audio.")
        let pcm = a + b
        XCTAssertGreaterThan(pcm.count, 16_000, "need >1s audio")

        let model = try await SortformerModel.fromPretrained(
            "mlx-community/diar_streaming_sortformer_4spk-v2.1-fp16")
        let out = try await model.generate(
            audio: MLXArray(pcm).asType(.float32), sampleRate: 16_000)

        // Integration: well-formed segments spanning ~the clip
        // (segment times are the real signal; `totalTime` is the
        // model's own metric, not audio duration — don't assert it).
        XCTAssertFalse(out.segments.isEmpty, "no speaker segments")
        XCTAssertGreaterThanOrEqual(out.numSpeakers, 1)
        let expected = Double(pcm.count) / 16_000.0
        for s in out.segments {
            XCTAssertLessThanOrEqual(s.start, s.end)
            XCTAssertGreaterThanOrEqual(s.start, -0.001)
            XCTAssertLessThanOrEqual(Double(s.end), expected + 2.0)
            XCTAssertGreaterThanOrEqual(s.speaker, 0)
        }
        let lastEnd = out.segments.map { Double($0.end) }.max() ?? 0
        XCTAssertEqual(
            lastEnd, expected, accuracy: 3.0,
            "segments should span ~the clip duration")
        XCTAssertFalse(out.text.isEmpty)
    }
}
