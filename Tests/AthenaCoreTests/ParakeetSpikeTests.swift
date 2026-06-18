import AthenaCore
import Foundation
import MLX
import XCTest

@testable import AthenaTranscription

/// THROWAWAY FEASIBILITY SPIKE (ADR 019). Heavy, real-model benchmark of the
/// Parakeet-TDT-0.6B-v3 MLX port: load → encode → greedy TDT decode a 60 s
/// clip, print decode speed + the full transcript, and assert the transcript
/// looks like real words (so a broken forward can't pass as a benchmark).
///
/// Gated on `ATHENA_RUN_MODEL_TESTS=1` (needs the MLX metallib + a ~2.4 GB
/// model download on first run) and an audio fixture at `ATHENA_DIAR_FIXTURE`
/// (default `/tmp/audio/diar60.wav`). Self-skips otherwise.
final class ParakeetSpikeTests: XCTestCase {
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

    func testTranscribeBenchmark() async throws {
        let clip = try gate()
        let pcm = try AudioDecode.pcm16kMono(from: clip)
        let audioSeconds = Double(pcm.count) / 16000.0

        let loadStart = CFAbsoluteTimeGetCurrent()
        let model = try await ParakeetLoader.fromPretrained(
            "mlx-community/parakeet-tdt-0.6b-v3")
        let loadSeconds = CFAbsoluteTimeGetCurrent() - loadStart

        let result = model.transcribe(pcm)

        let totalSeconds = result.encoderSeconds + result.decodeSeconds
        let decodeTokPerSec =
            result.decodeSeconds > 0
            ? Double(result.tokenIds.count) / result.decodeSeconds : 0
        let rtf = totalSeconds > 0 ? audioSeconds / totalSeconds : 0

        print(
            """

            ========================= PARAKEET SPIKE =========================
            audio:           \(String(format: "%.1f", audioSeconds)) s
            model load:      \(String(format: "%.2f", loadSeconds)) s
            encoder:         \(String(format: "%.1f", result.encoderSeconds * 1000)) ms
            decode:          \(String(format: "%.1f", result.decodeSeconds * 1000)) ms
            total inference: \(String(format: "%.1f", totalSeconds * 1000)) ms
            decode steps:    \(result.decodeSteps)
            emitted tokens:  \(result.tokenIds.count)
            decode tok/s:    \(String(format: "%.1f", decodeTokPerSec))
            real-time factor:\(String(format: "%.1f", rtf))x (audio-s / wall-s)
            ------------------------------ TRANSCRIPT ------------------------
            \(result.transcript)
            ==================================================================

            """)

        // Correctness gate: a real forward yields real words, not token noise.
        let t = result.transcript.trimmingCharacters(in: .whitespacesAndNewlines)
        XCTAssertFalse(t.isEmpty, "empty transcript — forward is broken")
        XCTAssertTrue(t.contains(" "), "no spaces — likely token garbage")
        let lettered = t.filter { $0.isLetter }.count
        XCTAssertGreaterThan(
            lettered, 50, "too few letters for 60 s of speech — forward suspect")
        // Guard against a degenerate repeated-token collapse: the transcript
        // should have reasonable word diversity for a minute of speech.
        let words = t.split(separator: " ").map { String($0).lowercased() }
        let distinct = Set(words).count
        XCTAssertGreaterThan(
            distinct, 15,
            "only \(distinct) distinct words — looks like a repeated-token collapse")
        print("[parakeet] distinct words: \(distinct) / total \(words.count)")
    }
}
