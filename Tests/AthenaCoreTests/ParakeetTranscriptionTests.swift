import AthenaCore
import Foundation
import MLX
import XCTest

@testable import AthenaTranscription

/// Heavy, real-model validation of the production Parakeet-TDT-0.6B-v3 engine
/// (ADR 020, hardened from the ADR-019 spike): load → encode → greedy TDT
/// decode a clip, print decode speed + the full transcript, and assert the
/// transcript looks like real words (so a broken forward can't pass as a
/// benchmark). Also pins the mel pipeline's per-feature normalization invariant
/// (R1 mel-exactness) without needing a model or Python.
///
/// Gated on `ATHENA_RUN_MODEL_TESTS=1` (needs the MLX metallib; the transcribe
/// test also needs a ~2.4 GB model download on first run) and an audio fixture
/// at `ATHENA_DIAR_FIXTURE` (default `/tmp/audio/diar60.wav`). Self-skips
/// otherwise.
final class ParakeetTranscriptionTests: XCTestCase {
    private func modelGate() throws {
        guard ProcessInfo.processInfo.environment["ATHENA_RUN_MODEL_TESTS"] == "1"
        else { throw XCTSkip("set ATHENA_RUN_MODEL_TESTS=1 (heavy)") }
    }

    private func fixture() throws -> URL {
        let path =
            ProcessInfo.processInfo.environment["ATHENA_DIAR_FIXTURE"]
            ?? "/tmp/audio/diar60.wav"
        let url = URL(fileURLWithPath: path)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw XCTSkip("no fixture at \(path)")
        }
        return url
    }

    /// R1 mel-exactness pin: per-feature (time-axis) normalization makes every
    /// mel feature zero-mean / unit-std over time. Holds for any non-degenerate
    /// signal and breaks if the normalization axis/formula regresses; the L1
    /// magnitude (`|re|+|im|`) keeps the spectrum finite and well-scaled. Needs
    /// the MLX metallib (gated) but no model download. Uses a linear chirp so
    /// energy sweeps across mel bins → per-feature variance is non-trivial.
    func testMelNormalizationInvariant() throws {
        try modelGate()
        let sr = 16000
        let n = sr  // 1 s
        var samples = [Float](repeating: 0, count: n)
        for i in 0..<n {
            let t = Double(i) / Double(sr)
            let f = 200.0 + 1800.0 * t  // 200 → 2000 Hz sweep
            samples[i] = Float(0.5 * sin(2.0 * .pi * f * t))
        }
        let mel = ParakeetMel.logMel(samples, ParakeetConfig())
        XCTAssertEqual(mel.ndim, 3)
        XCTAssertEqual(mel.dim(0), 1)
        XCTAssertEqual(mel.dim(2), 128, "128 mel features")
        let frames = mel.dim(1)
        XCTAssertGreaterThan(frames, 50)

        // All finite (no NaN/Inf from the magnitude/log path).
        let flat = mel.reshaped(-1)
        XCTAssertTrue(
            MLX.all(MLX.isFinite(flat)).item(Bool.self),
            "mel has non-finite values")

        // Per-feature mean ≈ 0 over the time axis (axis 1 of [1,T,128]).
        let mean = mel.mean(axis: 1)  // [1, 128]
        let maxAbsMean = MLX.max(MLX.abs(mean)).item(Float.self)
        XCTAssertLessThan(
            maxAbsMean, 1e-2, "per-feature mean should be ~0 after normalize")
        // Per-feature std ≈ 1 (the (x-mean)/(std+1e-5) target). Allow slack for
        // the +1e-5 denominator and any all-silent bins.
        let std = MLX.sqrt(mel.variance(axis: 1))  // [1, 128]
        let meanStd = std.mean().item(Float.self)
        XCTAssertEqual(
            meanStd, 1.0, accuracy: 0.05,
            "per-feature std should be ~1 after normalize (got \(meanStd))")
    }

    /// S4: a clip longer than `chunkSeconds` (120 s) is decoded in overlapping
    /// windows and stitched. Tile the fixture ×3 (~180 s) to force the chunked
    /// path, then assert the stitched timeline is coherent, monotonic, and
    /// reaches the tail (later chunks contributed with correct offsets).
    func testLongAudioChunking() async throws {
        try modelGate()
        let clip = try fixture()
        let one = try AudioDecode.pcm16kMono(from: clip)
        let pcm = one + one + one
        let audioSeconds = Double(pcm.count) / 16000.0
        XCTAssertGreaterThan(audioSeconds, 120, "need >120 s to exercise chunking")

        let model = try await ParakeetLoader.fromPretrained(
            "mlx-community/parakeet-tdt-0.6b-v3")
        let result = model.transcribe(pcm)

        let t = result.transcript.trimmingCharacters(in: .whitespacesAndNewlines)
        XCTAssertFalse(t.isEmpty, "empty transcript on long audio")
        XCTAssertFalse(t.contains("<|"), "control token leaked")
        XCTAssertFalse(result.tokens.isEmpty, "no aligned tokens")

        var last = -1.0
        for tk in result.tokens {
            XCTAssertGreaterThanOrEqual(
                tk.start, last - 1e-6, "stitched starts must be non-decreasing")
            last = tk.start
            XCTAssertLessThanOrEqual(
                tk.end, audioSeconds + 1.0, "token past clip end")
        }
        // The tail must be reached — proves a later chunk's offset timestamps
        // were merged in (not just the first window).
        XCTAssertGreaterThan(
            result.tokens.last!.start, 120.0,
            "stitched timeline never passes 120 s — chunking/stitch broken")
        print(
            "[parakeet] long-audio: \(String(format: "%.0f", audioSeconds)) s, "
                + "tokens \(result.tokens.count), last start "
                + "\(String(format: "%.1f", result.tokens.last!.start)) s")
    }

    func testTranscribeBenchmark() async throws {
        try modelGate()
        let clip = try fixture()
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

            ======================= PARAKEET TRANSCRIBE ======================
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
        // R2: no control tokens leaked through the detokenizer.
        XCTAssertFalse(
            t.contains("<|"), "special control token leaked into transcript")
        XCTAssertFalse(t.contains("\u{2581}"), "raw ▁ marker leaked")
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
        // Throughput floor: ADR 019 measured ~63× RT (Debug). Far below that
        // signals a perf regression (e.g. an un-eval'd graph or O(T²) blowup).
        XCTAssertGreaterThan(
            rtf, 5.0, "real-time factor \(rtf)x is far below the ~63x baseline")
        print("[parakeet] distinct words: \(distinct) / total \(words.count)")

        // S3: TDT-derived timestamps — non-decreasing token starts, all within
        // the clip; sentence segments + words monotonic and in-bounds.
        XCTAssertFalse(result.tokens.isEmpty, "no aligned tokens")
        var lastStart = -1.0
        for tok in result.tokens {
            XCTAssertGreaterThanOrEqual(tok.start, 0)
            XCTAssertLessThanOrEqual(
                tok.end, audioSeconds + 1.0, "token past clip end")
            XCTAssertGreaterThanOrEqual(
                tok.start, lastStart - 1e-6, "token starts must be non-decreasing")
            lastStart = tok.start
        }
        let segs = ParakeetAlignment.segments(
            from: result.tokens, attachWords: true)
        XCTAssertFalse(segs.isEmpty, "no segments")
        for s in segs {
            XCTAssertLessThanOrEqual(s.start, s.end)
            XCTAssertLessThanOrEqual(s.end, audioSeconds + 1.0)
        }
        let alignedWords = ParakeetAlignment.words(from: result.tokens)
        XCTAssertGreaterThan(
            alignedWords.count, 20, "too few timed words for 60 s of speech")
        for i in 1..<alignedWords.count {
            XCTAssertLessThanOrEqual(
                alignedWords[i - 1].start, alignedWords[i].start)
        }
        print(
            "[parakeet] segments: \(segs.count), timed words: "
                + "\(alignedWords.count), first seg: "
                + "[\(String(format: "%.2f", segs[0].start))–"
                + "\(String(format: "%.2f", segs[0].end))] "
                + "\"\(segs[0].text.prefix(60))\"")
    }
}
