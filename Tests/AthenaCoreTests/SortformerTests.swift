import AthenaCore
import Foundation
import MLX
import XCTest

@testable import AthenaTranscription

/// M24.4a — offline diarization positional-capacity math. The offline
/// path caps at `max_source_positions` diar frames; this verifies the
/// frame-count helper mirrors ConvSubsampling's stride-2 stages so the
/// guard triggers at the right length. Pure, CI-safe.
final class SortformerOfflineCapTests: XCTestCase {
    func testSubsamplingFrameMath() {
        // factor 8 = 3 stride-2 stages of floor((L-1)/2)+1.
        // 64 -> 32 -> 16 -> 8; tiny inputs floor toward 1.
        XCTAssertEqual(
            SortformerModel.offlineDiarFrameCount(
                melFrames: 64, subsamplingFactor: 8), 8)
        XCTAssertEqual(
            SortformerModel.offlineDiarFrameCount(
                melFrames: 1, subsamplingFactor: 8), 1)
    }

    /// At 16 kHz, hop 160, factor 8 the frame rate is 12.5 fps, so the
    /// 1500-position offline table corresponds to ~120 s — the observed
    /// real-audio ceiling. ~12000 mel frames (120 s) must stay within
    /// 1500 diar frames; ~16000 (160 s) must exceed it.
    func testNominal120sFitsAnd160sExceeds() {
        let f120 = SortformerModel.offlineDiarFrameCount(
            melFrames: 12000, subsamplingFactor: 8)
        let f160 = SortformerModel.offlineDiarFrameCount(
            melFrames: 16000, subsamplingFactor: 8)
        XCTAssertLessThanOrEqual(f120, 1500)
        XCTAssertGreaterThan(f160, 1500)
    }
}

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

    /// M4.3b — the governed `MLXDiarizationModule` end-to-end through
    /// the `MemoryGovernor` (register → ensureLoaded → diarize).
    func testGovernedDiarizationModule() async throws {
        guard
            ProcessInfo.processInfo.environment["ATHENA_RUN_MODEL_TESTS"]
                == "1"
        else { throw XCTSkip("set ATHENA_RUN_MODEL_TESTS=1 (heavy)") }

        let pcm =
            try sayClip("Alex", "First speaker, a little audio here.")
            + sayClip("Samantha", "Second speaker, some more audio.")
        var wav = Data()  // minimal RIFF/WAVE 16k mono float32
        func le<T: FixedWidthInteger>(_ v: T) -> Data {
            withUnsafeBytes(of: v.littleEndian) { Data($0) }
        }
        let bytes = pcm.withUnsafeBytes { Data($0) }
        wav.append("RIFF".data(using: .ascii)!)
        wav.append(le(UInt32(36 + bytes.count)))
        wav.append("WAVEfmt ".data(using: .ascii)!)
        wav.append(le(UInt32(16)))
        wav.append(le(UInt16(3)))  // IEEE float
        wav.append(le(UInt16(1)))  // mono
        wav.append(le(UInt32(16_000)))
        wav.append(le(UInt32(16_000 * 4)))
        wav.append(le(UInt16(4)))
        wav.append(le(UInt16(32)))
        wav.append("data".data(using: .ascii)!)
        wav.append(le(UInt32(bytes.count)))
        wav.append(bytes)

        let m = MLXDiarizationModule()
        XCTAssertEqual(m.id, .diarization)
        let gov = MemoryGovernor(totalBudgetBytes: Int(8) << 30)
        await gov.register(m, evictable: true)
        try await gov.ensureLoaded(.diarization)

        let r = try await m.diarize(audio: wav, filename: "a.wav")
        XCTAssertFalse(r.turns.isEmpty)
        XCTAssertGreaterThanOrEqual(r.numSpeakers, 1)
        for t in r.turns {
            XCTAssertLessThanOrEqual(t.start, t.end)
            XCTAssertGreaterThanOrEqual(t.speaker, 0)
        }
    }
}

/// M50.3 — regression for the allocator-pool leak class M46.6 caught
/// in the embedder. Drives many short diarize calls back-to-back and
/// asserts MLX's pool stays bounded — without the end-of-call clear in
/// `Sortformer.generate`, per-call encoder + STFT buffers accumulate.
/// Gated + heavy.
final class SortformerMemoryRegressionTests: XCTestCase {

    func testGeneratePoolStaysBoundedAcrossManyCalls() async throws {
        guard
            ProcessInfo.processInfo.environment["ATHENA_RUN_MODEL_TESTS"]
                == "1"
        else { throw XCTSkip("set ATHENA_RUN_MODEL_TESTS=1 (heavy)") }

        // 2 s of synthetic noise — enough audio for the transformer
        // encoder to engage without exceeding maxSourcePositions. The
        // leak is allocator-pool growth from the forward pass, not
        // content-dependent.
        var pcm = [Float](repeating: 0, count: 2 * 16_000)
        var rng = SystemRandomNumberGenerator()
        for i in 0..<pcm.count {
            pcm[i] = Float(Int(rng.next() % 1000)) / 1000.0 - 0.5
        }
        let audio = MLXArray(pcm).asType(.float32)

        let model = try await SortformerModel.fromPretrained(
            "mlx-community/diar_streaming_sortformer_4spk-v2.1-fp16")

        // Warmup so first-call lazy allocations settle.
        _ = try await model.generate(audio: audio, sampleRate: 16_000)
        MLX.Memory.clearCache()
        let baseline = MLX.Memory.cacheMemory

        for _ in 0..<22 {
            _ = try await model.generate(audio: audio, sampleRate: 16_000)
        }

        let after = MLX.Memory.cacheMemory
        // Without M50.3's clear, the pool scales linearly with the
        // per-call encoder/STFT footprint × 22.
        let ceiling = 512 * 1024 * 1024
        XCTAssertLessThan(
            after - baseline, ceiling,
            "MLX cache pool drifted \(after - baseline) bytes "
            + "above baseline after 22 diarize calls (M50.3 leak)")
    }
}
