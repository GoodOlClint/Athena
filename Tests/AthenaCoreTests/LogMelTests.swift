import AVFoundation
import Foundation
import MLX
import XCTest

@testable import AthenaTranscription

/// M4.2a — pure DSP, model-free, CI-safe. Exact numerical parity with
/// openai-whisper is asserted end-to-end at M4.2c; here: shapes,
/// finiteness, the slaney filterbank properties, and tone≠silence.
final class LogMelTests: XCTestCase {

    func testMelFilterbankShapeAndProperties() {
        let nMels = 128, nFreqs = LogMel.nFFT / 2 + 1  // 201
        let fb = LogMel.melFilterbank(nMels: nMels)
        XCTAssertEqual(fb.count, nMels * nFreqs)
        XCTAssertTrue(fb.allSatisfy { $0 >= 0 && $0.isFinite })
        // Every mel band must have at least one positive weight.
        for i in 0 ..< nMels {
            let row = fb[i * nFreqs ..< (i + 1) * nFreqs]
            XCTAssertTrue(
                row.contains { $0 > 0 }, "empty mel band \(i)")
        }
        // Low bands sit at low frequencies: band 0's peak bin index is
        // below band (nMels-1)'s peak bin index.
        func peakBin(_ i: Int) -> Int {
            let row = Array(fb[i * nFreqs ..< (i + 1) * nFreqs])
            return row.firstIndex(of: row.max()!) ?? 0
        }
        XCTAssertLessThan(peakBin(0), peakBin(nMels - 1))
    }

    func testLogMelShapeAndSilenceVsTone() throws {
        // MLX ops need the Metal lib (xcodebuild-only; `swift test`
        // lacks it). Gated like the other MLX-requiring tests.
        guard
            ProcessInfo.processInfo.environment["ATHENA_RUN_MODEL_TESTS"]
                == "1"
        else { throw XCTSkip("set ATHENA_RUN_MODEL_TESTS=1 (needs MLX/Metal)") }

        let silence = [Float](repeating: 0, count: LogMel.sampleRate)
        let s = LogMel.logMel(silence)
        XCTAssertEqual(s.shape, [128, LogMel.nFrames])
        let sFlat = s.asArray(Float.self)
        XCTAssertTrue(sFlat.allSatisfy { $0.isFinite })

        // 440 Hz tone, 1 s — padded to 30 s internally.
        let sr = Double(LogMel.sampleRate)
        let tone = (0 ..< LogMel.sampleRate).map {
            Float(0.5 * sin(2.0 * .pi * 440.0 * Double($0) / sr))
        }
        let t = LogMel.logMel(tone)
        XCTAssertEqual(t.shape, [128, LogMel.nFrames])
        let tFlat = t.asArray(Float.self)
        XCTAssertTrue(tFlat.allSatisfy { $0.isFinite })
        // Whisper normalization clamps to an 8-decade window then ÷4, so
        // a signal with both energetic and (padded) silent regions spans
        // exactly 2.0 — a strong check on the log10/clamp/scale chain.
        XCTAssertEqual(
            tFlat.max()! - tFlat.min()!, 2.0, accuracy: 1e-3)
        // The tone carries energy the (uniform-floor) silence doesn't.
        let tEnergy = tFlat.reduce(0, +) / Float(tFlat.count)
        let sEnergy = sFlat.reduce(0, +) / Float(sFlat.count)
        XCTAssertGreaterThan(tEnergy, sEnergy)
    }
}

/// AudioDecode round-trips a generated 16 kHz mono WAV.
final class AudioDecodeTests: XCTestCase {

    func testDecodesGeneratedWavToMono16k() throws {
        let dir = FileManager.default.temporaryDirectory
        let url = dir.appendingPathComponent(
            "athena-ad-\(UUID().uuidString).wav")
        defer { try? FileManager.default.removeItem(at: url) }

        let frames = 8_000  // 0.5 s @ 16 kHz
        guard
            let fmt = AVAudioFormat(
                commonFormat: .pcmFormatFloat32,
                sampleRate: 16_000, channels: 1, interleaved: false)
        else { return XCTFail("format") }
        // Scope the writer so it finalizes (flushes the WAV header /
        // closes the file) before AudioDecode reads the same URL.
        do {
            let file = try AVAudioFile(
                forWriting: url,
                settings: fmt.settings,
                commonFormat: .pcmFormatFloat32, interleaved: false)
            guard
                let buf = AVAudioPCMBuffer(
                    pcmFormat: fmt, frameCapacity: AVAudioFrameCount(frames))
            else { return XCTFail("buffer") }
            buf.frameLength = AVAudioFrameCount(frames)
            for i in 0 ..< frames {
                buf.floatChannelData![0][i] =
                    Float(
                        0.25 * sin(2.0 * .pi * 440.0 * Double(i) / 16_000.0))
            }
            try file.write(from: buf)
        }

        let pcm = try AudioDecode.pcm16kMono(from: url)
        // ~0.5 s of 16 kHz samples (codec priming may shift a little).
        XCTAssertEqual(Double(pcm.count), 8_000, accuracy: 256)
        XCTAssertTrue(pcm.contains { abs($0) > 0.05 }, "decoded silence?")
        XCTAssertTrue(pcm.allSatisfy { $0.isFinite && abs($0) <= 1.5 })
    }
}
