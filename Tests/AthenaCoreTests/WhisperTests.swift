import Foundation
import MLX
import XCTest

@testable import AthenaTranscription

/// M4.2b — sinusoids are pure (CI-safe); the load + encoder/decoder
/// forward needs the model + Metal, so it's gated and validated via
/// `xcodebuild test` (downloads ~1.5 GB to the HF cache on first run).
final class WhisperSinusoidsTests: XCTestCase {

    func testSinusoidShape() throws {
        guard
            ProcessInfo.processInfo.environment["ATHENA_RUN_MODEL_TESTS"]
                == "1"
        else { throw XCTSkip("set ATHENA_RUN_MODEL_TESTS=1 (needs MLX/Metal)") }
        let s = whisperSinusoids(length: 1500, channels: 1280)
        XCTAssertEqual(s.shape, [1500, 1280])
        let f = s.asArray(Float.self)
        XCTAssertTrue(f.allSatisfy { $0.isFinite && abs($0) <= 1.0001 })
    }
}

/// M4.2c — full pipeline on macOS `say` TTS of a known sentence:
/// audio → log-mel → Whisper greedy decode → text. Gated + heavy.
final class WhisperTranscribeIntegrationTests: XCTestCase {

    func testTranscribesKnownUtterance() async throws {
        guard
            ProcessInfo.processInfo.environment["ATHENA_RUN_MODEL_TESTS"]
                == "1"
        else { throw XCTSkip("set ATHENA_RUN_MODEL_TESTS=1 (heavy)") }

        let sentence = "the quick brown fox jumps over the lazy dog"
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("athena-say-\(UUID()).aiff")
        defer { try? FileManager.default.removeItem(at: url) }
        let p = Process()
        p.executableURL = URL(fileURLWithPath: "/usr/bin/say")
        p.arguments = ["-o", url.path, sentence]
        try p.run()
        p.waitUntilExit()
        XCTAssertEqual(p.terminationStatus, 0, "say failed")

        let pcm = try AudioDecode.pcm16kMono(from: url)
        let mel = LogMel.logMel(pcm)
        let model = try await WhisperLoader.load()
        let tokenizer = try await WhisperLoader.loadTokenizer()
        // language: nil ⇒ exercises M4.2e-1 auto-detection (English).
        let text = WhisperDecode.transcribe(
            model: model, mel: mel, tokenizer: tokenizer, language: nil)

        let norm = text.lowercased().filter {
            $0.isLetter || $0.isWhitespace
        }
        XCTAssertTrue(
            norm.contains("quick brown fox"),
            "got: \(text)")
        XCTAssertTrue(norm.contains("lazy dog"), "got: \(text)")
    }
}

final class WhisperLoadIntegrationTests: XCTestCase {

    func testLoadsAndRunsEncoderDecoder() async throws {
        guard
            ProcessInfo.processInfo.environment["ATHENA_RUN_MODEL_TESTS"]
                == "1"
        else { throw XCTSkip("set ATHENA_RUN_MODEL_TESTS=1 (heavy ~1.5GB)") }

        let model = try await WhisperLoader.load()
        XCTAssertEqual(model.config.n_audio_state, 1280)
        XCTAssertEqual(model.config.n_text_layer, 4)
        XCTAssertEqual(model.config.n_vocab, 51_866)

        // log-mel of 1 s silence (padded to 30 s) → [128, 3000]
        let mel = LogMel.logMel(
            [Float](repeating: 0, count: LogMel.sampleRate))
        let audio = model.embedAudio(mel)
        audio.eval()
        XCTAssertEqual(audio.shape, [1, 1_500, 1_280])
        XCTAssertTrue(
            audio.asArray(Float.self).prefix(64).allSatisfy {
                $0.isFinite
            })

        // two arbitrary valid token ids → logits over the vocab
        let toks = MLXArray([Int32(0), Int32(1)]).reshaped([1, 2])
        let lg = model.logits(toks, audio: audio)
        lg.eval()
        XCTAssertEqual(lg.shape, [1, 2, 51_866])
        XCTAssertTrue(
            lg.asArray(Float.self).prefix(64).allSatisfy { $0.isFinite })
    }
}
