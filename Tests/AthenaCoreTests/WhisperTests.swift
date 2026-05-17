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
