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

/// M4.2e-2 — audio longer than 30 s must be chunked, not truncated.
/// A ~50 s clip with a distinctive opening and closing word: both must
/// survive, proving ≥2 windows were decoded and concatenated.
final class WhisperChunkingIntegrationTests: XCTestCase {

    func testChunksLongAudioBeyond30s() async throws {
        guard
            ProcessInfo.processInfo.environment["ATHENA_RUN_MODEL_TESTS"]
                == "1"
        else { throw XCTSkip("set ATHENA_RUN_MODEL_TESTS=1 (heavy)") }

        let filler = Array(
            repeating:
                "one two three four five six seven eight nine ten,",
            count: 14
        ).joined(separator: " ")
        let sentence =
            "alpha is the opening word. \(filler) "
            + "omega is the closing word."
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("athena-long-\(UUID()).aiff")
        defer { try? FileManager.default.removeItem(at: url) }
        let p = Process()
        p.executableURL = URL(fileURLWithPath: "/usr/bin/say")
        p.arguments = ["-r", "170", "-o", url.path, sentence]
        try p.run()
        p.waitUntilExit()

        let pcm = try AudioDecode.pcm16kMono(from: url)
        XCTAssertGreaterThan(
            pcm.count, LogMel.nSamples,
            "clip must exceed 30 s to exercise chunking")

        let model = try await WhisperLoader.load()
        let tokenizer = try await WhisperLoader.loadTokenizer()
        let text = WhisperDecode.transcribe(
            model: model, pcm: pcm, tokenizer: tokenizer, language: "en")
        let norm = text.lowercased().filter {
            $0.isLetter || $0.isWhitespace
        }
        XCTAssertTrue(norm.contains("alpha"), "lost window 0: \(text)")
        XCTAssertTrue(norm.contains("omega"), "lost last window: \(text)")
    }
}

/// M26.2 — cross-attention DTW word timestamps on a real ICSI meeting
/// clip. Validates the acceptance contract: words exist, are globally
/// monotonic, fall within the audio, and each segment's attached words
/// lie within that segment's bounds. Heavy + gated; skips if the sample
/// clip is absent.
final class WhisperWordTimestampIntegrationTests: XCTestCase {

    func testWordTimesAreMonotonicAndWithinBounds() async throws {
        guard
            ProcessInfo.processInfo.environment["ATHENA_RUN_MODEL_TESTS"]
                == "1"
        else { throw XCTSkip("set ATHENA_RUN_MODEL_TESTS=1 (heavy)") }
        let clip = URL(fileURLWithPath: "/tmp/audio/clip60.wav")
        guard FileManager.default.fileExists(atPath: clip.path) else {
            throw XCTSkip("sample clip /tmp/audio/clip60.wav not present")
        }

        let pcm = try AudioDecode.pcm16kMono(from: clip)
        let duration = Double(pcm.count) / Double(LogMel.sampleRate)
        let model = try await WhisperLoader.load()
        let tokenizer = try await WhisperLoader.loadTokenizer()
        // Confirm the dropped alignment table is now loaded.
        XCTAssertFalse(
            model.alignmentHeads.isEmpty, "alignment_heads not loaded")

        let result = WhisperDecode.transcribeResult(
            model: model, pcm: pcm, tokenizer: tokenizer,
            language: "en", wordTimestamps: true)

        XCTAssertFalse(result.words.isEmpty, "no words aligned")
        let eps = 0.05
        var prev = -1.0
        for w in result.words {
            XCTAssertLessThanOrEqual(w.start, w.end, "word \(w.word)")
            XCTAssertGreaterThanOrEqual(w.start, -eps, "word \(w.word)")
            XCTAssertLessThanOrEqual(
                w.end, duration + 1.0, "word \(w.word) past end")
            XCTAssertGreaterThanOrEqual(
                w.start, prev - eps, "non-monotonic at \(w.word)")
            prev = w.start
            XCTAssertGreaterThanOrEqual(w.probability, 0)
            XCTAssertLessThanOrEqual(w.probability, 1.0001)
        }
        // Each segment's words are clamped into that segment, so they
        // sit within its bounds (± float tolerance).
        let tol = 0.05
        for seg in result.segments {
            for w in seg.words ?? [] {
                XCTAssertGreaterThanOrEqual(
                    w.start, seg.start - tol,
                    "word '\(w.word)' before seg")
                XCTAssertLessThanOrEqual(
                    w.end, seg.end + tol, "word '\(w.word)' after seg")
            }
        }
    }
}

/// M50.1 — regression for the allocator-pool leak class M46.6 caught
/// in the embedder. Drives many short transcription calls back-to-back
/// and asserts MLX's pool stays bounded — without the per-window /
/// per-token clears, the pool grows N-proportionally with call count
/// and the test fails by a comfortable margin. Gated + heavy.
final class WhisperMemoryRegressionTests: XCTestCase {

    func testTranscribePoolStaysBoundedAcrossManyCalls() async throws {
        guard
            ProcessInfo.processInfo.environment["ATHENA_RUN_MODEL_TESTS"]
                == "1"
        else { throw XCTSkip("set ATHENA_RUN_MODEL_TESTS=1 (heavy)") }

        let model = try await WhisperLoader.load()
        let tokenizer = try await WhisperLoader.loadTokenizer()
        // 1 s silence padded to 30 s by `logMel`. Silence exercises the
        // encoder + KV-cache decode + clearCache paths without needing
        // real audio; the leak class is allocator-pool growth from the
        // forward pass, independent of decoded content.
        let pcm = [Float](repeating: 0, count: LogMel.sampleRate)

        // Warmup so first-call lazy allocations settle, then clear the
        // pool so the baseline reflects steady-state.
        _ = WhisperDecode.transcribe(
            model: model, mel: LogMel.logMel(pcm),
            tokenizer: tokenizer, language: "en")
        MLX.Memory.clearCache()
        let baseline = MLX.Memory.cacheMemory

        for _ in 0 ..< 22 {
            _ = WhisperDecode.transcribe(
                model: model, mel: LogMel.logMel(pcm),
                tokenizer: tokenizer, language: "en")
        }

        let after = MLX.Memory.cacheMemory
        // Without M50.1's clears, `after` scales with the per-call
        // encoder pool delta × 22 (hundreds of MB → several GB).
        // The pool's allowed to hold one in-flight transient; the
        // generous 512 MB ceiling catches an N-proportional leak by
        // a wide margin while tolerating a single residual.
        let ceiling = 512 * 1024 * 1024
        XCTAssertLessThan(
            after - baseline, ceiling,
            "MLX cache pool drifted \(after - baseline) bytes "
                + "above baseline after 22 transcribes (M50.1 leak)")
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
