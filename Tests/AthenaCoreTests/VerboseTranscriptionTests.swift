import Foundation
import MLX
import XCTest

@testable import AthenaTranscription

final class VerboseTranscriptionTests: XCTestCase {

    private func json(_ r: TranscriptionResult) throws -> [String: Any] {
        let data = try JSONEncoder().encode(
            VerboseTranscriptionResponse.make(
                r, wantWords: false, speaker: { _ in nil }))
        return try XCTUnwrap(
            JSONSerialization.jsonObject(with: data) as? [String: Any])
    }

    func testWhisperSegmentEncodesNewFields() throws {
        let r = TranscriptionResult(
            text: "hi", language: "en", duration: 1,
            segments: [
                TranscriptionSegment(
                    start: 0, end: 1, text: "hi", avgLogprob: -0.5,
                    noSpeechProb: 0.25, compressionRatio: 1.5)
            ], model: "whisper-x")
        let o = try json(r)
        XCTAssertEqual(o["model"] as? String, "whisper-x")
        let seg = try XCTUnwrap((o["segments"] as? [[String: Any]])?.first)
        XCTAssertEqual(seg["no_speech_prob"] as? Double, 0.25)
        XCTAssertEqual(seg["compression_ratio"] as? Double, 1.5)
        XCTAssertEqual(seg["avg_logprob"] as? Double, -0.5)
    }

    func testParakeetSegmentsKeepAvgLogprobOmitWhisperOnlyFields() throws {
        let segs = ParakeetAlignment.segments(
            from: [
                ParakeetAlignment.Token(
                    id: 1, text: " Hello", start: 0, duration: 0.16,
                    confidence: 0.5)
            ], attachWords: false)
        let r = TranscriptionResult(
            text: "Hello", language: "auto", duration: 1, segments: segs,
            model: "parakeet-x")
        let o = try json(r)
        XCTAssertEqual(o["model"] as? String, "parakeet-x")
        let seg = try XCTUnwrap((o["segments"] as? [[String: Any]])?.first)
        XCTAssertEqual(
            try XCTUnwrap(seg["avg_logprob"] as? Double), log(0.5),
            accuracy: 1e-9)
        for k in ["no_speech_prob", "compression_ratio"] {
            XCTAssertNil(seg[k], "\(k) must be omitted, not zero")
        }
    }

    func testSubtitleFormatsIgnoreNewFields() {
        let plain = [
            TranscriptionSegment(start: 0, end: 2, text: "a"),
            TranscriptionSegment(start: 2, end: 3, text: "b"),
        ]
        let rich = [
            TranscriptionSegment(
                start: 0, end: 2, text: "a", avgLogprob: -1,
                noSpeechProb: 0.5, compressionRatio: 2),
            TranscriptionSegment(
                start: 2, end: 3, text: "b", avgLogprob: -1,
                noSpeechProb: 0.5, compressionRatio: 2),
        ]
        XCTAssertEqual(
            TranscriptionFormat.srt(plain), TranscriptionFormat.srt(rich))
        XCTAssertEqual(
            TranscriptionFormat.vtt(plain), TranscriptionFormat.vtt(rich))
    }

    func testCompressionRatio() throws {
        let rep = String(repeating: "hello world ", count: 100)
        let ratio = try XCTUnwrap(WhisperDecode.compressionRatio(rep))
        XCTAssertGreaterThan(ratio, 2.4)
        let varied = try XCTUnwrap(
            WhisperDecode.compressionRatio(
                "The committee adjourned shortly after noon."))
        XCTAssertLessThan(varied, 2.4)
        XCTAssertNil(WhisperDecode.compressionRatio(""))
        // Non-ASCII: 600 characters but 1500 UTF-8 bytes. Bytes give a ratio
        // near 44; a character count would give near 17.6.
        let wide =
            String(repeating: "é", count: 300)
            + String(repeating: "日本語", count: 100)
        XCTAssertGreaterThan(
            try XCTUnwrap(WhisperDecode.compressionRatio(wide)), 30)
    }

    func testNoSpeechProb() throws {
        let n = 51_866
        let uniform = [Float](repeating: 0, count: n)
        XCTAssertEqual(
            try XCTUnwrap(WhisperDecode.noSpeechProb(uniform)),
            1.0 / Double(n), accuracy: 1e-9)
        var peaked = [Float](repeating: 0, count: n)
        peaked[50_363] = 50
        XCTAssertGreaterThan(
            try XCTUnwrap(WhisperDecode.noSpeechProb(peaked)), 0.999)
        XCTAssertNil(WhisperDecode.noSpeechProb([]))
        XCTAssertNil(WhisperDecode.noSpeechProb([Float](repeating: 0, count: 50_000)))
    }
}

final class WhisperNoSpeechIntegrationTests: XCTestCase {

    /// The shipped `noSpeechProb` must equal the `<|nospeech|>` probability
    /// of the sot-position logits from an independent forward pass. (The
    /// turbo checkpoint emits ~1e-10 for silence and speech alike, so a
    /// threshold on silence cannot pin the read; position parity can: the
    /// last-position value differs by ~50x.)
    func testSegmentNoSpeechProbMatchesSotPositionLogits() async throws {
        guard
            ProcessInfo.processInfo.environment["ATHENA_RUN_MODEL_TESTS"]
                == "1"
        else { throw XCTSkip("set ATHENA_RUN_MODEL_TESTS=1 (heavy)") }
        let dir = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".athena/models/whisper-large-v3-turbo")
        guard FileManager.default.fileExists(atPath: dir.path) else {
            throw XCTSkip("whisper-large-v3-turbo not in the model store")
        }
        let model = try WhisperLoader.load(directory: dir)
        let tokenizer = try await WhisperLoader.loadTokenizer()

        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("athena-nsp-\(UUID()).aiff")
        defer { try? FileManager.default.removeItem(at: url) }
        let p = Process()
        p.executableURL = URL(fileURLWithPath: "/usr/bin/say")
        p.arguments = [
            "-o", url.path, "the quick brown fox jumps over the lazy dog",
        ]
        try p.run()
        p.waitUntilExit()
        let pcm = try AudioDecode.pcm16kMono(from: url)

        let result = WhisperDecode.transcribeResult(
            model: model, pcm: pcm, tokenizer: tokenizer, language: "en")
        let got = try XCTUnwrap(result.segments.first?.noSpeechProb)
        XCTAssertLessThan(got, 0.5)

        let audio = model.embedAudio(
            LogMel.logMel(Array(pcm.prefix(LogMel.nSamples))))
        audio.eval()
        let prefix = MLXArray(
            [
                Int32(WhisperDecode.sot), Int32(WhisperDecode.langBase),
                Int32(WhisperDecode.transcribe),
            ], [1, 3])
        let lg = model.logits(prefix, audio: audio, offset: 0, cache: nil)
        let want = try XCTUnwrap(
            WhisperDecode.noSpeechProb(lg[0, 0, 0...].asArray(Float.self)))
        XCTAssertEqual(got / want, 1.0, accuracy: 0.01)
    }
}
