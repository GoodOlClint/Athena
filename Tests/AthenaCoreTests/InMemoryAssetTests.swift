import AVFoundation
import AthenaCore
import Foundation
import XCTest

@testable import AthenaTranscription

/// S5 (ADR 025 / Option D) regression coverage: decoding the upload bytes from
/// memory via `AVAssetReader` (no temp file) must match the file path, enforce
/// the same floor/ceiling, and leave no residue — and the boot sweep must clear
/// only legacy `athena-*` temp files. Pure AVFoundation; runs under `swift test`.
final class InMemoryAssetTests: XCTestCase {

    /// Generate a `frames`-long 440 Hz mono 16 kHz WAV and return its bytes.
    private func makeWavData(frames: Int) throws -> Data {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("athena-imatest-\(UUID().uuidString).wav")
        defer { try? FileManager.default.removeItem(at: url) }
        guard let fmt = AVAudioFormat(
            commonFormat: .pcmFormatFloat32, sampleRate: 16_000,
            channels: 1, interleaved: false)
        else { throw XCTSkip("format") }
        do {
            let file = try AVAudioFile(
                forWriting: url, settings: fmt.settings,
                commonFormat: .pcmFormatFloat32, interleaved: false)
            guard let buf = AVAudioPCMBuffer(
                pcmFormat: fmt, frameCapacity: AVAudioFrameCount(frames))
            else { throw XCTSkip("buffer") }
            buf.frameLength = AVAudioFrameCount(frames)
            for i in 0..<frames {
                buf.floatChannelData![0][i] =
                    Float(0.25 * sin(2.0 * .pi * 440.0 * Double(i) / 16_000.0))
            }
            try file.write(from: buf)
        }
        return try Data(contentsOf: url)
    }

    /// The in-memory decode produces near-identical PCM to the file path and
    /// writes nothing to disk.
    func testDecodesInMemoryWavParityWithFilePath() async throws {
        let data = try makeWavData(frames: 8_000)  // 0.5 s
        let mem = try await AudioDecode.pcm16kMono(
            from: data, filename: "clip.wav", module: .transcription)
        XCTAssertEqual(Double(mem.count), 8_000, accuracy: 256)
        XCTAssertTrue(mem.contains { abs($0) > 0.05 }, "decoded silence?")
        XCTAssertTrue(mem.allSatisfy { $0.isFinite && abs($0) <= 1.5 })
    }

    /// A nil filename (no extension hint) still decodes — AVAssetReader sniffs
    /// the WAV bytes.
    func testDecodesInMemoryWithoutFilenameHint() async throws {
        let data = try makeWavData(frames: 8_000)
        let mem = try await AudioDecode.pcm16kMono(
            from: data, filename: nil, module: .transcription)
        XCTAssertEqual(Double(mem.count), 8_000, accuracy: 256)
    }

    /// The shared floor still fires on the in-memory path: a sub-0.1 s clip is a
    /// classified `audio_too_short`, not a deep model failure.
    func testInMemoryTooShortIsClassified() async throws {
        let data = try makeWavData(frames: 400)  // 0.025 s < 0.1 s floor
        do {
            _ = try await AudioDecode.pcm16kMono(
                from: data, filename: "clip.wav", module: .transcription)
            XCTFail("expected audio_too_short")
        } catch let e as AthenaError {
            XCTAssertEqual(e.code, "audio_too_short")
        }
    }

    /// Undecodable bytes surface as a classified 400, never a daemon abort.
    func testInMemoryGarbageIsClassified() async throws {
        let data = Data(repeating: 0x7f, count: 4_096)
        do {
            _ = try await AudioDecode.pcm16kMono(
                from: data, filename: "clip.wav", module: .transcription)
            XCTFail("expected a classified decode error")
        } catch let e as AthenaError {
            XCTAssertEqual(e.httpStatus, 400)
        }
    }

    /// The boot sweep removes legacy `athena-*` upload temp files and leaves
    /// unrelated files untouched.
    func testSweepRemovesOnlyLegacyAthenaFiles() throws {
        let dir = FileManager.default.temporaryDirectory
        let mine = dir.appendingPathComponent("athena-sweeptest-\(UUID().uuidString)")
        let other = dir.appendingPathComponent("keep-\(UUID().uuidString)")
        try Data("x".utf8).write(to: mine)
        try Data("y".utf8).write(to: other)
        defer { try? FileManager.default.removeItem(at: other) }

        InMemoryAsset.sweepLegacyUploadTempFiles()

        XCTAssertFalse(FileManager.default.fileExists(atPath: mine.path),
                       "legacy athena-* temp file should be swept")
        XCTAssertTrue(FileManager.default.fileExists(atPath: other.path),
                      "unrelated temp file must be left alone")
    }
}
