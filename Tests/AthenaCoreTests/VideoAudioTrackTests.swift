import AthenaCore
import Foundation
import XCTest

@testable import AthenaTranscription

/// ADR 022 / M78.1 S1 — the video audio-track extractor. The AVAssetReader path
/// is exercised in the logic tier against a synthesized WAV (a real container
/// with an audio track; AVFoundation is CPU, no Metal), and the shared
/// floor/ceiling verdict is unit-pinned. A real video container is covered by a
/// gated fixture hook. No transcription accuracy is asserted here — extraction
/// is plumbing, so a synthesized signal is the correct fixture.
final class VideoAudioTrackTests: XCTestCase {

    // MARK: shared bounds verdict (pure, MLX-free)

    func testSampleBoundError() {
        // below floor → tooShort
        if case .tooShort(let n, let m)? = AudioDecode.sampleBoundError(
            count: 320, minSamples: 1600, maxSamples: 10_000)
        {
            XCTAssertEqual(n, 320)
            XCTAssertEqual(m, 1600)
        } else {
            XCTFail("expected .tooShort")
        }
        // above ceiling → tooLong
        if case .tooLong(let m)? = AudioDecode.sampleBoundError(
            count: 20_000, minSamples: 1600, maxSamples: 10_000)
        {
            XCTAssertEqual(m, 10_000)
        } else {
            XCTFail("expected .tooLong")
        }
        // in range → nil
        XCTAssertNil(
            AudioDecode.sampleBoundError(
                count: 5000, minSamples: 1600, maxSamples: 10_000))
        // floor disabled (0) → nil even for a tiny count
        XCTAssertNil(
            AudioDecode.sampleBoundError(
                count: 1, minSamples: 0, maxSamples: 10_000))
    }

    // MARK: AVAssetReader extraction (logic tier, synthesized WAV)

    /// Minimal 16 kHz mono PCM16 WAV of `samples` frames of a quiet tone — a
    /// real container `AVAssetReader` reads (and `AVAudioFile` could too; the
    /// point is the extractor's track-read path, identical for a video track).
    private func writeWAV(samples: Int) throws -> URL {
        let sr: UInt32 = 16_000
        let bits: UInt16 = 16, ch: UInt16 = 1
        let dataBytes = UInt32(samples) * UInt32(bits / 8)
        var d = Data()
        func u32(_ v: UInt32) { withUnsafeBytes(of: v.littleEndian) { d.append(contentsOf: $0) } }
        func u16(_ v: UInt16) { withUnsafeBytes(of: v.littleEndian) { d.append(contentsOf: $0) } }
        d.append(contentsOf: Array("RIFF".utf8)); u32(36 + dataBytes)
        d.append(contentsOf: Array("WAVE".utf8))
        d.append(contentsOf: Array("fmt ".utf8)); u32(16); u16(1); u16(ch)
        u32(sr); u32(sr * UInt32(bits / 8)); u16(bits / 8); u16(bits)
        d.append(contentsOf: Array("data".utf8)); u32(dataBytes)
        // A faint tone so the samples aren't all-zero (extraction is unaffected).
        var pcm = Data(capacity: Int(dataBytes))
        for i in 0 ..< samples {
            let v = Int16(truncatingIfNeeded: (i % 64) - 32)
            withUnsafeBytes(of: v.littleEndian) { pcm.append(contentsOf: $0) }
        }
        d.append(pcm)
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("vat-\(UUID().uuidString).wav")
        try d.write(to: url)
        return url
    }

    func testExtractsPCMFromAudioContainer() async throws {
        let url = try writeWAV(samples: 8000)  // 0.5 s
        defer { try? FileManager.default.removeItem(at: url) }
        let pcm = try await VideoAudioTrack.extractPCM(from: url)
        // 16 k→16 k is identity; allow a small resampler delta.
        XCTAssertGreaterThan(pcm.count, 7000)
        XCTAssertLessThan(pcm.count, 9000)
    }

    func testSubFloorTrackIsAudioTooShort() async throws {
        let url = try writeWAV(samples: 320)  // 0.02 s
        defer { try? FileManager.default.removeItem(at: url) }
        do {
            _ = try await VideoAudioTrack.extractPCM(from: url)
            XCTFail("expected audio_too_short")
        } catch let e as AthenaError {
            XCTAssertEqual(e.code, "audio_too_short")
            XCTAssertEqual(e.httpStatus, 400)
        }
    }

    func testMissingFileIsClassified() async throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("vat-missing-\(UUID().uuidString).mp4")
        do {
            _ = try await VideoAudioTrack.extractPCM(from: url)
            XCTFail("expected a classified AthenaError")
        } catch let e as AthenaError {
            // Unreadable container ⇒ invalidAudio (or no-track) — both 400.
            XCTAssertEqual(e.httpStatus, 400)
        }
    }

    // MARK: real video container (gated)

    /// Point `ATHENA_VIDEO_FIXTURE` at a real `.mp4`/`.mov` with an audio track
    /// to exercise the actual video-container path (which `AVAudioFile` cannot
    /// open). Skipped when unset, so CI/logic-tier is unaffected.
    func testExtractsFromRealVideoFixture() async throws {
        guard let path = ProcessInfo.processInfo.environment["ATHENA_VIDEO_FIXTURE"]
        else { throw XCTSkip("set ATHENA_VIDEO_FIXTURE to a real video file") }
        let url = URL(fileURLWithPath: path)
        let pcm = try await VideoAudioTrack.extractPCM(from: url)
        XCTAssertGreaterThan(pcm.count, AudioDecode.defaultMinSamples)
    }

    // MARK: error shape

    func testVideoNoAudioTrackErrorShape() {
        let e = AthenaError.videoNoAudioTrack(module: .transcription)
        XCTAssertEqual(e.httpStatus, 400)
        XCTAssertEqual(e.code, "video_no_audio_track")
        XCTAssertEqual(e.type, "invalid_request_error")
        XCTAssertTrue(e.message.lowercased().contains("no audio track"))
    }
}
