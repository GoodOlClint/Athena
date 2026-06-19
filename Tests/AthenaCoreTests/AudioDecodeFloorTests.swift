import AthenaCore
import Foundation
import XCTest

@testable import AthenaTranscription

/// The shared decode floor (defense-in-depth for the "degenerate audio aborts
/// the daemon" class): a decode below the 0.1 s minimum is rejected ONCE at the
/// `AudioDecode` chokepoint with a uniform 400 `audio_too_short`, so no audio
/// route hands a degenerate clip to a model conv. AVFoundation file I/O is CPU
/// (no Metal), so this runs in the logic tier.
final class AudioDecodeFloorTests: XCTestCase {

    /// Write a minimal 16 kHz mono PCM16 WAV of `samples` frames (silence) — a
    /// real container AVAudioFile reads. 16 k→16 k convert is identity, so the
    /// decoded frame count matches.
    private func writeWAV(samples: Int) throws -> URL {
        let sr: UInt32 = 16_000
        let bits: UInt16 = 16
        let ch: UInt16 = 1
        let dataBytes = UInt32(samples) * UInt32(ch) * UInt32(bits / 8)
        var d = Data()
        func u32(_ v: UInt32) { withUnsafeBytes(of: v.littleEndian) { d.append(contentsOf: $0) } }
        func u16(_ v: UInt16) { withUnsafeBytes(of: v.littleEndian) { d.append(contentsOf: $0) } }
        d.append(contentsOf: Array("RIFF".utf8)); u32(36 + dataBytes)
        d.append(contentsOf: Array("WAVE".utf8))
        d.append(contentsOf: Array("fmt ".utf8)); u32(16); u16(1); u16(ch)
        u32(sr); u32(sr * UInt32(ch) * UInt32(bits / 8))
        u16(ch * bits / 8); u16(bits)
        d.append(contentsOf: Array("data".utf8)); u32(dataBytes)
        d.append(Data(count: Int(dataBytes)))  // silence
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("adf-\(UUID().uuidString).wav")
        try d.write(to: url)
        return url
    }

    func testSubFloorClipIsClassified400NotProcessed() throws {
        let url = try writeWAV(samples: 320)  // 0.02 s — an accidental capture
        defer { try? FileManager.default.removeItem(at: url) }
        XCTAssertThrowsError(
            try AudioDecode.pcm16kMono(from: url, module: .transcription)
        ) { err in
            guard let e = err as? AthenaError else {
                return XCTFail("expected AthenaError, got \(err)")
            }
            XCTAssertEqual(e.code, "audio_too_short")
            XCTAssertEqual(e.httpStatus, 400)
            XCTAssertEqual(e.type, "invalid_request_error")
        }
    }

    func testEmptyClipIsAlsoTooShort() throws {
        let url = try writeWAV(samples: 0)
        defer { try? FileManager.default.removeItem(at: url) }
        XCTAssertThrowsError(
            try AudioDecode.pcm16kMono(from: url, module: .diarization)
        ) { err in
            XCTAssertEqual((err as? AthenaError)?.code, "audio_too_short")
        }
    }

    func testAboveFloorClipDecodesNormally() throws {
        let url = try writeWAV(samples: 8000)  // 0.5 s — well above the floor
        defer { try? FileManager.default.removeItem(at: url) }
        let pcm = try AudioDecode.pcm16kMono(from: url, module: .transcription)
        XCTAssertGreaterThan(pcm.count, AudioDecode.defaultMinSamples)
    }

    func testFloorIsDisabledByMinSamplesZero() throws {
        let url = try writeWAV(samples: 320)
        defer { try? FileManager.default.removeItem(at: url) }
        // A caller that opts out (minSamples: 0) still gets the short PCM.
        let pcm = try AudioDecode.pcm16kMono(from: url, minSamples: 0)
        XCTAssertGreaterThan(pcm.count, 0)
        XCTAssertLessThan(pcm.count, AudioDecode.defaultMinSamples)
    }

    // MARK: error shape

    func testAudioTooShortErrorShape() {
        let e = AthenaError.audioTooShort(
            module: .transcription, seconds: 0.02, minSeconds: 0.1)
        XCTAssertEqual(e.httpStatus, 400)
        XCTAssertEqual(e.code, "audio_too_short")
        XCTAssertEqual(e.type, "invalid_request_error")
        XCTAssertTrue(e.message.contains("0.02"), e.message)
        XCTAssertTrue(e.message.contains("0.10"), e.message)
    }
}
