import AVFoundation
import Foundation

/// Decode an audio file (any format AVFoundation reads — wav/mp3/m4a/
/// flac/…) to the mono 16 kHz Float32 PCM Whisper expects. Pure
/// AVFoundation; no MLX. M4.2a.
public enum AudioDecode {
    public static let sampleRate = 16_000

    /// Hard ceiling on decoded samples (~4 h @ 16 kHz ≈ 0.9 GB of
    /// `Float`). Bounds a decompression-bomb file (a tiny crafted input
    /// that decodes to an enormous PCM stream) before it exhausts host
    /// memory (D4). Comfortably above any legitimate transcription clip,
    /// including the 1-hour streaming path; overridable per call.
    public static let defaultMaxSamples = sampleRate * 3600 * 4

    public enum DecodeError: Error, CustomStringConvertible {
        case open(String)
        case converterInit
        case convert(String)
        case tooLong(maxSamples: Int)
        public var description: String {
            switch self {
            case .open(let s): return "audio open failed: \(s)"
            case .converterInit: return "audio converter init failed"
            case .convert(let s): return "audio convert failed: \(s)"
            case .tooLong(let m):
                return "audio exceeds the \(m)-sample (~\(m / sampleRate)s) "
                    + "decode limit"
            }
        }
    }

    /// Decode `url` → `[Float]` mono @ 16 kHz, range ~[-1, 1]. Decoding
    /// stops with `.tooLong` once `maxSamples` is exceeded.
    public static func pcm16kMono(
        from url: URL, maxSamples: Int = defaultMaxSamples
    ) throws -> [Float] {
        let file: AVAudioFile
        do {
            file = try AVAudioFile(forReading: url)
        } catch {
            throw DecodeError.open(error.localizedDescription)
        }
        let inFormat = file.processingFormat
        guard
            let outFormat = AVAudioFormat(
                commonFormat: .pcmFormatFloat32,
                sampleRate: Double(sampleRate),
                channels: 1,
                interleaved: false)
        else { throw DecodeError.converterInit }

        guard let converter = AVAudioConverter(from: inFormat, to: outFormat)
        else { throw DecodeError.converterInit }

        // Pull the whole file through the converter. The input block
        // feeds source frames until the file is drained, then signals
        // end-of-stream. `AVAudioConverter.convert` invokes the block
        // synchronously on this thread, so the box is single-threaded
        // despite the @Sendable signature.
        let srcCapacity: AVAudioFrameCount = 1 << 16
        final class State: @unchecked Sendable { var finished = false }
        let state = State()
        let inputBlock: AVAudioConverterInputBlock = { _, outStatus in
            if state.finished {
                outStatus.pointee = .endOfStream
                return nil
            }
            guard
                let buf = AVAudioPCMBuffer(
                    pcmFormat: inFormat, frameCapacity: srcCapacity)
            else {
                outStatus.pointee = .endOfStream
                return nil
            }
            do {
                try file.read(into: buf, frameCount: srcCapacity)
            } catch {
                outStatus.pointee = .endOfStream
                return nil
            }
            if buf.frameLength == 0 {
                state.finished = true
                outStatus.pointee = .endOfStream
                return nil
            }
            outStatus.pointee = .haveData
            return buf
        }

        var out: [Float] = []
        // Generous reserve: input frames scaled by the resample ratio.
        // Guard the header-driven estimate — a crafted `file.length` or a
        // zero/tiny input sample rate could make the Double non-finite or
        // overflow the `Int` cast (a trap) or reserve gigabytes. Clamp to
        // `maxSamples` and never feed a non-finite value to `Int()` (D4).
        let inSR = inFormat.sampleRate > 0 ? inFormat.sampleRate : Double(sampleRate)
        let estimate = (Double(file.length) * Double(sampleRate) / inSR).rounded()
        let reserve = estimate.isFinite
            ? Int(min(estimate, Double(maxSamples))) : maxSamples
        out.reserveCapacity(reserve + sampleRate)

        let chunkFrames: AVAudioFrameCount = 1 << 15
        while true {
            guard
                let outBuf = AVAudioPCMBuffer(
                    pcmFormat: outFormat, frameCapacity: chunkFrames)
            else { throw DecodeError.converterInit }

            var err: NSError?
            let status = converter.convert(
                to: outBuf, error: &err, withInputFrom: inputBlock)
            if let err {
                throw DecodeError.convert(err.localizedDescription)
            }
            if let ch = outBuf.floatChannelData, outBuf.frameLength > 0 {
                out.append(
                    contentsOf: UnsafeBufferPointer(
                        start: ch[0], count: Int(outBuf.frameLength)))
                // Decompression-bomb guard: stop as soon as the decoded
                // stream passes the ceiling (D4).
                if out.count > maxSamples {
                    throw DecodeError.tooLong(maxSamples: maxSamples)
                }
            }
            if status == .endOfStream || status == .error { break }
            if status == .inputRanDry && outBuf.frameLength == 0 { break }
        }
        return out
    }
}
