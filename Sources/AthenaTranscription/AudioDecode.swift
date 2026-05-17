import AVFoundation
import Foundation

/// Decode an audio file (any format AVFoundation reads — wav/mp3/m4a/
/// flac/…) to the mono 16 kHz Float32 PCM Whisper expects. Pure
/// AVFoundation; no MLX. M4.2a.
public enum AudioDecode {
    public static let sampleRate = 16_000

    public enum DecodeError: Error, CustomStringConvertible {
        case open(String)
        case converterInit
        case convert(String)
        public var description: String {
            switch self {
            case .open(let s): return "audio open failed: \(s)"
            case .converterInit: return "audio converter init failed"
            case .convert(let s): return "audio convert failed: \(s)"
            }
        }
    }

    /// Decode `url` → `[Float]` mono @ 16 kHz, range ~[-1, 1].
    public static func pcm16kMono(from url: URL) throws -> [Float] {
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
        out.reserveCapacity(
            Int(Double(file.length) * Double(sampleRate)
                / inFormat.sampleRate) + sampleRate)

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
            }
            if status == .endOfStream || status == .error { break }
            if status == .inputRanDry && outBuf.frameLength == 0 { break }
        }
        return out
    }
}
