import AVFoundation
import AthenaCore
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

    /// Floor on decoded samples: 0.1 s (1600 @ 16 kHz), matching OpenAI's
    /// Whisper-API minimum. A degenerate upload — an accidental record+delete,
    /// a truncated capture, near-silence — decodes to a handful of samples that
    /// is meaningless to transcribe/diarize/embed and can drive a model's conv
    /// frontend into an invalid shape (MLX errors abort the whole daemon
    /// process-wide). Enforcing the floor once HERE, at the shared decode
    /// chokepoint every audio route funnels through, turns that class into a
    /// uniform classified 400 instead of relying on each model's own input
    /// handling. `0` disables it (callers that decode a deliberately tiny clip).
    public static let defaultMinSamples = sampleRate / 10

    public enum DecodeError: Error, CustomStringConvertible {
        case open(String)
        case converterInit
        case convert(String)
        case tooLong(maxSamples: Int)
        case tooShort(samples: Int, minSamples: Int)
        public var description: String {
            switch self {
            case .open(let s): return "audio open failed: \(s)"
            case .converterInit: return "audio converter init failed"
            case .convert(let s): return "audio convert failed: \(s)"
            case .tooLong(let m):
                return "audio exceeds the \(m)-sample (~\(m / sampleRate)s) "
                    + "decode limit"
            case let .tooShort(n, m):
                return "audio is \(n) samples (~\(Double(n) / Double(sampleRate))"
                    + "s), below the \(m)-sample (~\(Double(m) / Double(sampleRate))"
                    + "s) minimum"
            }
        }

        /// Map a decode failure to a classified client (400) `AthenaError`,
        /// so a bad upload surfaces as `invalid_audio` / `audio_too_long` /
        /// `audio_format_unsupported` instead of the catch-all `500
        /// module_load_failed` (issue #6). The raw `description` (which can
        /// carry the temp file path) rides in the error's `serverDetail`
        /// (server log only, NE7), never the client body.
        public func athenaError(module: ModuleID) -> AthenaError {
            switch self {
            case .open, .convert:
                return .invalidAudio(module: module, detail: description)
            case .converterInit:
                return .audioFormatUnsupported(
                    module: module, detail: description)
            case .tooLong(let m):
                let secs = Double(m) / Double(sampleRate)
                return .audioTooLong(
                    module: module, seconds: secs, maxSeconds: secs)
            case let .tooShort(n, m):
                return .audioTooShort(
                    module: module,
                    seconds: Double(n) / Double(sampleRate),
                    minSeconds: Double(m) / Double(sampleRate))
            }
        }
    }

    /// Decode `url` like `pcm16kMono(from:maxSamples:)`, but translate any
    /// `DecodeError` into a classified 400 `AthenaError` tagged with `module`
    /// (issue #6). Callers on the serve path use this so the governed
    /// `classified(_:module:)` seam emits a cause-naming client error.
    public static func pcm16kMono(
        from url: URL, module: ModuleID, maxSamples: Int = defaultMaxSamples,
        minSamples: Int = defaultMinSamples
    ) throws -> [Float] {
        do {
            return try pcm16kMono(
                from: url, maxSamples: maxSamples, minSamples: minSamples)
        } catch let e as DecodeError {
            throw e.athenaError(module: module)
        }
    }

    /// Decode `url` → `[Float]` mono @ 16 kHz, range ~[-1, 1]. Decoding
    /// stops with `.tooLong` once `maxSamples` is exceeded; a result below
    /// `minSamples` is rejected with `.tooShort` (`0` ⇒ no floor).
    public static func pcm16kMono(
        from url: URL, maxSamples: Int = defaultMaxSamples,
        minSamples: Int = defaultMinSamples
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
        // Lower bound (symmetric with the `.tooLong` ceiling): reject a
        // degenerate too-short decode once, here, so every audio route gets a
        // uniform 400 instead of a model-specific deep failure / process abort.
        if minSamples > 0, out.count < minSamples {
            throw DecodeError.tooShort(
                samples: out.count, minSamples: minSamples)
        }
        return out
    }
}
