import Foundation
import MLX
import MLXFFT

/// Whisper log-mel spectrogram. Mirrors `openai-whisper`
/// `log_mel_spectrogram` (n_fft 400, hop 160, periodic Hann, slaney
/// librosa mel, log10 + the (clamp→+4)/4 normalization) and torch.stft
/// `center=True` reflect padding. Framing/reflect-pad run on CPU
/// (one 30 s clip is tiny); FFT + mel projection run on MLX. M4.2a.
///
/// Numerical parity with the reference is asserted only by sanity here;
/// exact correctness is confirmed end-to-end at M4.2c (transcription).
public enum LogMel {
    public static let sampleRate = 16_000
    public static let nFFT = 400
    public static let hop = 160
    public static let chunkSeconds = 30
    /// 30 s @ 16 kHz.
    public static let nSamples = 480_000
    /// `nSamples / hop`.
    public static let nFrames = 3_000

    // MARK: Slaney mel scale (librosa htk=False)

    private static let fSp = 200.0 / 3.0
    private static let minLogHz = 1_000.0
    private static let minLogMel = (1_000.0 - 0.0) / (200.0 / 3.0)  // 15
    private static let logStep = log(6.4) / 27.0

    private static func hzToMel(_ f: Double) -> Double {
        var mel = f / fSp
        if f >= minLogHz { mel = minLogMel + log(f / minLogHz) / logStep }
        return mel
    }
    private static func melToHz(_ m: Double) -> Double {
        var hz = fSp * m
        if m >= minLogMel { hz = minLogHz * exp(logStep * (m - minLogMel)) }
        return hz
    }

    /// librosa.filters.mel(sr=16000, n_fft=400, n_mels=`nMels`,
    /// fmin=0, fmax=8000, htk=False, norm='slaney') → `[nMels, 201]`
    /// row-major. Exposed for unit tests.
    public static func melFilterbank(nMels: Int = 128) -> [Float] {
        let nFreqs = nFFT / 2 + 1  // 201
        let fMax = Double(sampleRate) / 2.0  // 8000
        let fftFreqs = (0 ..< nFreqs).map {
            Double($0) * fMax / Double(nFreqs - 1)
        }
        let minMel = hzToMel(0), maxMel = hzToMel(fMax)
        let melPts = (0 ..< (nMels + 2)).map {
            melToHz(
                minMel + (maxMel - minMel) * Double($0) / Double(nMels + 1))
        }
        let fdiff = (0 ..< (nMels + 1)).map { melPts[$0 + 1] - melPts[$0] }

        var w = [Float](repeating: 0, count: nMels * nFreqs)
        for i in 0 ..< nMels {
            let enorm = 2.0 / (melPts[i + 2] - melPts[i])  // slaney
            for j in 0 ..< nFreqs {
                let lower = -(melPts[i] - fftFreqs[j]) / fdiff[i]
                let upper = (melPts[i + 2] - fftFreqs[j]) / fdiff[i + 1]
                let v = max(0.0, min(lower, upper))
                w[i * nFreqs + j] = Float(v * enorm)
            }
        }
        return w
    }

    // MARK: log-mel

    /// `samples` (mono 16 kHz) → log-mel `MLXArray` of shape
    /// `[nMels, nFrames]` (3000 frames; clip padded/trimmed to 30 s).
    public static func logMel(_ samples: [Float], nMels: Int = 128)
        -> MLXArray
    {
        // pad/trim to exactly 30 s
        var x = samples
        if x.count < nSamples {
            x.append(
                contentsOf: [Float](repeating: 0, count: nSamples - x.count))
        } else if x.count > nSamples {
            x = Array(x[0 ..< nSamples])
        }

        // torch.stft center=True: reflect-pad by nFFT/2 (no edge repeat)
        let p = nFFT / 2  // 200
        var padded = [Float](repeating: 0, count: x.count + 2 * p)
        for k in 0 ..< p { padded[k] = x[p - k] }  // x[p..1]
        for i in 0 ..< x.count { padded[p + i] = x[i] }
        for k in 0 ..< p { padded[p + x.count + k] = x[x.count - 2 - k] }

        // periodic Hann (torch.hann_window default periodic=True)
        let hann = (0 ..< nFFT).map {
            Float(0.5 * (1.0 - cos(2.0 * .pi * Double($0) / Double(nFFT))))
        }

        let totalFrames = 1 + (padded.count - nFFT) / hop  // 3001
        var frames = [Float](repeating: 0, count: totalFrames * nFFT)
        for t in 0 ..< totalFrames {
            let base = t * hop
            for k in 0 ..< nFFT {
                frames[t * nFFT + k] = padded[base + k] * hann[k]
            }
        }

        let frameArr = MLXArray(frames, [totalFrames, nFFT])
        // rfft → [totalFrames, 201] complex; drop the last frame so the
        // result is exactly nFrames (whisper's stft[..., :-1]).
        let spec = rfft(frameArr, n: nFFT, axis: -1)[0 ..< nFrames, 0...]
        let power = MLX.abs(spec).square()  // |X|^2  [nFrames, 201]

        let nFreqs = nFFT / 2 + 1
        let filters = MLXArray(
            melFilterbank(nMels: nMels), [nMels, nFreqs])
        // [nFrames,201] @ [201,nMels] → [nFrames,nMels] → [nMels,nFrames]
        var mel = MLX.matmul(power, filters.transposed()).transposed()

        mel = MLX.log10(MLX.maximum(mel, MLXArray(Float(1e-10))))
        mel = MLX.maximum(mel, mel.max() - 8.0)
        mel = (mel + 4.0) / 4.0
        mel.eval()
        // End-of-call allocator-pool flush (M50.1). The intermediate
        // `frameArr`, `spec`, `power`, `filters` arrays go out of
        // scope here; `mel` is still referenced so it isn't reclaimed.
        MLX.Memory.clearCache()
        return mel
    }
}
