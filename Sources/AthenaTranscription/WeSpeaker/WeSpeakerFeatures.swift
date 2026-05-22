// Vendored from soniqo/speech-swift (Apache-2.0), adapted for Athena.
// 80-dim log-mel frontend for the WeSpeaker ResNet34-LM speaker-
// embedding network. This is the feature pipeline the WeSpeaker weights
// expect: nFFT=400 (25 ms) / hop=160 (10 ms) @ 16 kHz, Hamming window,
// HTK mel scale with Slaney area-normalization, pre-emphasis 0.97,
// per-utterance cepstral-mean normalization (CMN). M25.1.
import Accelerate
import Foundation
import MLX

/// Computes the 80-dim log-mel features WeSpeaker consumes. Pure
/// Accelerate/vDSP — no MLX FFT — mirroring the validated reference
/// pipeline so the embeddings discriminate speakers correctly.
final class WeSpeakerFeatures {
    let sampleRate = 16_000
    let nFFT = 400
    let hopLength = 160
    let nMels = 80
    let preEmphasis: Float = 0.97

    private let paddedFFT = 512
    private let log2PaddedFFT: vDSP_Length = 9
    private let fftSetup: FFTSetup
    private let window: [Float]
    private let melFilterbank: [Float]  // [nMels, nBins] row-major

    init() {
        // Hamming window — pyannote/wespeaker inference uses
        // window_type='hamming' (not Kaldi's default 'povey').
        var w = [Float](repeating: 0, count: nFFT)
        for i in 0..<nFFT {
            w[i] = 0.54 - 0.46 * cos(2.0 * Float.pi * Float(i) / Float(nFFT - 1))
        }
        self.window = w

        guard let setup = vDSP_create_fftsetup(9, FFTRadix(kFFTRadix2)) else {
            fatalError("WeSpeakerFeatures: vDSP FFT setup failed")
        }
        self.fftSetup = setup
        self.melFilterbank = Self.buildMelFilterbank(
            nMels: nMels, paddedFFT: paddedFFT, sampleRate: sampleRate)
    }

    deinit { vDSP_destroy_fftsetup(fftSetup) }

    private static func buildMelFilterbank(
        nMels: Int, paddedFFT: Int, sampleRate: Int
    ) -> [Float] {
        let fMin: Float = 20.0
        let fMax = Float(sampleRate) / 2.0
        let nBins = paddedFFT / 2 + 1

        // HTK mel scale (Kaldi default): mel = 2595·log10(1 + hz/700).
        func hzToMel(_ hz: Float) -> Float { 2595.0 * log10(1.0 + hz / 700.0) }
        func melToHz(_ mel: Float) -> Float { 700.0 * (pow(10.0, mel / 2595.0) - 1.0) }

        var fftFreqs = [Float](repeating: 0, count: nBins)
        for i in 0..<nBins {
            fftFreqs[i] = Float(i) * Float(sampleRate) / Float(paddedFFT)
        }

        let melMin = hzToMel(fMin)
        let melMax = hzToMel(fMax)
        let nMelPoints = nMels + 2
        var melPoints = [Float](repeating: 0, count: nMelPoints)
        for i in 0..<nMelPoints {
            melPoints[i] = melMin + Float(i) * (melMax - melMin) / Float(nMelPoints - 1)
        }
        let filterFreqs = melPoints.map(melToHz)
        var filterDiff = [Float](repeating: 0, count: nMelPoints - 1)
        for i in 0..<(nMelPoints - 1) {
            filterDiff[i] = filterFreqs[i + 1] - filterFreqs[i]
        }

        // [nMels, nBins] row-major triangular filters with Slaney
        // area-normalization (matches the reference frontend).
        var fb = [Float](repeating: 0, count: nMels * nBins)
        for m in 0..<nMels {
            let lo = filterFreqs[m]
            let hi = filterFreqs[m + 2]
            let enorm = 2.0 / (filterFreqs[m + 2] - filterFreqs[m])
            for k in 0..<nBins {
                let freq = fftFreqs[k]
                let down = (freq - lo) / filterDiff[m]
                let up = (hi - freq) / filterDiff[m + 1]
                fb[m * nBins + k] = max(0.0, min(down, up)) * enorm
            }
        }
        return fb
    }

    /// Extract `[nFrames * 80]` row-major log-mel features (CMN applied).
    func extractRaw(_ audio: [Float]) -> (melSpec: [Float], nFrames: Int) {
        let nBins = paddedFFT / 2 + 1
        let halfPadded = paddedFFT / 2

        // Pre-emphasis y[n] = x[n] - 0.97·x[n-1].
        var emphasized = [Float](repeating: 0, count: audio.count)
        if !audio.isEmpty {
            emphasized[0] = audio[0]
            for i in 1..<audio.count {
                emphasized[i] = audio[i] - preEmphasis * audio[i - 1]
            }
        }

        // Reflect-pad by nFFT/2 each side (librosa center=True).
        let padLength = nFFT / 2
        let srcCount = emphasized.count
        var paddedAudio = [Float](
            repeating: 0, count: padLength + srcCount + padLength)
        for i in 0..<padLength {
            let idx = min(padLength - i, max(srcCount - 1, 0))
            paddedAudio[i] = srcCount > 0 ? emphasized[max(0, idx)] : 0
        }
        for i in 0..<srcCount { paddedAudio[padLength + i] = emphasized[i] }
        for i in 0..<padLength {
            let idx = srcCount - 2 - i
            paddedAudio[padLength + srcCount + i] =
                srcCount > 0 ? emphasized[max(0, idx)] : 0
        }

        let nFrames = max(0, (paddedAudio.count - nFFT) / hopLength + 1)
        guard nFrames > 0 else { return ([], 0) }

        var splitReal = [Float](repeating: 0, count: halfPadded)
        var splitImag = [Float](repeating: 0, count: halfPadded)
        var frameBuf = [Float](repeating: 0, count: paddedFFT)
        var magnitude = [Float](repeating: 0, count: nFrames * nBins)

        for frame in 0..<nFrames {
            let start = frame * hopLength
            paddedAudio.withUnsafeBufferPointer { buf in
                vDSP_vmul(
                    buf.baseAddress! + start, 1, window, 1,
                    &frameBuf, 1, vDSP_Length(nFFT))
            }
            for i in nFFT..<paddedFFT { frameBuf[i] = 0 }

            for i in 0..<halfPadded {
                splitReal[i] = frameBuf[2 * i]
                splitImag[i] = frameBuf[2 * i + 1]
            }
            splitReal.withUnsafeMutableBufferPointer { rb in
                splitImag.withUnsafeMutableBufferPointer { ib in
                    var split = DSPSplitComplex(
                        realp: rb.baseAddress!, imagp: ib.baseAddress!)
                    vDSP_fft_zrip(
                        fftSetup, &split, 1, log2PaddedFFT,
                        FFTDirection(kFFTDirection_Forward))
                }
            }

            // vDSP packs DC in real[0], Nyquist in imag[0] (×2 scaling
            // cancels in CMN + L2-normalized cosine, so it is left as-is
            // to match the reference power spectrum).
            let base = frame * nBins
            magnitude[base] = splitReal[0] * splitReal[0]
            magnitude[base + halfPadded] = splitImag[0] * splitImag[0]
            for k in 1..<halfPadded {
                magnitude[base + k] =
                    splitReal[k] * splitReal[k] + splitImag[k] * splitImag[k]
            }
        }

        // Mel projection: magnitude[nFrames, nBins] · fbᵀ[nBins, nMels].
        // melFilterbank is [nMels, nBins] row-major; transpose to
        // fbT [nBins, nMels] (mtrans: M,N = rows,cols of the OUTPUT).
        var melSpec = [Float](repeating: 0, count: nFrames * nMels)
        var fbT = [Float](repeating: 0, count: nBins * nMels)
        vDSP_mtrans(
            melFilterbank, 1, &fbT, 1, vDSP_Length(nBins), vDSP_Length(nMels))
        vDSP_mmul(
            magnitude, 1, fbT, 1, &melSpec, 1,
            vDSP_Length(nFrames), vDSP_Length(nMels), vDSP_Length(nBins))

        // log(max(x, 1e-10)).
        var lowClip: Float = 1e-10
        var hiClip = Float.greatestFiniteMagnitude
        var count = Int32(melSpec.count)
        vDSP_vclip(
            melSpec, 1, &lowClip, &hiClip, &melSpec, 1, vDSP_Length(melSpec.count))
        vvlogf(&melSpec, melSpec, &count)

        // CMN: subtract per-bin temporal mean (WeSpeaker: feat -= mean over time).
        var binMeans = [Float](repeating: 0, count: nMels)
        for frame in 0..<nFrames {
            let b = frame * nMels
            for m in 0..<nMels { binMeans[m] += melSpec[b + m] }
        }
        var inv = 1.0 / Float(nFrames)
        vDSP_vsmul(binMeans, 1, &inv, &binMeans, 1, vDSP_Length(nMels))
        melSpec.withUnsafeMutableBufferPointer { ms in
            binMeans.withUnsafeBufferPointer { bm in
                for frame in 0..<nFrames {
                    vDSP_vsub(
                        bm.baseAddress!, 1,
                        ms.baseAddress! + frame * nMels, 1,
                        ms.baseAddress! + frame * nMels, 1,
                        vDSP_Length(nMels))
                }
            }
        }

        return (melSpec, nFrames)
    }

    /// `[T, 80]` time-major log-mel features as an MLXArray (T may be 0
    /// for empty input; callers guard that).
    func extract(_ audio: [Float]) -> MLXArray {
        let (mel, nFrames) = extractRaw(audio)
        return MLXArray(mel, [nFrames, nMels])
    }
}
