import AVFoundation
import AthenaCore
import CoreMedia
import Foundation

/// Demux the audio track out of a video container (ADR 022, M78.1). Video
/// transcription is an audio-extraction problem, not a video problem: pull the
/// first audio track through `AVAssetReader` into the **same** 16 kHz mono
/// Float32 PCM the transcription tenant already consumes, then hand it to the
/// unchanged Whisper/Parakeet engine.
///
/// The extracted PCM funnels through the SAME floor + ceiling as `AudioDecode`
/// (`sampleBoundError`) — a degenerate video (no audio track, sub-0.1 s audio, a
/// decompression bomb) becomes a cause-naming 4xx, never a daemon abort,
/// carrying forward the M77.x audio crash-hardening. Pure AVFoundation; no MLX.
public enum VideoAudioTrack {
    /// Extract the first audio track of the asset at `url` as 16 kHz mono
    /// Float32 PCM. Throws a classified 400 (`AthenaError`) on any fault:
    /// unreadable container ⇒ `invalidAudio`; no audio track ⇒
    /// `videoNoAudioTrack`; below the floor ⇒ `audioTooShort`; over the ceiling
    /// ⇒ `audioTooLong`.
    public static func extractPCM(
        from url: URL, module: ModuleID = .transcription,
        maxSamples: Int = AudioDecode.defaultMaxSamples,
        minSamples: Int = AudioDecode.defaultMinSamples
    ) async throws -> [Float] {
        let asset = AVURLAsset(url: url)

        let tracks: [AVAssetTrack]
        do {
            tracks = try await asset.loadTracks(withMediaType: .audio)
        } catch {
            throw AthenaError.invalidAudio(
                module: module,
                detail: "video open failed: \(error.localizedDescription)")
        }
        guard let track = tracks.first else {
            throw AthenaError.videoNoAudioTrack(module: module)
        }

        let reader: AVAssetReader
        do {
            reader = try AVAssetReader(asset: asset)
        } catch {
            throw AthenaError.invalidAudio(
                module: module,
                detail: "video reader init failed: \(error.localizedDescription)")
        }

        // Decode straight to the canonical transcription PCM format so the
        // downstream engine path is identical to the audio route's.
        let settings: [String: Any] = [
            AVFormatIDKey: kAudioFormatLinearPCM,
            AVSampleRateKey: Double(AudioDecode.sampleRate),
            AVNumberOfChannelsKey: 1,
            AVLinearPCMBitDepthKey: 32,
            AVLinearPCMIsFloatKey: true,
            AVLinearPCMIsBigEndianKey: false,
            AVLinearPCMIsNonInterleaved: false,
        ]
        let output = AVAssetReaderTrackOutput(
            track: track, outputSettings: settings)
        output.alwaysCopiesSampleData = false
        guard reader.canAdd(output) else {
            throw AthenaError.invalidAudio(
                module: module, detail: "video reader cannot add audio output")
        }
        reader.add(output)
        guard reader.startReading() else {
            throw AthenaError.invalidAudio(
                module: module,
                detail: "video read start failed: "
                    + (reader.error?.localizedDescription ?? "unknown"))
        }

        var out: [Float] = []
        while let sample = output.copyNextSampleBuffer() {
            defer { CMSampleBufferInvalidate(sample) }
            guard let block = CMSampleBufferGetDataBuffer(sample) else { continue }
            let len = CMBlockBufferGetDataLength(block)
            guard len > 0 else { continue }
            let count = len / MemoryLayout<Float>.size
            var tmp = [Float](repeating: 0, count: count)
            let copied = tmp.withUnsafeMutableBytes { ptr -> OSStatus in
                guard let base = ptr.baseAddress else { return -1 }
                return CMBlockBufferCopyDataBytes(
                    block, atOffset: 0, dataLength: len, destination: base)
            }
            guard copied == kCMBlockBufferNoErr else {
                reader.cancelReading()
                throw AthenaError.invalidAudio(
                    module: module, detail: "video sample copy failed")
            }
            out.append(contentsOf: tmp)
            // Decompression-bomb ceiling: stop as soon as we pass it (D4).
            if out.count > maxSamples {
                reader.cancelReading()
                throw AudioDecode.DecodeError
                    .tooLong(maxSamples: maxSamples).athenaError(module: module)
            }
        }

        if reader.status == .failed {
            throw AthenaError.invalidAudio(
                module: module,
                detail: "video read failed: "
                    + (reader.error?.localizedDescription ?? "unknown"))
        }

        // Same floor + ceiling as the audio decode (shared verdict).
        if let e = AudioDecode.sampleBoundError(
            count: out.count, minSamples: minSamples, maxSamples: maxSamples)
        {
            throw e.athenaError(module: module)
        }
        return out
    }
}
