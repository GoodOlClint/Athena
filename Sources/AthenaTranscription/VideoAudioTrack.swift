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
        try await InMemoryAsset.readFirstAudioTrackPCM(
            asset: AVURLAsset(url: url), keepAlive: nil, module: module,
            maxSamples: maxSamples, minSamples: minSamples,
            onMissingTrack: .videoNoAudioTrack(module: module))
    }

    /// Demux the audio track from an in-memory video upload `Data` — the Option-D
    /// (ADR 025 S5) entry the daemon's `/v1/video/*` route uses, so the raw video
    /// bytes are never staged to disk. Same decode/floor/ceiling as the URL path.
    public static func extractPCM(
        from data: Data, filename: String?, module: ModuleID = .transcription,
        maxSamples: Int = AudioDecode.defaultMaxSamples,
        minSamples: Int = AudioDecode.defaultMinSamples
    ) async throws -> [Float] {
        let (asset, loader) = InMemoryAsset.make(data: data, filename: filename)
        return try await InMemoryAsset.readFirstAudioTrackPCM(
            asset: asset, keepAlive: loader, module: module,
            maxSamples: maxSamples, minSamples: minSamples,
            onMissingTrack: .videoNoAudioTrack(module: module))
    }
}
