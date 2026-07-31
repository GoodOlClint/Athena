import AVFoundation
import AthenaCore
import CoreMedia
import Foundation
import UniformTypeIdentifiers

/// Feed Apple's AV decoders from an in-memory upload `Data` with **no file on
/// disk** (ADR 025 S5 / Option D). An `AVURLAsset` over a custom `athena-mem://`
/// scheme is backed by an `AVAssetResourceLoaderDelegate` that serves byte
/// ranges straight from the upload buffer, so `AVAssetReader` demuxes audio (and
/// a video container's audio track) exactly as it would from a file — the same
/// system codecs, hardware decode retained — but the daemon never writes the raw
/// request bytes to `NSTemporaryDirectory()`. This removes the upload
/// crash-residue / forensic surface ADR 025 flagged.
///
/// Honesty boundary (ADR 024): the upload `Data` and the decoded PCM still live
/// in process RAM during the request — that is the live co-resident read threat
/// owned by ADR 024 Tier 1, not closed here. S5 removes disk residue, not the
/// in-memory working set. Pure AVFoundation; no MLX.
///
/// Phase-0 Spike A validated this path against the daemon's prior file-based
/// decode across wav/mp3/m4a-aac/flac/ogg-opus/mp4/mov with RMS parity.

/// Serves byte ranges of an in-memory `Data` to `AVAssetReader` via the resource
/// loader. One instance per asset; the consumer must keep it alive for the
/// asset's whole read (the resource loader holds the delegate weakly).
public final class InMemoryResourceLoader: NSObject, AVAssetResourceLoaderDelegate {
    private let data: Data
    private let contentType: String?  // UTI, e.g. "public.mp3"; nil ⇒ let AV sniff

    public init(data: Data, contentType: String?) {
        self.data = data
        self.contentType = contentType
    }

    public func resourceLoader(
        _ resourceLoader: AVAssetResourceLoader,
        shouldWaitForLoadingOfRequestedResource req: AVAssetResourceLoadingRequest
    ) -> Bool {
        if let info = req.contentInformationRequest {
            if let contentType { info.contentType = contentType }
            info.contentLength = Int64(data.count)
            info.isByteRangeAccessSupported = true
        }
        if let dr = req.dataRequest {
            let start = Int(dr.requestedOffset)
            if start < 0 || start > data.count {
                req.finishLoading(
                    with: NSError(
                        domain: "athena-mem", code: 1,
                        userInfo: [NSLocalizedDescriptionKey: "range out of bounds"]))
                return true
            }
            let len: Int
            if dr.requestsAllDataToEndOfResource {
                len = data.count - start
            } else {
                len = min(dr.requestedLength, data.count - start)
            }
            if len > 0 {
                dr.respond(with: data.subdata(in: start ..< (start + len)))
            }
        }
        req.finishLoading()
        return true
    }
}

public enum InMemoryAsset {
    static let scheme = "athena-mem"

    /// Sniff the container from the upload's magic bytes → a filename extension.
    /// A resource-loader-backed `AVURLAsset` does NOT byte-sniff (unlike
    /// `AVAudioFile(forReading:)`, which the pre-S5 temp-file path relied on), so
    /// when the upload has no usable filename hint we recover the content type
    /// here to preserve decode parity for unnamed uploads.
    static func sniffExtension(_ d: Data) -> String? {
        func tag(_ s: String, at off: Int) -> Bool {
            let bytes = Array(s.utf8)
            guard d.count >= off + bytes.count else { return false }
            let base = d.startIndex
            for (i, b) in bytes.enumerated() where d[base + off + i] != b {
                return false
            }
            return true
        }
        if tag("RIFF", at: 0), tag("WAVE", at: 8) { return "wav" }
        if tag("FORM", at: 0), tag("AIFF", at: 8) || tag("AIFC", at: 8) { return "aiff" }
        if tag("caff", at: 0) { return "caf" }
        if tag("fLaC", at: 0) { return "flac" }
        if tag("OggS", at: 0) { return "ogg" }
        if tag("ID3", at: 0) { return "mp3" }
        if tag("ftyp", at: 4) { return tag("M4A", at: 8) ? "m4a" : "mp4" }
        // MP3 frame sync (0xFFEx/0xFFFx) for headerless MP3.
        if d.count >= 2 {
            let b0 = d[d.startIndex], b1 = d[d.index(after: d.startIndex)]
            if b0 == 0xFF, (b1 & 0xE0) == 0xE0 { return "mp3" }
        }
        return nil
    }

    /// Build an `AVURLAsset` backed by `data` over the `athena-mem://` scheme.
    /// Returns the asset and its resource loader — the caller MUST keep the
    /// loader alive for the asset's entire read (pass it as `keepAlive:` to
    /// `readFirstAudioTrackPCM`).
    static func make(
        data: Data, filename: String?
    ) -> (asset: AVURLAsset, loader: InMemoryResourceLoader) {
        // Resolve a content type: filename hint first, else sniff the magic
        // bytes (the resource-loader asset won't sniff on its own). The path
        // extension is advisory; contentType drives detection.
        let nameExt = ((filename as NSString?)?.pathExtension)
            .flatMap { $0.isEmpty ? nil : $0 }
        // The filename extension is attacker-controlled; a URL-illegal byte in
        // it (space, `|`, control char) would make `URL(string:)` return nil
        // and a force-unwrap would abort the daemon. Use the filename ext only
        // when it's a safe (alphanumeric) token; otherwise recover a safe
        // extension by sniffing the magic bytes — so a valid upload with a
        // hostile filename still decodes (AVFoundation uses the URL path
        // extension as a primary type hint, not just contentType). The final
        // `?? UUID-only` keeps the URL build total even if all hints fail.
        func safe(_ e: String?) -> String? {
            guard let e, !e.isEmpty,
                e.unicodeScalars.allSatisfy({ CharacterSet.alphanumerics.contains($0) })
            else { return nil }
            return e
        }
        let ext = safe(nameExt) ?? safe(sniffExtension(data)) ?? ""
        let suffix = ext.isEmpty ? "" : ".\(ext)"
        let url =
            URL(string: "\(scheme)://\(UUID().uuidString)\(suffix)")
            ?? URL(string: "\(scheme)://\(UUID().uuidString)")!
        let asset = AVURLAsset(url: url)
        // Derive the contentType from the SAME resolved-safe `ext`, not the raw
        // filename: `UTType(filenameExtension:)` on a hostile extension (e.g.
        // "wav evil") returns a bogus `dyn.*` dynamic type — not nil — which
        // would poison detection and make AVFoundation fail to open even valid
        // audio. `ext` is already either a safe filename ext or a sniffed known
        // format token.
        let contentType =
            ext.isEmpty
            ? nil : UTType(filenameExtension: ext)?.identifier
        let loader = InMemoryResourceLoader(data: data, contentType: contentType)
        let queue = DispatchQueue(label: "athena.inmem-asset.\(UUID().uuidString)")
        asset.resourceLoader.setDelegate(loader, queue: queue)
        return (asset, loader)
    }

    /// Decode the first audio track of `asset` to 16 kHz mono Float32 PCM,
    /// enforcing the shared floor/ceiling (`AudioDecode.sampleBoundError`). This
    /// is the ONE reader core both the audio chokepoint (`AudioDecode`) and the
    /// video audio-track extractor (`VideoAudioTrack`) run, over either a file
    /// URL or an in-memory `athena-mem://` asset.
    ///
    /// - keepAlive: retained for the whole call so an in-memory asset's resource
    ///   loader survives every `await` (pass `nil` for file-backed assets).
    /// - onMissingTrack: the classified error to throw when the container has no
    ///   audio track (audio routes ⇒ `invalidAudio`; video ⇒ `videoNoAudioTrack`).
    static func readFirstAudioTrackPCM(
        asset: AVURLAsset, keepAlive: AnyObject?, module: ModuleID,
        maxSamples: Int, minSamples: Int, onMissingTrack: AthenaError
    ) async throws -> [Float] {
        // `keepAlive` is a strong parameter binding, so the resource loader
        // stays alive for the whole body (including the awaits below).
        let tracks: [AVAssetTrack]
        do {
            tracks = try await asset.loadTracks(withMediaType: .audio)
        } catch {
            throw AthenaError.invalidAudio(
                module: module,
                detail: "audio open failed: \(error.localizedDescription)")
        }
        guard let track = tracks.first else { throw onMissingTrack }

        let reader: AVAssetReader
        do {
            reader = try AVAssetReader(asset: asset)
        } catch {
            throw AthenaError.invalidAudio(
                module: module,
                detail: "reader init failed: \(error.localizedDescription)")
        }

        // Decode straight to the canonical transcription PCM format.
        let settings: [String: Any] = [
            AVFormatIDKey: kAudioFormatLinearPCM,
            AVSampleRateKey: Double(AudioDecode.sampleRate),
            AVNumberOfChannelsKey: 1,
            AVLinearPCMBitDepthKey: 32,
            AVLinearPCMIsFloatKey: true,
            AVLinearPCMIsBigEndianKey: false,
            AVLinearPCMIsNonInterleaved: false,
        ]
        let output = AVAssetReaderTrackOutput(track: track, outputSettings: settings)
        output.alwaysCopiesSampleData = false
        guard reader.canAdd(output) else {
            throw AthenaError.audioFormatUnsupported(
                module: module, detail: "cannot decode audio track to PCM")
        }
        reader.add(output)
        guard reader.startReading() else {
            throw AthenaError.invalidAudio(
                module: module,
                detail: "read start failed: "
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
                    module: module, detail: "sample copy failed")
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
                detail: "read failed: "
                    + (reader.error?.localizedDescription ?? "unknown"))
        }

        // Shared floor + ceiling verdict (unit-pinned, ADR 008/009).
        if let e = AudioDecode.sampleBoundError(
            count: out.count, minSamples: minSamples, maxSamples: maxSamples)
        {
            throw e.athenaError(module: module)
        }
        // Touch keepAlive at the very end so the optimizer cannot release the
        // resource loader before the reader has drained the asset.
        withExtendedLifetime(keepAlive) {}
        return out
    }

    /// Startup sweep of any legacy `athena-*` upload temp files left in
    /// `NSTemporaryDirectory()` by the pre-S5 file-staging decode path (a crash
    /// could orphan one). A no-op once Option D is the only path; migration
    /// insurance. Best-effort — failures are ignored.
    public static func sweepLegacyUploadTempFiles() {
        let dir = FileManager.default.temporaryDirectory
        guard
            let entries = try? FileManager.default.contentsOfDirectory(
                at: dir, includingPropertiesForKeys: nil)
        else { return }
        for url in entries where url.lastPathComponent.hasPrefix("athena-") {
            try? FileManager.default.removeItem(at: url)
        }
    }
}
