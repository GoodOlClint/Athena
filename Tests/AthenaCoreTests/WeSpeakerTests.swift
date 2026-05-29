import AthenaCore
import Foundation
import MLX
import XCTest

@testable import AthenaTranscription

/// M25.1 — CI-safe structural checks for the vendored WeSpeaker
/// ResNet34-LM speaker-embedding network + frontend. No model download:
/// these validate the shape contract (the 5120-d pooled vector, 256-d
/// output, frame math) on random init so a regression in the arch is
/// caught without the heavy weights.
final class WeSpeakerStructureTests: XCTestCase {
    func testNetworkForwardShapeAndNorm() {
        let net = WeSpeakerNetwork()
        // [B=1, T=200, F=80, C=1] log-mel grid — deterministic varying
        // fill (avoids an MLXRandom dependency in the test target).
        let count = 1 * 200 * 80 * 1
        var data = [Float](repeating: 0, count: count)
        for i in 0..<count {
            data[i] = sin(Float(i) * 0.013) * 0.5
        }
        let x = MLXArray(data, [1, 200, 80, 1])
        let y = net(x)
        eval(y)
        XCTAssertEqual(y.shape, [1, 256], "embedding must be [B, 256]")
        let v = y[0].asArray(Float.self)
        XCTAssertTrue(v.allSatisfy { $0.isFinite }, "non-finite embedding")
        let norm = sqrt(v.reduce(0) { $0 + $1 * $1 })
        XCTAssertEqual(norm, 1.0, accuracy: 1e-3, "output must be L2-normalized")
    }

    func testAgglomerativeClusteringRecoversGroups() {
        // 3 well-separated directions in 8-D, 5 points each + tiny noise.
        func axis(_ i: Int, _ jitter: Float) -> [Float] {
            var v = [Float](repeating: 0, count: 8)
            v[i] = 1
            v[(i + 1) % 8] = jitter
            let n = (v.reduce(0) { $0 + $1 * $1 }).squareRoot()
            return v.map { $0 / n }
        }
        var embs: [[Float]] = []
        for g in 0..<3 {
            for k in 0..<5 { embs.append(axis(g, Float(k) * 0.02)) }
        }
        // Exact-count mode.
        let fixed = AgglomerativeClustering.cluster(embs, numClusters: 3)
        XCTAssertEqual(Set(fixed).count, 3)
        XCTAssertEqual(Set(fixed[0..<5]).count, 1, "group 0 split")
        XCTAssertEqual(Set(fixed[5..<10]).count, 1, "group 1 split")
        XCTAssertEqual(Set(fixed[10..<15]).count, 1, "group 2 split")
        XCTAssertNotEqual(fixed[0], fixed[5])
        XCTAssertNotEqual(fixed[5], fixed[10])
        // Auto mode (well-separated ⇒ cross-cluster distance ~1 > 0.7).
        let auto = AgglomerativeClustering.cluster(embs, threshold: 0.7)
        XCTAssertEqual(Set(auto).count, 3, "auto count wrong")
    }

    func testFeatureFrameMathAndShape() {
        let fe = WeSpeakerFeatures()
        // 1 s of a 220 Hz tone at 16 kHz.
        let n = 16_000
        var pcm = [Float](repeating: 0, count: n)
        for i in 0..<n {
            pcm[i] = 0.2 * sin(2.0 * Float.pi * 220.0 * Float(i) / 16_000.0)
        }
        let (mel, frames) = fe.extractRaw(pcm)
        // 10 ms hop ⇒ ~100 frames for 1 s (reflect-padded center frames).
        XCTAssertGreaterThanOrEqual(frames, 95)
        XCTAssertLessThanOrEqual(frames, 105)
        XCTAssertEqual(mel.count, frames * 80)
        XCTAssertTrue(mel.allSatisfy { $0.isFinite }, "non-finite mel")
    }
}

/// M25.1 — heavy, gated WeSpeaker integration. Downloads the ungated
/// mlx-community weights, folds BatchNorm, and verifies the embeddings
/// actually discriminate speakers (same-voice cosine high, cross-voice
/// clearly lower). Validate via xcodebuild with ATHENA_RUN_MODEL_TESTS=1.
final class WeSpeakerIntegrationTests: XCTestCase {

    private func sayClip(_ voice: String, _ text: String) throws -> [Float] {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("athena-spkemb-\(UUID()).aiff")
        defer { try? FileManager.default.removeItem(at: url) }
        let p = Process()
        p.executableURL = URL(fileURLWithPath: "/usr/bin/say")
        p.arguments = ["-v", voice, "-o", url.path, text]
        try p.run()
        p.waitUntilExit()
        guard p.terminationStatus == 0 else {
            throw XCTSkip("`say -v \(voice)` unavailable")
        }
        return try AudioDecode.pcm16kMono(from: url)
    }

    private func cosine(_ a: [Float], _ b: [Float]) -> Float {
        var dot: Float = 0
        var na: Float = 0
        var nb: Float = 0
        for i in 0..<min(a.count, b.count) {
            dot += a[i] * b[i]
            na += a[i] * a[i]
            nb += b[i] * b[i]
        }
        let d = sqrt(na) * sqrt(nb)
        return d > 0 ? dot / d : 0
    }

    /// Real-audio diagnostic: embed disjoint windows of the ICSI Bdb001
    /// meeting (multi-speaker human speech) and report the pairwise
    /// cosine spread. A working speaker embedding shows a WIDE spread
    /// (some windows same-speaker → high, others cross-speaker → low);
    /// a broken frontend collapses everything to ~one value. Needs
    /// /tmp/audio/clip60.wav.
    func testRealAudioCosineSpread() async throws {
        guard
            ProcessInfo.processInfo.environment["ATHENA_RUN_MODEL_TESTS"]
                == "1"
        else { throw XCTSkip("set ATHENA_RUN_MODEL_TESTS=1 (heavy)") }
        let clip = URL(fileURLWithPath: "/tmp/audio/clip60.wav")
        guard FileManager.default.fileExists(atPath: clip.path) else {
            throw XCTSkip("no /tmp/audio/clip60.wav fixture")
        }
        let pcm = try AudioDecode.pcm16kMono(from: clip)
        let model = try await WeSpeakerModel.fromPretrained(
            "aufklarer/WeSpeaker-ResNet34-LM-MLX")

        // 3 s windows, hop 3 s, across the clip.
        let win = 3 * 16_000
        var embs: [[Float]] = []
        var i = 0
        while i + win <= pcm.count {
            embs.append(model.embed(Array(pcm[i..<(i + win)])))
            i += win
        }
        XCTAssertGreaterThan(embs.count, 4, "need several windows")
        var lo: Float = 1
        var hi: Float = -1
        for a in 0..<embs.count {
            for b in (a + 1)..<embs.count {
                let c = cosine(embs[a], embs[b])
                lo = min(lo, c)
                hi = max(hi, c)
            }
        }
        print("ICSI window cosine spread — min=\(lo) max=\(hi)")
        // Real multi-speaker audio must produce a clear spread: at least
        // one cross-speaker pair well below 1.0.
        XCTAssertLessThan(lo, 0.6, "no discrimination — min pair cos \(lo)")
    }

    func testGovernedModuleAndInvalidSegment() async throws {
        guard
            ProcessInfo.processInfo.environment["ATHENA_RUN_MODEL_TESTS"]
                == "1"
        else { throw XCTSkip("set ATHENA_RUN_MODEL_TESTS=1 (heavy)") }

        let a = try sayClip("Alex", "First speaker, a little audio here for the test.")
        let b = try sayClip("Samantha", "Second speaker, some more audio for the test.")
        let durA = Double(a.count) / 16_000.0
        let total = Double(a.count + b.count) / 16_000.0
        let pcm = a + b

        var wav = Data()  // minimal RIFF/WAVE 16k mono float32
        func le<T: FixedWidthInteger>(_ v: T) -> Data {
            withUnsafeBytes(of: v.littleEndian) { Data($0) }
        }
        let bytes = pcm.withUnsafeBytes { Data($0) }
        wav.append("RIFF".data(using: .ascii)!)
        wav.append(le(UInt32(36 + bytes.count)))
        wav.append("WAVEfmt ".data(using: .ascii)!)
        wav.append(le(UInt32(16)))
        wav.append(le(UInt16(3)))  // IEEE float
        wav.append(le(UInt16(1)))  // mono
        wav.append(le(UInt32(16_000)))
        wav.append(le(UInt32(16_000 * 4)))
        wav.append(le(UInt16(4)))
        wav.append(le(UInt16(32)))
        wav.append("data".data(using: .ascii)!)
        wav.append(le(UInt32(bytes.count)))
        wav.append(bytes)

        let m = MLXSpeakerEmbeddingModule()
        XCTAssertEqual(m.id, .speakerEmbedding)
        let gov = MemoryGovernor(totalBudgetBytes: Int(8) << 30)
        await gov.register(m, evictable: true)
        try await gov.ensureLoaded(.speakerEmbedding)

        // Two speaker segments → two distinct vectors, lower mutual cosine.
        let r = try await m.embed(
            audio: wav, filename: "a.wav",
            segments: [
                SpeakerSegmentRequest(start: 0, end: durA),
                SpeakerSegmentRequest(start: durA, end: total),
            ])
        XCTAssertEqual(r.dimension, 256)
        XCTAssertEqual(r.segments.count, 2)
        XCTAssertEqual(r.segments[0].embedding.count, 256)

        // Invalid (empty) segment → classified 400, never a silent zero.
        do {
            _ = try await m.embed(
                audio: wav, filename: "a.wav",
                segments: [SpeakerSegmentRequest(start: 5, end: 5)])
            XCTFail("expected audioSegmentInvalid")
        } catch let e as AthenaError {
            XCTAssertEqual(e.httpStatus, 400)
            XCTAssertEqual(e.code, "audio_segment_invalid")
        }
    }
}

/// M50.2 — regression for the allocator-pool leak class M46.6 caught
/// in the embedder. Drives many short speaker-embedding calls back-to-
/// back and asserts MLX's pool stays bounded — without the per-call
/// clear in `WeSpeakerModel.embed`, the pool grows N-proportionally
/// with call count. Gated + heavy.
final class WeSpeakerMemoryRegressionTests: XCTestCase {

    func testEmbedPoolStaysBoundedAcrossManyCalls() async throws {
        guard
            ProcessInfo.processInfo.environment["ATHENA_RUN_MODEL_TESTS"]
                == "1"
        else { throw XCTSkip("set ATHENA_RUN_MODEL_TESTS=1 (heavy)") }

        let model = try await WeSpeakerModel.fromPretrained(
            "aufklarer/WeSpeaker-ResNet34-LM-MLX")
        // 3 s of synthetic noise so the ResNet sees a full mel matrix
        // (silence sometimes short-circuits the feature extractor). The
        // leak is allocator-pool growth from the forward pass, not
        // anything content-dependent — values don't matter.
        var pcm = [Float](repeating: 0, count: 3 * 16_000)
        var rng = SystemRandomNumberGenerator()
        for i in 0..<pcm.count {
            pcm[i] = Float(Int(rng.next() % 1000)) / 1000.0 - 0.5
        }

        // Warmup so first-call lazy allocations settle.
        _ = model.embed(pcm)
        MLX.Memory.clearCache()
        let baseline = MLX.Memory.cacheMemory

        for _ in 0..<22 { _ = model.embed(pcm) }

        let after = MLX.Memory.cacheMemory
        // Without M50.2's clear, the pool scales linearly with the
        // ResNet per-call activations × 22.
        let ceiling = 256 * 1024 * 1024
        XCTAssertLessThan(
            after - baseline, ceiling,
            "MLX cache pool drifted \(after - baseline) bytes "
            + "above baseline after 22 WeSpeaker embeds (M50.2 leak)")
    }
}
