import AthenaCore
import Foundation
import XCTest

@testable import AthenaTranscription

/// Robustness sweep: run a corpus of tiny / accidental / corrupt audio files
/// through EVERY audio pipeline (Whisper + Parakeet transcription, Sortformer +
/// pyannote diarization, WeSpeaker speaker-embedding) and assert none aborts the
/// process. MLX errors abort process-wide (EXC_BREAKPOINT), so a degenerate
/// input that slips past the input guards crashes the whole daemon — this sweep
/// catches that class against real files, end to end through the module actors
/// (which include the AVFoundation decode + every conv path the daemon runs).
///
/// Gated on `ATHENA_RUN_MODEL_TESTS=1` (loads real models + runs MLX). Corpus
/// dir from `ATHENA_QUARANTINE_DIR` (default the field-reported quarantine);
/// skipped when absent, so CI is unaffected. A *clean* thrown error (a
/// classified 4xx — too-short/undecodable audio) is a PASS: the daemon stays
/// up. Only a process abort fails. A marker is flushed before each call so, if
/// a file ever does abort, the last line names the (pipeline, file) culprit.
final class QuarantineAudioSweepTests: XCTestCase {
    private static let storeRoot = URL(
        fileURLWithPath: NSHomeDirectory() + "/.athena/models", isDirectory: true)

    private func gate() throws -> [URL] {
        guard ProcessInfo.processInfo.environment["ATHENA_RUN_MODEL_TESTS"] == "1"
        else { throw XCTSkip("set ATHENA_RUN_MODEL_TESTS=1 (heavy)") }
        guard
            let dir = ProcessInfo.processInfo.environment["ATHENA_QUARANTINE_DIR"]
        else {
            throw XCTSkip(
                "set ATHENA_QUARANTINE_DIR to a folder of degenerate audio")
        }
        let url = URL(fileURLWithPath: dir, isDirectory: true)
        let files =
            ((try? FileManager.default.contentsOfDirectory(
                at: url, includingPropertiesForKeys: nil)) ?? [])
            .filter {
                ["m4a", "wav", "mp3", "flac", "aac"].contains($0.pathExtension.lowercased())
            }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        guard !files.isEmpty else {
            throw XCTSkip("no audio files under \(dir)")
        }
        return files
    }

    private func mark(_ s: String) {
        FileHandle.standardError.write(Data(("[sweep] " + s + "\n").utf8))
    }

    /// Run `body` and classify the outcome. A thrown error is a clean,
    /// daemon-surviving rejection (PASS). The only failure mode is a process
    /// abort, which this function cannot catch — the flushed marker before the
    /// call is what survives to name the culprit.
    private func probe(_ label: String, _ body: () async throws -> String)
        async
    {
        mark("→ \(label)")
        do {
            let summary = try await body()
            mark("  ok: \(label) — \(summary)")
        } catch {
            mark("  handled (clean error): \(label) — \(error)")
        }
    }

    func testQuarantineCorpusCrashesNoPipeline() async throws {
        let files = try gate()
        mark("corpus: \(files.count) files; store=\(Self.storeRoot.path)")

        func bytes(_ u: URL) -> Data { (try? Data(contentsOf: u)) ?? Data() }

        // --- Transcription: Whisper, then Parakeet (shared slot, rebind). ---
        let t = MLXTranscriptionModule(modelStoreRoot: Self.storeRoot)
        for model in ["whisper-large-v3-turbo", "parakeet-tdt-0.6b-v3"] {
            do { try await t.rebind(to: model) } catch {
                mark("SKIP transcription/\(model): load failed — \(error)")
                continue
            }
            for f in files {
                await probe("transcribe[\(model)] | \(f.lastPathComponent)") {
                    let r = try await t.transcribe(
                        audio: bytes(f), filename: f.lastPathComponent,
                        language: nil, wordTimestamps: true)
                    return "dur=\(r.duration) segs=\(r.segments.count)"
                }
            }
        }

        // --- Diarization: Sortformer (diarize), then pyannote (segment). ---
        let d = MLXDiarizationModule(modelStoreRoot: Self.storeRoot)
        if (try? await d.rebind(to: "diar_streaming_sortformer_4spk-v2.1-fp16"))
            != nil
        {
            for f in files {
                await probe("diarize[sortformer] | \(f.lastPathComponent)") {
                    let r = try await d.diarize(
                        audio: bytes(f), filename: f.lastPathComponent)
                    return "turns=\(r.turns.count) spk=\(r.numSpeakers)"
                }
            }
        } else {
            mark("SKIP diarize/sortformer: load failed")
        }

        if (try? await d.rebind(to: "Pyannote-Segmentation-MLX")) != nil {
            for f in files {
                await probe("segment[pyannote] | \(f.lastPathComponent)") {
                    let r = try await d.segment(
                        audio: bytes(f), filename: f.lastPathComponent)
                    return "regions=\(r.count)"
                }
            }
        } else {
            mark("SKIP segment/pyannote: load failed")
        }

        // --- Speaker embedding: WeSpeaker (whole-clip + sliding window). ---
        let s = MLXSpeakerEmbeddingModule(modelStoreRoot: Self.storeRoot)
        if (try? await s.rebind(to: "WeSpeaker-ResNet34-LM-MLX")) != nil {
            for f in files {
                await probe("spk.embed | \(f.lastPathComponent)") {
                    let r = try await s.embed(
                        audio: bytes(f), filename: f.lastPathComponent,
                        segments: [])
                    return "vecs=\(r.segments.count) dim=\(r.dimension)"
                }
                await probe("spk.window | \(f.lastPathComponent)") {
                    let r = try await s.windowEmbeddings(
                        audio: bytes(f), filename: f.lastPathComponent,
                        windowSeconds: 1.5, hopSeconds: 0.75)
                    return "wins=\(r.segments.count)"
                }
            }
        } else {
            mark("SKIP spk: load failed")
        }

        // Reaching here means no pipeline aborted the process on any file.
        mark("DONE — all pipelines survived the corpus")
    }
}
