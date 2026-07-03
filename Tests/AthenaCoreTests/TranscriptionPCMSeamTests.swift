import AthenaCore
import Foundation
import XCTest

@testable import AthenaTranscription

/// ADR 022 M78.1 S2 — the shared PCM transcription seam. Pins that
/// `transcribePCM` exists on the `TranscriptionModule` protocol (so the video
/// route can reach the engine the same way the audio route does, with no
/// re-encode) and is wired through the stub. The MLX equivalence of
/// `transcribe(audio:)` → `transcribePCM` is covered by the gated transcription
/// tests, which now funnel through this seam.
final class TranscriptionPCMSeamTests: XCTestCase {
    func testStubExposesPCMSeam() async throws {
        let m = StubTranscriptionModule()
        let r = try await m.transcribePCM(
            [0.0, 0.1, 0.2], language: "en", wordTimestamps: false)
        XCTAssertFalse(r.text.isEmpty)
        XCTAssertEqual(r.language, "en")
    }

    /// The seam is reachable polymorphically through the protocol existential —
    /// the shape the serve path holds.
    func testReachableViaProtocol() async throws {
        let m: any TranscriptionModule = StubTranscriptionModule()
        let r = try await m.transcribePCM(
            [], language: nil, wordTimestamps: true, requestedModel: nil)
        XCTAssertFalse(r.words.isEmpty)
    }

    /// ADR 029 audio residual (v0.10.252) — `requestedModel` is re-bound
    /// atomically with the forward, so a concurrent different-model rebind
    /// can't produce a wrong-model transcription. Pins the rebind-then-serve
    /// decision on the stub tier (ADR 009); nil/empty leaves the slot alone.
    func testRequestedModelRebindsAtomically() async throws {
        let m = StubTranscriptionModule(
            modelIds: ["whisper-a", "parakeet-b"],
            configuredDefault: "whisper-a")
        try await m.rebind(to: "whisper-a")
        _ = try await m.transcribePCM(
            [], language: nil, wordTimestamps: false,
            requestedModel: "parakeet-b")
        let bound = await m.residentModelId()
        XCTAssertEqual(bound, "parakeet-b")
        // nil / empty requestedModel must NOT touch the slot.
        _ = try await m.transcribePCM(
            [], language: nil, wordTimestamps: false, requestedModel: nil)
        _ = try await m.transcribePCM(
            [], language: nil, wordTimestamps: false, requestedModel: "")
        let still = await m.residentModelId()
        XCTAssertEqual(still, "parakeet-b")
    }
}
