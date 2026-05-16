import AthenaCore
import Foundation

/// Typed inference contract for the LLM module. The MLX-backed
/// implementation lands in M1; this protocol is what the serve path holds.
public protocol LLMModule: InferenceModule {
    /// Stream generated text chunks. M0 is a canned stub; M1 replaces the
    /// body with native `TokenIterator` generation.
    nonisolated func generate(prompt: String) -> AsyncStream<String>
}

/// M0 governed stub. It holds no model — it exists to prove the thesis path
/// end-to-end: a request loads it through the governor (reserving real
/// budget), streams tokens over the API, and releases on unload.
public actor StubLLMModule: LLMModule {
    public nonisolated let id: ModuleID = .llm

    private let reserveBytes: Int
    private var loaded = false

    /// `reserveBytes` defaults to a representative multi-GB LLM footprint so
    /// budget pressure behaves realistically; tests inject small values.
    public init(reserveBytes: Int = 8 * 1024 * 1024 * 1024) {
        self.reserveBytes = reserveBytes
    }

    public var residentBytes: Int { loaded ? reserveBytes : 0 }

    public func memoryEstimate() -> Int { reserveBytes }

    public func load(reservation: MemoryReservation) async throws {
        loaded = true
    }

    public func unload() async {
        loaded = false
    }

    public nonisolated func generate(prompt: String) -> AsyncStream<String> {
        AsyncStream { continuation in
            let chunks = [
                "Athena ", "M0 ", "stub", ": ", "governed ", "memory ",
                "path ", "is ", "live", ".",
            ]
            let task = Task {
                for chunk in chunks {
                    continuation.yield(chunk)
                    try? await Task.sleep(nanoseconds: 15_000_000)
                }
                continuation.finish()
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }
}
