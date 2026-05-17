import Foundation

/// Static configuration for the governed serve path.
public struct GovernorConfig: Sendable {
    /// The single global memory budget all modules share. Defaults to 75% of
    /// physical unified memory (the box also runs the OS, Vector, launchd).
    public var totalBudgetBytes: Int

    /// The one Athena listener. Default 7447 — Athena's own port, not Ollama's.
    /// Derivation: 7 = the Pythagorean heptad named "Athena" (the motherless,
    /// virgin number); 447 = 447 BCE, the year the Parthenon was begun.
    public var listenHost: String
    public var listenPort: Int

    /// Global ceiling on the KV/prompt-cache bytes a single request may
    /// need (brief item 4b). Owned here so it is part of the unified
    /// budget story, not a constant bolted onto the decode loop. The
    /// LLM module rejects an over-cap prompt with a governed 503.
    /// Defaults to ¼ of the total budget.
    public var promptCacheCapBytes: Int

    public static let defaultPort = 7447

    public init(
        totalBudgetBytes: Int? = nil,
        listenHost: String = "127.0.0.1",
        listenPort: Int = GovernorConfig.defaultPort,
        promptCacheCapBytes: Int? = nil
    ) {
        let budget =
            totalBudgetBytes
            ?? Int(Double(ProcessInfo.processInfo.physicalMemory) * 0.75)
        self.totalBudgetBytes = budget
        self.listenHost = listenHost
        self.listenPort = listenPort
        self.promptCacheCapBytes = promptCacheCapBytes ?? (budget / 4)
    }
}
