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

    public static let defaultPort = 7447

    public init(
        totalBudgetBytes: Int? = nil,
        listenHost: String = "127.0.0.1",
        listenPort: Int = GovernorConfig.defaultPort
    ) {
        self.totalBudgetBytes =
            totalBudgetBytes
            ?? Int(Double(ProcessInfo.processInfo.physicalMemory) * 0.75)
        self.listenHost = listenHost
        self.listenPort = listenPort
    }
}
