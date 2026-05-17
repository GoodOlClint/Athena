import Foundation

/// Identity of an in-process inference module. The governor owns module
/// lifecycle keyed by this identity; there is exactly one instance per id.
public enum ModuleID: String, Sendable, CaseIterable, Codable {
    case llm
    case transcription
    case textEmbedding
    case diarization

    /// Friendly token for the unified-log category (`model.<token>`).
    public var logCategory: String {
        switch self {
        case .llm: return "llm"
        case .transcription: return "transcription"
        case .textEmbedding: return "embedding"
        case .diarization: return "diarization"
        }
    }
}

/// swift-log label convention, shared across targets as pure String
/// constants so AthenaCore stays dependency-free. The unified-log
/// bridge in the `athena` target maps `athena.<x>` → os.Logger
/// category `<x>` (`athena.daemon` → "daemon", `athena.model.llm` →
/// "model.llm", …).
public enum AthenaLogLabel {
    public static let daemon = "athena.daemon"
    public static func model(_ id: ModuleID) -> String {
        "athena.model.\(id.logCategory)"
    }
}

/// Lifecycle state of a module as tracked by the governor.
public enum ModuleState: String, Sendable, Codable {
    case unloaded
    case loading
    case loaded
    case unloading
}

/// A grant of governed memory issued by ``MemoryGovernor``. A module may only
/// hold resident bytes against a live reservation; releasing it returns the
/// bytes to the global budget.
public struct MemoryReservation: Sendable, Equatable {
    public let id: UUID
    public let module: ModuleID
    public let bytes: Int

    public init(id: UUID = UUID(), module: ModuleID, bytes: Int) {
        self.id = id
        self.module = module
        self.bytes = bytes
    }
}
