import Foundation

/// Identity of an in-process inference module. The governor owns module
/// lifecycle keyed by this identity; there is exactly one instance per id.
public enum ModuleID: String, Sendable, CaseIterable, Codable {
    case llm
    case transcription
    case textEmbedding
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
