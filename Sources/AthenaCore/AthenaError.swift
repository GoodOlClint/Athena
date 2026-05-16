import Foundation

/// Errors surfaced by the governor and module lifecycle. Each carries an HTTP
/// classification so the serve path never turns a memory-budget event into an
/// unhandled Metal abort — it becomes a classified 503 instead.
public enum AthenaError: Error, Sendable, Equatable {
    /// Admission was refused: the request would exceed the global budget and
    /// nothing evictable could be freed. Governed backpressure → 503.
    case memoryBudgetExceeded(requested: Int, available: Int, module: ModuleID)
    /// A module load failed in the substrate.
    case moduleLoadFailed(ModuleID, reason: String)
    /// No module is registered under this id.
    case moduleNotRegistered(ModuleID)

    /// HTTP status the serve path should return for this error.
    public var httpStatus: Int {
        switch self {
        case .memoryBudgetExceeded: return 503
        case .moduleLoadFailed: return 500
        case .moduleNotRegistered: return 404
        }
    }

    /// Stable machine-readable code (OpenAI-style error `code`).
    public var code: String {
        switch self {
        case .memoryBudgetExceeded: return "memory_budget_exceeded"
        case .moduleLoadFailed: return "module_load_failed"
        case .moduleNotRegistered: return "module_not_registered"
        }
    }

    public var message: String {
        switch self {
        case let .memoryBudgetExceeded(requested, available, module):
            return "Insufficient governed memory for \(module.rawValue): "
                + "requested \(requested) B, \(available) B available after eviction."
        case let .moduleLoadFailed(module, reason):
            return "Module \(module.rawValue) failed to load: \(reason)"
        case let .moduleNotRegistered(module):
            return "No module registered for \(module.rawValue)."
        }
    }
}
